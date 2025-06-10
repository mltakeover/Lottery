import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Bidirectional,
                                     BatchNormalization, Conv1D, Multiply, Softmax, Layer)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import datetime
import warnings

# Configure reproducibility and suppress warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# Directory setup
log_dir = "ml_logs"
model_dir = "saved_models"
plots_dir = "analysis_plots"
for directory in [log_dir, model_dir, plots_dir]:
    os.makedirs(directory, exist_ok=True)


class ReduceSumLayer(Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)


class EnhancedNumberPredictor:
    def __init__(self, sequence_length=15):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = [
            'mean', 'std', 'min', 'max', 'median',
            'q25', 'q75', 'diff_mean', 'diff_std'
        ]

    def preprocess_data(self, data):
        """Clean and normalize input data with robust outlier removal."""
        data = np.array(data, dtype=np.float32)

        # Handle outliers using IQR method
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered = data[(data >= lower_bound) & (data <= upper_bound)]
        processed = np.clip(filtered, 1, 50)
        return np.round(processed).astype(int)

    def create_sequences(self, data):
        """Create time-series sequences with enhanced feature engineering."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            window = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]

            # Calculate window statistics
            window_features = [
                np.mean(window), np.std(window),
                np.min(window), np.max(window),
                np.median(window), np.percentile(window, 25),
                np.percentile(window, 75), np.diff(window).mean(),
                np.diff(window).std()
            ]

            # Create 3D input with features
            window_array = []
            for value in window:
                window_array.append([value] + window_features)

            X.append(window_array)
            y.append(target)

        return np.array(X), np.array(y)

    def build_temporal_model(self, input_shape):
        """Construct optimized neural architecture with temporal attention."""
        inputs = Input(shape=input_shape)

        # Feature extraction layer
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Temporal processing
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Attention mechanism
        attention = Dense(1, activation='tanh')(x)
        attention = Softmax(axis=1)(attention)
        context = Multiply()([x, attention])

        # Use custom layer for reduce_sum
        context = ReduceSumLayer()(context)

        # Prediction head
        x = Dense(64, activation='relu')(context)
        x = Dropout(0.3)(x)
        outputs = Dense(1)(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        return model

    def fit(self, X, y):
        """Train model with comprehensive data processing pipeline."""
        # Reshape and scale data
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_flat)
        X = X_scaled.reshape(original_shape)

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model initialization
        self.model = self.build_temporal_model((self.sequence_length, len(self.feature_names) + 1))

        # Training callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(model_dir, 'best_model.keras'),
                save_best_only=True,
                monitor='val_loss'
            ),
            TensorBoard(log_dir=log_dir)
        ]

        # Model training
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

        # Model evaluation
        y_pred = self.model.predict(X_val).flatten()
        print(f"\nValidation MAE: {mean_absolute_error(y_val, y_pred):.2f}")
        print(f"Validation MSE: {mean_squared_error(y_val, y_pred):.2f}")

    def predict_next_numbers(self, last_sequence, num_predictions=5):
        """Generate unique predictions using iterative refinement."""
        predictions = set()
        current_sequence = last_sequence.copy()

        for _ in range(num_predictions * 3):  # Allow limited attempts
            if len(predictions) >= num_predictions:
                break

            # Calculate current features
            window_features = [
                np.mean(current_sequence), np.std(current_sequence),
                np.min(current_sequence), np.max(current_sequence),
                np.median(current_sequence), np.percentile(current_sequence, 25),
                np.percentile(current_sequence, 75), np.diff(current_sequence).mean(),
                np.diff(current_sequence).std()
            ]

            # Prepare input tensor
            sequence_data = []
            for value in current_sequence:
                sequence_data.append([value] + window_features)
            X = np.array([sequence_data])

            # Scale and predict
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_flat)
            X_scaled = X_scaled.reshape(X.shape)

            pred = self.model.predict(X_scaled, verbose=0)[0][0]
            processed_pred = int(np.clip(round(pred), 1, 50))

            # Update sequence state
            predictions.add(processed_pred)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = processed_pred

        # Ensure full prediction set
        while len(predictions) < num_predictions:
            predictions.add(np.random.randint(1, 51))

        return sorted(predictions)[:num_predictions]


def main():
    try:

        # Data loading and preparation
        raw_data = pd.read_csv('data.txt', sep=r'\s+', header=None).values.flatten()

        predictor = EnhancedNumberPredictor(sequence_length=15)
        processed_data = predictor.preprocess_data(raw_data)

        X, y = predictor.create_sequences(processed_data)
        if len(X) == 0:
            raise ValueError("Insufficient data for sequence generation")

        # Model training
        predictor.fit(X, y)

        # Generate predictions
        final_sequence = processed_data[-predictor.sequence_length:]
        predictions = predictor.predict_next_numbers(final_sequence)

        print("\n=== Prediction Results ===")
        print("Top predicted numbers:", predictions)

    except Exception as e:
        print(f"\nError in processing pipeline: {str(e)}")


if __name__ == "__main__":
    main()
