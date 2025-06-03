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
        self.scaler_main = StandardScaler()
        self.scaler_bonus = StandardScaler()
        self.model_main = None
        self.model_bonus = None
        self.feature_names = [
            'mean', 'std', 'min', 'max', 'median',
            'q25', 'q75', 'diff_mean', 'diff_std'
        ]

    def preprocess_data(self, raw_data):
        """Split and clip main and bonus number arrays."""
        main_numbers = raw_data[:, :5]
        bonus_numbers = raw_data[:, 5:]

        main_numbers = np.clip(main_numbers, 1, 50)
        bonus_numbers = np.clip(bonus_numbers, 1, 12)

        return main_numbers.astype(int), bonus_numbers.astype(int)

    def create_sequences(self, data):
        """Create sequences for time-series prediction."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            window = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]

            # Window statistics
            window_features = [
                np.mean(window), np.std(window),
                np.min(window), np.max(window),
                np.median(window), np.percentile(window, 25),
                np.percentile(window, 75), np.diff(window).mean(),
                np.diff(window).std()
            ]

            # 3D input with features
            window_array = []
            for value in window:
                window_array.append([value] + window_features)

            X.append(window_array)
            y.append(target)

        return np.array(X), np.array(y)

    def build_temporal_model(self, input_shape):
        """Build temporal attention model."""
        inputs = Input(shape=input_shape)

        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        attention = Dense(1, activation='tanh')(x)
        attention = Softmax(axis=1)(attention)
        context = Multiply()([x, attention])
        context = ReduceSumLayer()(context)

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

    def fit(self, X, y, kind="main"):
        """Train model based on kind (main or bonus)."""
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])

        scaler = self.scaler_main if kind == "main" else self.scaler_bonus
        X_scaled = scaler.fit_transform(X_flat)
        X = X_scaled.reshape(original_shape)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = self.build_temporal_model((self.sequence_length, len(self.feature_names) + 1))

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(model_dir, f'best_model_{kind}.keras'),
                save_best_only=True,
                monitor='val_loss'
            ),
            TensorBoard(log_dir=os.path.join(log_dir, kind))
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

        y_pred = model.predict(X_val).flatten()
        print(f"\nValidation ({kind}) MAE: {mean_absolute_error(y_val, y_pred):.2f}")
        print(f"Validation ({kind}) MSE: {mean_squared_error(y_val, y_pred):.2f}")

        if kind == "main":
            self.model_main = model
        else:
            self.model_bonus = model

    def predict_next_numbers(self, last_sequence, num_predictions=5, upper_bound=50, kind="main"):
        """Predict unique numbers for main or bonus."""
        predictions = set()
        current_sequence = last_sequence.copy()
        scaler = self.scaler_main if kind == "main" else self.scaler_bonus
        model = self.model_main if kind == "main" else self.model_bonus

        for _ in range(num_predictions * 3):
            if len(predictions) >= num_predictions:
                break

            window_features = [
                np.mean(current_sequence), np.std(current_sequence),
                np.min(current_sequence), np.max(current_sequence),
                np.median(current_sequence), np.percentile(current_sequence, 25),
                np.percentile(current_sequence, 75), np.diff(current_sequence).mean(),
                np.diff(current_sequence).std()
            ]

            sequence_data = [[val] + window_features for val in current_sequence]
            X = np.array([sequence_data])

            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = scaler.transform(X_flat)
            X_scaled = X_scaled.reshape(X.shape)

            pred = model.predict(X_scaled, verbose=0)[0][0]
            processed_pred = int(np.clip(round(pred), 1, upper_bound))

            predictions.add(processed_pred)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = processed_pred

        while len(predictions) < num_predictions:
            predictions.add(np.random.randint(1, upper_bound + 1))

        return sorted(predictions)[:num_predictions]


def main():
    try:
        # Load data
        raw_data = pd.read_csv('data.txt', sep=r'\s+', header=None).values

        predictor = EnhancedNumberPredictor(sequence_length=15)
        main_data, bonus_data = predictor.preprocess_data(raw_data)

        main_X, main_y = predictor.create_sequences(main_data.flatten())
        bonus_X, bonus_y = predictor.create_sequences(bonus_data.flatten())

        if len(main_X) == 0 or len(bonus_X) == 0:
            raise ValueError("Insufficient data for training.")

        print("Training main number model...")
        predictor.fit(main_X, main_y, kind="main")

        print("Training bonus ball model...")
        predictor.fit(bonus_X, bonus_y, kind="bonus")

        # Predictions
        final_main_sequence = main_data.flatten()[-predictor.sequence_length:]
        final_bonus_sequence = bonus_data.flatten()[-predictor.sequence_length:]

        predicted_main = predictor.predict_next_numbers(
            final_main_sequence, num_predictions=5, upper_bound=50, kind="main")
        predicted_bonus = predictor.predict_next_numbers(
            final_bonus_sequence, num_predictions=2, upper_bound=12, kind="bonus")

        print("\n=== Prediction Results ===")
        print("Predicted Main Numbers:", predicted_main)
        print("Predicted Bonus Balls :", predicted_bonus)

    except Exception as e:
        print(f"\nError in processing pipeline: {str(e)}")


if __name__ == "__main__":
    main()
