import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization, Add, Concatenate, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import datetime
import warnings

# Suppress warnings and set random seeds for reproducibility
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# Set directories
log_dir = "ml_logs"
model_dir = "saved_models"
plots_dir = "analysis_plots"
for directory in [log_dir, model_dir, plots_dir]:
    os.makedirs(directory, exist_ok=True)

class AdvancedNumberPredictor:
    def __init__(self, sequence_length=8965):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.lstm_model = None

    def preprocess_data(self, data):
        # Convert data to numpy array and remove outliers
        data = np.array(data)
        Q1, Q3 = np.percentile(data, 25), np.percentile(data, 75)
        IQR = Q3 - Q1
        data = data[(data >= (Q1 - 1.5 * IQR)) & (data <= (Q3 + 1.5 * IQR))]
        return np.clip(np.round(data), 1, 50)  # Clip data to valid range [1, 50]

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]

            # Feature extraction
            seq_features = [
                np.mean(sequence), np.std(sequence), np.min(sequence), np.max(sequence),
                np.median(sequence), np.percentile(sequence, 25), np.percentile(sequence, 75),
                np.diff(sequence).mean(), np.diff(sequence).std()
            ]

            X.append(np.concatenate([sequence, seq_features]))
            y.append(target)

        X, y = np.array(X), np.array(y)
        return X, y

    def build_lstm_model(self, input_shape):
        inputs = Input(shape=input_shape)

        # Convolutional layer for feature extraction
        conv = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        conv = BatchNormalization()(conv)

        # Stacked LSTM layers with attention mechanism
        lstm1 = Bidirectional(LSTM(4092, return_sequences=True))(conv)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(0.3)(lstm1)

        lstm2 = Bidirectional(LSTM(1024, return_sequences=True))(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(0.3)(lstm2)

        lstm3 = Bidirectional(LSTM(256, return_sequences=True))(lstm2)
        lstm3 = BatchNormalization()(lstm3)
        lstm3 = Dropout(0.3)(lstm3)


        attention = Dense(1, activation='sigmoid')(lstm3)
        attended_lstm = Concatenate()([lstm2, attention])

        x = Bidirectional(LSTM(16))(attended_lstm)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1)(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae', 'mse'])
        return model

    def fit(self, X, y):
        # Scale the features and reshape for LSTM input
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Build and train the LSTM model
        self.lstm_model = self.build_lstm_model(input_shape=(X_train.shape[1], 1))
        checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'), save_best_only=True,
                                     monitor='val_loss')
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

        self.lstm_model.fit(
            X_train, y_train, epochs=10, batch_size=64, validation_split=0.2,
            callbacks=[early_stopping, checkpoint], verbose=1
        )

        # Evaluate on the test set
        y_pred = self.lstm_model.predict(X_test).flatten()
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"LSTM Model - MAE: {mae:.4f}, MSE: {mse:.4f}")

    def predict_next_numbers(self, last_sequence, num_predictions=5):
        predictions = set()  # Use a set to store unique predictions
        attempts = 0
        max_attempts = 50  # Maximum number of attempts to get unique numbers

        while len(predictions) < num_predictions and attempts < max_attempts:
            # Prepare input sequence
            seq_features = [
                np.mean(last_sequence), np.std(last_sequence), np.min(last_sequence), np.max(last_sequence),
                np.median(last_sequence), np.percentile(last_sequence, 25), np.percentile(last_sequence, 75),
                np.diff(last_sequence).mean(), np.diff(last_sequence).std()
            ]

            full_sequence = np.concatenate([last_sequence, seq_features])
            full_sequence = self.scaler.transform(full_sequence.reshape(1, -1))
            full_sequence = full_sequence.reshape((1, full_sequence.shape[1], 1))

            # Predict the next number
            pred = self.lstm_model.predict(full_sequence, verbose=0)[0, 0]
            rounded_pred = int(round(np.clip(pred, 1, 50)))  # Ensure prediction is between 1 and 50

            # Add prediction if it's unique
            if rounded_pred not in predictions:
                predictions.add(rounded_pred)
                last_sequence = np.append(last_sequence[1:], pred)

            attempts += 1

        # If we don't have enough unique predictions, fill remaining slots
        while len(predictions) < num_predictions:
            new_num = np.random.randint(1, 51)  # Generate random number between 1 and 50
            if new_num not in predictions:
                predictions.add(new_num)

        # Convert set to sorted list
        return sorted(list(predictions))


def main():
    try:
        # Read data more flexibly by handling both tabs and spaces
        data = pd.read_csv('data.txt', sep=r'\s+', header=None, dtype=float).values.flatten()

        predictor = AdvancedNumberPredictor(sequence_length=15)
        cleaned_data = predictor.preprocess_data(data)
        X, y = predictor.create_sequences(cleaned_data)

        if len(X) == 0 or len(y) == 0:
            print("Error: No valid sequences could be created from the data.")
            return

        predictor.fit(X, y)
        last_sequence = cleaned_data[-predictor.sequence_length:]

        # Get unique predictions
        predictions = predictor.predict_next_numbers(last_sequence)

        print("\n=== Final Predictions ===")
        print("Next 5 numbers:", predictions)

    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main()