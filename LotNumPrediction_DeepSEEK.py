import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Bidirectional,
                                     BatchNormalization, Conv1D, Multiply, Softmax,
                                     Layer, Attention, Flatten, Reshape, RepeatVector,
                                     LayerNormalization, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import MultiHeadAttention
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
from typing import Tuple
from scipy.signal import savgol_filter

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


class TemporalAttention(Layer):
    """Temporal attention layer."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weights',
            shape=(input_shape[-1], 1),  # (features, 1)
            initializer='glorot_uniform'
        )
        self.b = self.add_weight(
            name='att_bias',
            shape=(input_shape[1],),  # (sequence length,)
            initializer='zeros'
        )
        super().build(input_shape)

    def call(self, x):
        # Apply linear transformation to get attention scores
        e = tf.squeeze(tf.matmul(x, self.W), axis=-1) + self.b  # Shape: (batch, seq)
        a = tf.nn.softmax(e, axis=1)  # Softmax over sequence length
        a = tf.expand_dims(a, axis=-1)  # Expand for broadcasting, (batch, seq, 1)
        weighted = x * a  # Apply attention weights
        return tf.reduce_sum(weighted, axis=1)  # Summing over sequence dimension

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])  # (batch, features)



class EnhancedNumberPredictor:
    def __init__(self, sequence_length=15, forecast_horizon=5):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = [
            'value', 'mean', 'std', 'min', 'max', 'median',
            'q25', 'q75', 'diff_mean', 'diff_std', 'ema_3', 'ema_7',
            'momentum', 'seasonal'
        ]

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Advanced preprocessing with adaptive smoothing"""
        data = np.array(data, dtype=np.float32)

        # Seasonal smoothing with Savitzky-Golay filter
        data = savgol_filter(data, window_length=11, polyorder=2)

        # Adaptive outlier detection using Hampel filter
        median = pd.Series(data).rolling(30, min_periods=1).median().values
        mad = pd.Series(np.abs(data - median)).rolling(30, min_periods=1).median().values
        threshold = 3 * 1.4826 * mad
        data = np.where(np.abs(data - median) > threshold, median, data)

        return np.clip(data, 1, 50).astype(int)

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences with correct temporal feature calculation"""
        X, y = [], []
        df = pd.DataFrame({'value': data})

        # Precompute global features without lookahead
        df['diff'] = df['value'].diff().fillna(0)
        df['ema_3'] = df['value'].ewm(span=3, adjust=False).mean()
        df['ema_7'] = df['value'].ewm(span=7, adjust=False).mean()
        df['momentum'] = df['value'].pct_change(periods=3).fillna(0)
        df['seasonal'] = df['value'].diff(periods=7).fillna(0)

        for i in range(len(df) - self.sequence_length - self.forecast_horizon + 1):
            window_start = i
            window_end = i + self.sequence_length
            target_start = window_end
            target_end = target_start + self.forecast_horizon

            X_window = []
            for j in range(window_start, window_end):
                # Calculate statistics using only data up to current point
                historical_data = df.iloc[max(0, j - 29):j + 1]

                stats = {
                    'mean': historical_data['value'].mean(),
                    'std': historical_data['value'].std(),
                    'min': historical_data['value'].min(),
                    'max': historical_data['value'].max(),
                    'median': historical_data['value'].median(),
                    'q25': historical_data['value'].quantile(0.25),
                    'q75': historical_data['value'].quantile(0.75),
                    'diff_mean': historical_data['diff'].mean(),
                    'diff_std': historical_data['diff'].std()
                }

                features = [
                    df['value'].iloc[j],
                    stats['mean'],
                    stats['std'],
                    stats['min'],
                    stats['max'],
                    stats['median'],
                    stats['q25'],
                    stats['q75'],
                    stats['diff_mean'],
                    stats['diff_std'],
                    df['ema_3'].iloc[j],
                    df['ema_7'].iloc[j],
                    df['momentum'].iloc[j],
                    df['seasonal'].iloc[j]
                ]
                X_window.append(features)

            X.append(X_window)
            y.append(df['value'].iloc[target_start:target_end].values)

        return np.array(X), np.array(y)

    def build_dual_attention_model(self, input_shape: Tuple[int, int]) -> Model:
        """Enhanced architecture with transformer-inspired components"""
        inputs = Input(shape=input_shape)

        # Multi-scale feature extraction
        conv1 = Conv1D(128, 3, activation='relu', padding='causal')(inputs)
        conv2 = Conv1D(128, 5, activation='relu', padding='causal')(inputs)
        conv3 = Conv1D(128, 7, activation='relu', padding='causal')(inputs)
        x = Concatenate()([conv1, conv2, conv3])
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Temporal processing with skip connections
        lstm_out = Bidirectional(LSTM(256, return_sequences=True))(x)
        lstm_out = Bidirectional(LSTM(256, return_sequences=True))(lstm_out)

        # Temporal attention
        temp_att = TemporalAttention()(lstm_out)

        # Feature attention
        feat_att = Dense(256, activation='swish')(temp_att)
        feat_att = LayerNormalization()(feat_att)

        # Ensure feat_att has a 3D shape before Attention layer
        feat_att = tf.expand_dims(feat_att, axis=1)  # Convert (batch, features) → (batch, 1, features)

        # Apply Attention
        feat_att = Attention()([feat_att, feat_att])  # Ensure both inputs are valid Keras Tensors
        feat_att = tf.squeeze(feat_att, axis=1)  # Convert back to (batch, features) after Attention

        # Multi-horizon forecasting
        x = Concatenate()([temp_att, feat_att])
        x = RepeatVector(self.forecast_horizon)(x)
        x = LSTM(512, return_sequences=True)(x)
        x = Dense(256, activation='swish')(x)
        outputs = Dense(1)(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0003, clipnorm=1.0),
            loss='huber',
            metrics=['mae', 'mse']
        )
        return model

    def temporal_train_test_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple:
        split_idx = int(len(X) * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_train, X_val, y_train, y_val = self.temporal_train_test_split(X, y)

        # Feature-wise scaling
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_flat)
        X_train = self.scaler.transform(X_train_flat).reshape(X_train.shape)
        X_val = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        self.model = self.build_dual_attention_model((self.sequence_length, len(self.feature_names)))

        callbacks = [
            EarlyStopping(patience=25, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'), save_best_only=True),
            TensorBoard(log_dir=log_dir)
        ]

        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=300,
            batch_size=128,
            callbacks=callbacks,
            verbose=1
        )
        self._analyze_performance(X_val, y_val)

    def _analyze_performance(self, X: np.ndarray, y: np.ndarray) -> None:
        y_pred = self.model.predict(X).reshape(-1, self.forecast_horizon)

        print("\n=== Forecasting Performance Analysis ===")
        for horizon in range(self.forecast_horizon):
            y_true = y[:, horizon]
            y_pred_h = y_pred[:, horizon]

            mae = mean_absolute_error(y_true, y_pred_h)
            mse = mean_squared_error(y_true, y_pred_h)
            accuracy = np.mean(np.isclose(y_true, y_pred_h, atol=1.5))

            print(f"Horizon {horizon + 1}:")
            print(f"  MAE: {mae:.2f}  MSE: {mse:.2f}")
            print(f"  Accuracy (±1.5): {accuracy:.2%}")

    def predict_next_numbers(self, last_sequence: np.ndarray, num_predictions: int = 5) -> list:
        if num_predictions > self.forecast_horizon:
            raise ValueError(f"Model trained for {self.forecast_horizon} steps maximum")

        # Prepare input with proper feature calculation
        df = pd.DataFrame({'value': last_sequence})
        df['diff'] = df['value'].diff().fillna(0)
        df['ema_3'] = df['value'].ewm(span=3, adjust=False).mean()
        df['ema_7'] = df['value'].ewm(span=7, adjust=False).mean()
        df['momentum'] = df['value'].pct_change(periods=3).fillna(0)
        df['seasonal'] = df['value'].diff(periods=7).fillna(0)

        X_window = []
        for j in range(len(df)):
            historical_data = df.iloc[max(0, j - 29):j + 1]

            stats = {
                'mean': historical_data['value'].mean(),
                'std': historical_data['value'].std(),
                'min': historical_data['value'].min(),
                'max': historical_data['value'].max(),
                'median': historical_data['value'].median(),
                'q25': historical_data['value'].quantile(0.25),
                'q75': historical_data['value'].quantile(0.75),
                'diff_mean': historical_data['diff'].mean(),
                'diff_std': historical_data['diff'].std()
            }

            features = [
                df['value'].iloc[j],
                stats['mean'],
                stats['std'],
                stats['min'],
                stats['max'],
                stats['median'],
                stats['q25'],
                stats['q75'],
                stats['diff_mean'],
                stats['diff_std'],
                df['ema_3'].iloc[j],
                df['ema_7'].iloc[j],
                df['momentum'].iloc[j],
                df['seasonal'].iloc[j]
            ]
            X_window.append(features)

        X = np.array([X_window])
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        # Probabilistic predictions with MC dropout
        predictions = []
        for _ in range(500):
            pred = self.model.predict(X_scaled, verbose=0, training=True)[0]
            predictions.extend(pred[:, 0])

        predictions = np.clip(predictions, 1, 50)
        rounded = np.round(predictions).astype(int)
        unique, counts = np.unique(rounded, return_counts=True)
        top_indices = np.argsort(-counts)[:num_predictions]

        return sorted(unique[top_indices].tolist())


def main():
    try:
        raw_data = pd.read_csv('data.txt', sep=r'\s+', header=None).values.flatten()
        predictor = EnhancedNumberPredictor(sequence_length=15, forecast_horizon=5)
        processed_data = predictor.preprocess_data(raw_data)

        X, y = predictor.create_sequences(processed_data)
        if len(X) == 0:
            raise ValueError("Insufficient data for sequence generation")

        predictor.fit(X, y)

        final_sequence = processed_data[-predictor.sequence_length:]
        predictions = predictor.predict_next_numbers(final_sequence)

        print("\n=== Final Prediction Results ===")
        print("Most probable numbers:", predictions)

    except Exception as e:
        print(f"\nProcessing error: {str(e)}")


if __name__ == "__main__":
    main()
