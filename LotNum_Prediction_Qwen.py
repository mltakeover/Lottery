#!/usr/bin/env python3
"""
Enhanced Lottery Number Prediction Script
This script analyzes historical lottery data to predict the next set of numbers
using advanced feature engineering, multiple ML models, and optimized ensemble methods.
It predicts 5 unique main numbers in the range of 1-50 and 2 unique bonus numbers in the
range of 1-12, based on sophisticated pattern analysis and machine learning techniques.

**Important Note:** Lottery draws are random. This script identifies patterns in historical
data to make *educated guesses*. It cannot predict future draws with 100% accuracy.
"""
import numpy as np
import pandas as pd
from collections import Counter
import random
import os
import sys
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import xgboost as xgb
import lightgbm as lgb
import configparser

# --- Configuration and Logging ---
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Default configuration
DEFAULT_CONFIG = {
    'SETTINGS': {
        'window_size': '5',
        'main_num_range': '50',
        'bonus_num_range': '12',
        'num_main_to_predict': '5',
        'num_bonus_to_predict': '2',
        'test_size': '0.2',
        'n_splits': '3',
        'date_format': '%Y-%m-%d' # Added for configurable date parsing
    }
}

# --- Global Variables ---
# Global variables to store feature dimensions for consistency
MAIN_FEATURE_DIM = None
BONUS_FEATURE_DIM = None

# --- Configuration Handling ---
def get_config_value(config, section, key, value_type, default):
    """Safely retrieves and converts a configuration value."""
    try:
        if value_type == int:
            return config[section].getint(key)
        elif value_type == float:
            return config[section].getfloat(key)
        else: # string or other
            return config[section].get(key)
    except (ValueError, KeyError) as e:
        logging.warning(f"Invalid or missing config for {section}.{key}, using default: {default}. Error: {e}")
        return default

def create_default_config():
    """Create a default configuration file if it doesn't exist."""
    if not os.path.exists("config.ini"):
        config = configparser.ConfigParser()
        config['SETTINGS'] = DEFAULT_CONFIG['SETTINGS']
        with open("config.ini", 'w') as configfile:
            config.write(configfile)
        logging.info("Created default config.ini file")
    return configparser.ConfigParser()

# Load configuration
config_parser = create_default_config()
config_parser.read("config.ini")

# --- File and Data Loading ---
def get_data_file_path():
    """Get the path to the data file. Assumes data.txt is in the script's directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data.txt")
    if not os.path.isfile(file_path):
        logging.error(f"data.txt not found in script directory: {script_dir}")
        logging.error("Please ensure 'data.txt' is in the same directory as the script.")
        sys.exit(1)
    return file_path

def load_data(file_path):
    """Load and parse the data file containing number selections with enhanced validation."""
    main_number_sets = []
    bonus_number_sets = []
    processed_lines = set()  # For deduplication
    dates = []  # Optional: If dates are available in the data
    date_format = get_config_value(config_parser, 'SETTINGS', 'date_format', str, DEFAULT_CONFIG['SETTINGS']['date_format'])

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line_num, line in enumerate(lines):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            if stripped_line in processed_lines:
                logging.warning(f"Skipping duplicate line {line_num + 1}: {stripped_line}")
                continue
            processed_lines.add(stripped_line)
            try:
                # Assuming format: [date (optional)] num1 num2 num3 num4 num5 bonus1 bonus2
                parts = stripped_line.split()
                # Check if first part might be a date (simple heuristic)
                has_date = False
                if len(parts) > 7 and not parts[0].isdigit():
                    try:
                        # Attempt to parse as date using the configurable format
                        date_str = parts[0]
                        datetime.strptime(date_str, date_format) # Validate format
                        dates.append(date_str)
                        parts = parts[1:]  # Remove date part
                        has_date = True
                    except ValueError:
                        # If parsing fails, treat as part of numbers or log
                        logging.debug(f"Could not parse '{parts[0]}' as date with format '{date_format}'. Treating as data.")
                        pass  # Not a date, continue with normal processing
                numbers = [int(num) for num in parts if num.isdigit()]
                if len(numbers) == 7:  # Expecting 5 main and 2 bonus numbers
                    main_nums = numbers[:5]
                    bonus_nums = numbers[5:]
                    # Validate ranges
                    main_num_range = get_config_value(config_parser, 'SETTINGS', 'main_num_range', int, 50)
                    bonus_num_range = get_config_value(config_parser, 'SETTINGS', 'bonus_num_range', int, 12)
                    main_valid = all(1 <= num <= main_num_range for num in main_nums)
                    bonus_valid = all(1 <= num <= bonus_num_range for num in bonus_nums)
                    if not main_valid:
                        logging.warning(
                            f"Main numbers out of range (1-{main_num_range}) in line {line_num + 1}: {main_nums}")
                        continue
                    if not bonus_valid:
                        logging.warning(
                            f"Bonus numbers out of range (1-{bonus_num_range}) in line {line_num + 1}: {bonus_nums}")
                        continue
                    # Check for duplicates within the same draw
                    if len(set(main_nums)) != len(main_nums):
                        logging.warning(f"Duplicate main numbers in line {line_num + 1}: {main_nums}")
                        continue
                    if len(set(bonus_nums)) != len(bonus_nums):
                        logging.warning(f"Duplicate bonus numbers in line {line_num + 1}: {bonus_nums}")
                        continue
                    main_number_sets.append(main_nums)
                    bonus_number_sets.append(bonus_nums)
                    # If no date was found but we need one for time features, add a placeholder
                    if not has_date:
                        dates.append(None)
                else:
                    logging.warning(
                        f"Skipping line {line_num + 1} due to incorrect number count (expected 7): {stripped_line}")
            except ValueError:
                logging.error(f"Invalid number format in line {line_num + 1}: {stripped_line}")
                continue
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        logging.error("Please ensure 'data.txt' is in the same directory as the script.")
        sys.exit(1)  # Exit if file not found
    return main_number_sets, bonus_number_sets, dates


# --- Feature Engineering ---
def create_enhanced_features(number_sets, window_size, is_main=True, dates=None):
    """Create advanced features for machine learning prediction with sophisticated feature engineering."""
    global MAIN_FEATURE_DIM, BONUS_FEATURE_DIM
    if len(number_sets) <= window_size:
        return np.array([]), np.array([])

    features = []
    targets = []
    # Determine the number range based on whether we're processing main or bonus numbers
    num_range = get_config_value(config_parser, 'SETTINGS', 'main_num_range', int, 50) if is_main else get_config_value(config_parser, 'SETTINGS', 'bonus_num_range', int, 12)
    date_format = get_config_value(config_parser, 'SETTINGS', 'date_format', str, DEFAULT_CONFIG['SETTINGS']['date_format'])

    for i in range(len(number_sets) - window_size):
        window = number_sets[i:i + window_size]
        feature_vector = []
        # 1. Basic features (flattened window)
        basic_features = [num for subset in window for num in subset]
        feature_vector.extend(basic_features)
        # 2. Statistical features
        flat_window = np.array(window).flatten()
        feature_vector.append(np.mean(flat_window))
        feature_vector.append(np.std(flat_window))
        feature_vector.append(np.median(flat_window))
        feature_vector.append(np.min(flat_window))
        feature_vector.append(np.max(flat_window))
        feature_vector.append(np.percentile(flat_window, 25))
        feature_vector.append(np.percentile(flat_window, 75))
        # 3. Frequency-based features - using fixed range
        counter = Counter(flat_window)
        frequencies = [counter.get(num, 0) for num in range(1, num_range + 1)]
        feature_vector.extend(frequencies)
        # 4. Pattern-based features
        # 4.1 Consecutive numbers
        consecutive_count = 0
        for j in range(len(flat_window) - 1):
            if flat_window[j] + 1 == flat_window[j + 1]:
                consecutive_count += 1
        feature_vector.append(consecutive_count)
        # 4.2 Even/Odd ratio
        even_count = sum(1 for num in flat_window if num % 2 == 0)
        odd_count = len(flat_window) - even_count
        feature_vector.append(even_count / len(flat_window) if len(flat_window) > 0 else 0)
        # 4.3 Low/High ratio (numbers below/above half of range)
        mid_point = num_range // 2
        low_count = sum(1 for num in flat_window if num <= mid_point) # Changed to <= for better split
        feature_vector.append(low_count / len(flat_window) if len(flat_window) > 0 else 0)
        # 5. Lag features (previous draws)
        for j in range(window_size):
            feature_vector.extend(window[j])
        # 6. Difference features
        for j in range(window_size - 1):
            diffs = [window[j + 1][k] - window[j][k] for k in range(len(window[j]))]
            feature_vector.extend(diffs)
        # 7. Time-based features (if dates are available)
        if dates and i < len(dates) and dates[i] is not None:
            try:
                date = datetime.strptime(dates[i], date_format) # Use configurable format
                feature_vector.append(date.weekday())  # Day of week (0-6)
                feature_vector.append(date.day)  # Day of month
                feature_vector.append(date.month)  # Month
                feature_vector.append(date.year)  # Year
            except ValueError:
                # Add placeholder values if date parsing fails
                logging.warning(f"Failed to parse date '{dates[i]}' with format '{date_format}'. Adding placeholders.")
                feature_vector.extend([0, 0, 0, 0])
        else:
            # Add placeholder values if no dates
            feature_vector.extend([0, 0, 0, 0])
        target = number_sets[i + window_size]
        features.append(feature_vector)
        targets.append(target)

    # Convert to numpy arrays
    features_array = np.array(features)
    targets_array = np.array(targets)

    # Store feature dimensions for consistency
    if is_main:
        MAIN_FEATURE_DIM = features_array.shape[1]
    else:
        BONUS_FEATURE_DIM = features_array.shape[1]

    logging.debug(f"Created features for {'main' if is_main else 'bonus'} numbers. Shape: {features_array.shape}")
    return features_array, targets_array


# --- Model Training and Evaluation ---
def train_traditional_model(features, targets, model_type="xgboost"):
    """Train a traditional machine learning model with optimized hyperparameters."""
    if features.size == 0 or targets.size == 0 or len(features) < 2:
        logging.warning(f"Insufficient samples ({len(features)}) for {model_type} model training.")
        return [], None

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    models = []
    # Dynamically adjust hyperparameters based on dataset size
    dataset_size = len(features)
    small_dataset_threshold = 50 # Example threshold

    for i in range(targets.shape[1]):
        try:
            if model_type == "xgboost":
                # Example: simpler model for smaller datasets
                if dataset_size < small_dataset_threshold:
                    model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        n_estimators=50, # Reduced
                        max_depth=3,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        random_state=42
                    )
                else:
                    model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        n_estimators=100,
                        max_depth=6, # Increased for larger datasets
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        random_state=42
                    )
            elif model_type == "lightgbm":
                 if dataset_size < small_dataset_threshold:
                     model = lgb.LGBMRegressor(
                         n_estimators=50, # Reduced
                         learning_rate=0.1,
                         num_leaves=15, # Reduced
                         max_depth=3, # Reduced
                         subsample=0.8,
                         colsample_bytree=0.8,
                         reg_alpha=0.1,
                         reg_lambda=1.0,
                         min_child_samples=2,
                         random_state=42
                     )
                 else:
                     model = lgb.LGBMRegressor(
                         n_estimators=100,
                         learning_rate=0.1,
                         num_leaves=31, # Increased
                         max_depth=-1, # Let LightGBM decide
                         subsample=0.8,
                         colsample_bytree=0.8,
                         reg_alpha=0.1,
                         reg_lambda=1.0,
                         min_child_samples=20, # Increased
                         random_state=42
                     )
            elif model_type == "random_forest":
                 if dataset_size < small_dataset_threshold:
                     model = RandomForestRegressor(
                         n_estimators=50, # Reduced
                         max_depth=3, # Reduced
                         min_samples_split=2,
                         min_samples_leaf=1,
                         random_state=42
                     )
                 else:
                     model = RandomForestRegressor(
                         n_estimators=100,
                         max_depth=None, # No limit
                         min_samples_split=2,
                         min_samples_leaf=1,
                         random_state=42
                     )
            elif model_type == "gradient_boosting":
                 if dataset_size < small_dataset_threshold:
                     model = GradientBoostingRegressor(
                         n_estimators=50, # Reduced
                         learning_rate=0.1,
                         max_depth=3, # Reduced
                         subsample=0.8,
                         random_state=42
                     )
                 else:
                     model = GradientBoostingRegressor(
                         n_estimators=100,
                         learning_rate=0.1,
                         max_depth=3,
                         subsample=0.8,
                         random_state=42
                     )
            elif model_type == "bayesian_ridge":
                model = BayesianRidge(
                    alpha_1=1e-6,
                    alpha_2=1e-6,
                    lambda_1=1e-6,
                    lambda_2=1e-6
                )
            elif model_type == "gaussian_process":
                # Add a check to skip GP for very large datasets
                gp_size_threshold = 100 # Example threshold
                if dataset_size > gp_size_threshold:
                    logging.info(f"Skipping Gaussian Process for {model_type} due to large dataset size ({dataset_size}).")
                    models.append(None)
                    continue

                kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
                model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-10,
                    normalize_y=True,
                    random_state=42
                )
            model.fit(features_scaled, targets[:, i])
            models.append(model)
        except Exception as e:
            logging.warning(f"Error training {model_type} model for target {i}: {str(e)}")
            models.append(None)  # Append None for failed models
    return models, scaler

def evaluate_model(models, scaler, features, targets, model_type="xgboost"):
    """Evaluate the trained machine learning model with multiple metrics."""
    if not models or features.size == 0 or targets.size == 0:
        return {f'{model_type}_mae': float('inf'), f'{model_type}_rmse': float('inf'),
                f'{model_type}_r2': float('-inf')}
    predictions = []
    features_scaled = scaler.transform(features)
    # Standard models
    valid_predictions = False
    for i, model in enumerate(models):
        try:
            if model is None:
                # Skip None models (failed during training)
                logging.warning(f"Skipping evaluation for {model_type} model {i} (None)")
                # Use zeros as placeholder predictions
                pred = np.zeros(len(features))
            else:
                pred = model.predict(features_scaled)
                valid_predictions = True
            predictions.append(pred.flatten() if hasattr(pred, 'flatten') else pred)
        except Exception as e:
            logging.warning(f"Error during prediction with {model_type} model {i}: {str(e)}")
            # Use zeros as placeholder predictions
            predictions.append(np.zeros(len(features)))
    # If no valid predictions were made, return infinity metrics
    if not valid_predictions:
        return {f'{model_type}_mae': float('inf'), f'{model_type}_rmse': float('inf'),
                f'{model_type}_r2': float('-inf')}
    # Transpose predictions to match targets shape (num_samples, num_features)
    predictions = np.array(predictions).T
    try:
        # Calculate multiple metrics
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
    except Exception as e:
        logging.warning(f"Error calculating metrics for {model_type}: {str(e)}")
        mae = float('inf')
        rmse = float('inf')
        r2 = float('-inf')
    return {
        f'{model_type}_mae': mae,
        f'{model_type}_rmse': rmse,
        f'{model_type}_r2': r2
    }

def cross_validate_model(features, targets, model_type="xgboost", n_splits=5):
    """Perform time-series cross-validation for a given model type."""
    # Check if we have enough samples for any meaningful modeling
    if len(features) < 3:  # Need at least 3 samples for train/test split
        logging.warning(
            f"Insufficient samples ({len(features)}) for {model_type} model evaluation. Skipping cross-validation.")
        return {
            f'{model_type}_cv_mae': float('inf'),
            f'{model_type}_cv_rmse': float('inf'),
            f'{model_type}_cv_r2': float('-inf')
        }
    # Dynamically adjust n_splits based on data size
    actual_n_splits = min(n_splits, len(features) - 1)
    if actual_n_splits < 2:
        # Not enough data for cross-validation
        logging.warning(f"Not enough samples for cross-validation. Using simple train-test split instead.")
        # Use simple train-test split as fallback
        if len(features) <= 3:  # Very small dataset
            # Use all data for both training and testing (just to get a baseline)
            X_train, y_train = features, targets
            X_test, y_test = features, targets
            logging.warning(f"Dataset too small for proper evaluation. Using same data for training and testing.")
        else:
            # Regular train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
        try:
            models, scaler = train_traditional_model(X_train, y_train, model_type=model_type)
            if models and scaler:
                evaluation_results = evaluate_model(models, scaler, X_test, y_test, model_type=model_type)
                return {
                    f'{model_type}_cv_mae': evaluation_results[f'{model_type}_mae'],
                    f'{model_type}_cv_rmse': evaluation_results[f'{model_type}_rmse'],
                    f'{model_type}_cv_r2': evaluation_results[f'{model_type}_r2']
                }
        except Exception as e:
            logging.warning(f"Error during {model_type} model training/evaluation: {str(e)}")
        return {
            f'{model_type}_cv_mae': float('inf'),
            f'{model_type}_cv_rmse': float('inf'),
            f'{model_type}_cv_r2': float('-inf')
        }
    # Use TimeSeriesSplit for more realistic evaluation when enough data
    tscv = TimeSeriesSplit(n_splits=actual_n_splits)
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    for train_index, test_index in tscv.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        # Skip if we don't have enough samples in either split
        if len(X_train) < 2 or len(X_test) < 1:
            logging.warning(f"Skipping a fold with insufficient samples (train: {len(X_train)}, test: {len(X_test)})")
            continue
        # Train model based on type
        models, scaler = train_traditional_model(X_train, y_train, model_type=model_type)
        if models and scaler:
            evaluation_results = evaluate_model(models, scaler, X_test, y_test, model_type=model_type)
            mae_scores.append(evaluation_results[f'{model_type}_mae'])
            rmse_scores.append(evaluation_results[f'{model_type}_rmse'])
            r2_scores.append(evaluation_results[f'{model_type}_r2'])
    if mae_scores:
        return {
            f'{model_type}_cv_mae': np.mean(mae_scores),
            f'{model_type}_cv_rmse': np.mean(rmse_scores),
            f'{model_type}_cv_r2': np.mean(r2_scores)
        }
    else:
        return {
            f'{model_type}_cv_mae': float('inf'),
            f'{model_type}_cv_rmse': float('inf'),
            f'{model_type}_cv_r2': float('-inf')
        }

# --- Prediction Logic ---
def advanced_ensemble_predict(all_predictions, model_evaluations, num_range, num_to_predict):
    """Combine predictions from multiple models using weighted averaging based on model performance."""
    if not all_predictions:
        return sorted(random.sample(range(1, num_range + 1), num_to_predict))
    # Extract model types and their predictions
    model_types = list(all_predictions.keys())
    # Calculate weights based on inverse MAE (lower MAE = higher weight)
    weights = {}
    for model_type in model_types:
        mae_key = f"{model_type}_mae"
        if mae_key in model_evaluations and model_evaluations[mae_key] > 0:
            # Inverse MAE (better models get higher weights)
            weights[model_type] = 1.0 / (model_evaluations[mae_key] + 1e-8) # Add epsilon for numerical stability
        else:
            weights[model_type] = 1.0  # Default weight if no evaluation
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        for model_type in weights:
            weights[model_type] /= total_weight
    # Apply weighted average to predictions
    weighted_predictions = {}
    for model_type in model_types:
        prediction = all_predictions[model_type]
        if len(prediction) == num_to_predict:  # Ensure prediction has correct length
            for i, num in enumerate(prediction):
                if i not in weighted_predictions:
                    weighted_predictions[i] = 0
                weighted_predictions[i] += num * weights[model_type]
    # Round to nearest integer and ensure numbers are within range and unique
    final_prediction_set = set()
    for i in sorted(weighted_predictions.keys()):
        rounded_num = max(1, min(num_range, round(weighted_predictions[i])))
        final_prediction_set.add(rounded_num)
    # If not enough unique numbers, fill with random unique numbers
    while len(final_prediction_set) < num_to_predict:
        num = random.randint(1, num_range)
        final_prediction_set.add(num)
    return sorted(list(final_prediction_set)[:num_to_predict])

def create_prediction_features(number_sets, window_size, is_main=True):
    """Create feature vector for prediction with consistent dimensions."""
    global MAIN_FEATURE_DIM, BONUS_FEATURE_DIM
    if len(number_sets) < window_size:
        logging.error(f"Not enough historical data for prediction. Need at least {window_size} draws.")
        return None
    # Determine the number range based on whether we're processing main or bonus numbers
    num_range = get_config_value(config_parser, 'SETTINGS', 'main_num_range', int, 50) if is_main else get_config_value(config_parser, 'SETTINGS', 'bonus_num_range', int, 12)
    expected_dim = MAIN_FEATURE_DIM if is_main else BONUS_FEATURE_DIM
    recent_window = number_sets[-window_size:]
    # Create the same feature vector as in training
    feature_vector = []
    # 1. Basic features (flattened window)
    basic_features = [num for subset in recent_window for num in subset]
    feature_vector.extend(basic_features)
    # 2. Statistical features
    flat_window = np.array(recent_window).flatten()
    feature_vector.append(np.mean(flat_window))
    feature_vector.append(np.std(flat_window))
    feature_vector.append(np.median(flat_window))
    feature_vector.append(np.min(flat_window))
    feature_vector.append(np.max(flat_window))
    feature_vector.append(np.percentile(flat_window, 25))
    feature_vector.append(np.percentile(flat_window, 75))
    # 3. Frequency-based features - using fixed range
    counter = Counter(flat_window)
    frequencies = [counter.get(num, 0) for num in range(1, num_range + 1)]
    feature_vector.extend(frequencies)
    # 4. Pattern-based features
    # 4.1 Consecutive numbers
    consecutive_count = 0
    for j in range(len(flat_window) - 1):
        if flat_window[j] + 1 == flat_window[j + 1]:
            consecutive_count += 1
    feature_vector.append(consecutive_count)
    # 4.2 Even/Odd ratio
    even_count = sum(1 for num in flat_window if num % 2 == 0)
    feature_vector.append(even_count / len(flat_window) if len(flat_window) > 0 else 0)
    # 4.3 Low/High ratio
    mid_point = num_range // 2
    low_count = sum(1 for num in flat_window if num <= mid_point)
    feature_vector.append(low_count / len(flat_window) if len(flat_window) > 0 else 0)
    # 5. Lag features
    for j in range(window_size):
        feature_vector.extend(recent_window[j])
    # 6. Difference features
    for j in range(window_size - 1):
        diffs = [recent_window[j + 1][k] - recent_window[j][k] for k in range(len(recent_window[j]))]
        feature_vector.extend(diffs)
    # 7. Add time-based features placeholders
    feature_vector.extend([0, 0, 0, 0])  # Placeholder for weekday, day, month, year
    # Check if dimensions match expected and adjust if needed
    if expected_dim is not None:
        current_dim = len(feature_vector)
        if current_dim < expected_dim:
            # Pad with zeros if too short
            logging.warning(f"Feature vector too short ({current_dim} vs {expected_dim}). Padding with zeros.")
            feature_vector.extend([0] * (expected_dim - current_dim))
        elif current_dim > expected_dim:
            # Truncate if too long
            logging.warning(f"Feature vector too long ({current_dim} vs {expected_dim}). Truncating.")
            feature_vector = feature_vector[:expected_dim]
    return np.array([feature_vector])

def predict_with_traditional_model(number_sets, num_range, num_to_predict, models, scaler, window_size, is_main=True,
                                   model_type="xgboost"):
    """Predict the next numbers based on the trained traditional ML model."""
    predicted_set = set()  # Use a set to store unique predictions
    if models and scaler and len(number_sets) >= window_size:
        # Create feature vector with consistent dimensions
        feature_for_prediction = create_prediction_features(number_sets, window_size, is_main)
        if feature_for_prediction is None:
            logging.warning(f"Failed to create prediction features. Using random numbers.")
            return sorted(random.sample(range(1, num_range + 1), num_to_predict))
        try:
            feature_scaled = scaler.transform(feature_for_prediction)
            for model_idx, model in enumerate(models):
                if model is None:
                    continue
                prediction = round(model.predict(feature_scaled)[0])
                prediction = max(1, min(num_range, prediction))
                predicted_set.add(prediction)  # Add to set to ensure uniqueness
        except Exception as e:
            logging.warning(f"Prediction error: {str(e)}. Using random numbers.")
            return sorted(random.sample(range(1, num_range + 1), num_to_predict))
    # Convert set to list and sort
    ml_prediction = sorted(list(predicted_set))
    # If prediction didn't produce enough unique numbers, fill with random unique numbers
    while len(ml_prediction) < num_to_predict:
        num = random.randint(1, num_range)
        if num not in ml_prediction:
            ml_prediction.append(num)
    return sorted(ml_prediction[:num_to_predict])

# --- Visualization ---
def plot_frequency(number_sets, title, filename, num_range):
    """Plots the frequency of each number in the given sets with enhanced visualization."""
    all_numbers = [num for sublist in number_sets for num in sublist]
    counts = Counter(all_numbers)
    numbers = list(range(1, num_range + 1))  # Include all possible numbers
    frequencies = [counts.get(n, 0) for n in numbers]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(numbers, frequencies, color='skyblue')
    # Highlight most and least frequent numbers
    if frequencies:
        max_freq = max(frequencies)
        min_freq = min(frequencies)
        for i, freq in enumerate(frequencies):
            if freq == max_freq:
                bars[i].set_color('red')
            elif freq == min_freq:
                bars[i].set_color('green')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(numbers, rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add mean frequency line
    mean_freq = np.mean(frequencies)
    plt.axhline(y=mean_freq, color='orange', linestyle='--', label=f'Mean: {mean_freq:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved frequency plot to {filename}")

def plot_model_comparison(evaluations, metric='mae', title='Model Comparison', filename='model_comparison.png'):
    """Plot a comparison of model performance based on specified metric."""
    models = []
    values = []
    for key, value in evaluations.items():
        if key.endswith(f'_{metric}'):
            model_name = key.replace(f'_{metric}', '')
            models.append(model_name)
            values.append(value)
    # Sort by performance (lower is better for MAE and RMSE, higher is better for R2)
    if metric in ['mae', 'rmse']:
        sorted_indices = np.argsort(values)
    else:  # r2
        sorted_indices = np.argsort(values)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_models, sorted_values, color='skyblue')
    # Highlight best model
    if sorted_values:
        bars[0].set_color('green')
    plt.xlabel('Model')
    plt.ylabel(metric.upper())
    plt.title(f'{title} ({metric.upper()})')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved model comparison plot to {filename}")

# --- Main Execution ---
def main():
    try:
        file_path = get_data_file_path()
        main_number_sets, bonus_number_sets, dates = load_data(file_path)
        logging.info(
            f"Loaded {len(main_number_sets)} sets of main numbers and {len(bonus_number_sets)} sets of bonus numbers from {file_path}")

        # Traditional ML model types
        traditional_model_types = ["xgboost", "lightgbm", "random_forest", "gradient_boosting", "bayesian_ridge"]

        main_predictions = {}
        bonus_predictions = {}
        main_evaluations = {}
        bonus_evaluations = {}
        main_cv_evaluations = {}
        bonus_cv_evaluations = {}

        window_size = get_config_value(config_parser, 'SETTINGS', 'window_size', int, 5)
        main_num_range = get_config_value(config_parser, 'SETTINGS', 'main_num_range', int, 50)
        bonus_num_range = get_config_value(config_parser, 'SETTINGS', 'bonus_num_range', int, 12)
        num_main_to_predict = get_config_value(config_parser, 'SETTINGS', 'num_main_to_predict', int, 5)
        num_bonus_to_predict = get_config_value(config_parser, 'SETTINGS', 'num_bonus_to_predict', int, 2)
        test_size = get_config_value(config_parser, 'SETTINGS', 'test_size', float, 0.2)
        n_splits = get_config_value(config_parser, 'SETTINGS', 'n_splits', int, 3)

        # --- Main Numbers Prediction ---
        logging.info("Processing main numbers")
        main_features, main_targets = create_enhanced_features(main_number_sets, window_size=window_size, is_main=True,
                                                               dates=dates)
        if main_features.size > 0 and main_targets.size > 0:
            # Split data for evaluation
            X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(
                main_features, main_targets, test_size=test_size, random_state=42
            )
            # Train and evaluate traditional ML models
            for model_type in traditional_model_types:
                logging.info(f"Training {model_type} model for main numbers...")
                main_models, main_scaler = train_traditional_model(X_train_main, y_train_main, model_type=model_type)
                main_prediction = predict_with_traditional_model(
                    main_number_sets, main_num_range, num_main_to_predict,
                    main_models, main_scaler, window_size=window_size, is_main=True, model_type=model_type
                )
                main_predictions[model_type] = main_prediction
                # Evaluate the model
                evaluation_results = evaluate_model(main_models, main_scaler, X_test_main, y_test_main,
                                                    model_type=model_type)
                main_evaluations.update(evaluation_results)
                # Perform cross-validation
                cv_results = cross_validate_model(main_features, main_targets, model_type=model_type, n_splits=n_splits)
                main_cv_evaluations.update(cv_results)

            # Advanced ensemble prediction for main numbers (weighted by model performance)
            ensemble_main_prediction = advanced_ensemble_predict(
                main_predictions, main_evaluations, main_num_range, num_main_to_predict
            )
            main_predictions["ensemble"] = ensemble_main_prediction
        else:
            logging.warning("Not enough data to create features for main numbers model training.")

        # --- Bonus Numbers Prediction ---
        logging.info("Processing bonus numbers")
        bonus_features, bonus_targets = create_enhanced_features(bonus_number_sets, window_size=window_size, is_main=False,
                                                                 dates=dates)
        if bonus_features.size > 0 and bonus_targets.size > 0:
            # Split data for evaluation
            X_train_bonus, X_test_bonus, y_train_bonus, y_test_bonus = train_test_split(
                bonus_features, bonus_targets, test_size=test_size, random_state=42
            )
            # Train and evaluate traditional ML models
            for model_type in traditional_model_types:
                logging.info(f"Training {model_type} model for bonus numbers...")
                bonus_models, bonus_scaler = train_traditional_model(X_train_bonus, y_train_bonus, model_type=model_type)
                bonus_prediction = predict_with_traditional_model(
                    bonus_number_sets, bonus_num_range, num_bonus_to_predict,
                    bonus_models, bonus_scaler, window_size=window_size, is_main=False, model_type=model_type
                )
                bonus_predictions[model_type] = bonus_prediction
                # Evaluate the model
                evaluation_results = evaluate_model(bonus_models, bonus_scaler, X_test_bonus, y_test_bonus,
                                                    model_type=model_type)
                bonus_evaluations.update(evaluation_results)
                # Perform cross-validation
                cv_results = cross_validate_model(bonus_features, bonus_targets, model_type=model_type, n_splits=n_splits)
                bonus_cv_evaluations.update(cv_results)

            # Advanced ensemble prediction for bonus numbers (weighted by model performance)
            ensemble_bonus_prediction = advanced_ensemble_predict(
                bonus_predictions, bonus_evaluations, bonus_num_range, num_bonus_to_predict
            )
            bonus_predictions["ensemble"] = ensemble_bonus_prediction
        else:
            logging.warning("Not enough data to create features for bonus numbers model training.")

        # --- Data Visualization ---
        if main_number_sets:
            plot_frequency(main_number_sets, 'Main Numbers Frequency', 'main_numbers_frequency.png', main_num_range)
        if bonus_number_sets:
            plot_frequency(bonus_number_sets, 'Bonus Numbers Frequency', 'bonus_numbers_frequency.png', bonus_num_range)

        # Model comparison plots
        if main_evaluations:
            plot_model_comparison(main_evaluations, metric='mae', title='Main Numbers Model Comparison (MAE)',
                                  filename='main_numbers_model_comparison_mae.png')
            plot_model_comparison(main_evaluations, metric='rmse', title='Main Numbers Model Comparison (RMSE)',
                                  filename='main_numbers_model_comparison_rmse.png')
        if bonus_evaluations:
            plot_model_comparison(bonus_evaluations, metric='mae', title='Bonus Numbers Model Comparison (MAE)',
                                  filename='bonus_numbers_model_comparison_mae.png')
            plot_model_comparison(bonus_evaluations, metric='rmse', title='Bonus Numbers Model Comparison (RMSE)',
                                  filename='bonus_numbers_model_comparison_rmse.png')

        # --- Print Predictions ---
        print("\n--- Predictions ---")
        print("\nMain Numbers:")
        for model_type, prediction in main_predictions.items():
            print(f'{model_type.capitalize()}: {" ".join(map(str, prediction))}')
        print("\nBonus Numbers:")
        for model_type, prediction in bonus_predictions.items():
            print(f'{model_type.capitalize()}: {" ".join(map(str, prediction))}')

        # --- Print Evaluations ---
        print("\n--- Model Evaluations (Mean Absolute Error) ---")
        print("\nMain Numbers Models (Test Set):")
        for model_type in traditional_model_types + ["ensemble"]:
            mae_key = f'{model_type}_mae'
            if mae_key in main_evaluations:
                print(f'{model_type.capitalize()}: {main_evaluations[mae_key]:.2f}')
        print("\nMain Numbers Models (Cross-Validation):")
        for model_type in traditional_model_types:
            cv_mae_key = f'{model_type}_cv_mae'
            if cv_mae_key in main_cv_evaluations:
                print(f'{model_type.capitalize()}: {main_cv_evaluations[cv_mae_key]:.2f}')
        print("\nBonus Numbers Models (Test Set):")
        for model_type in traditional_model_types + ["ensemble"]:
            mae_key = f'{model_type}_mae'
            if mae_key in bonus_evaluations:
                print(f'{model_type.capitalize()}: {bonus_evaluations[mae_key]:.2f}')
        print("\nBonus Numbers Models (Cross-Validation):")
        for model_type in traditional_model_types:
            cv_mae_key = f'{model_type}_cv_mae'
            if cv_mae_key in bonus_cv_evaluations:
                print(f'{model_type.capitalize()}: {bonus_cv_evaluations[cv_mae_key]:.2f}')

    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()