#!/usr/bin/env python3
"""
Enhanced Lottery Number Prediction Script (Fixed Version)

This script analyzes historical lottery data to predict the next set of numbers
using advanced feature engineering, robust ensemble methods, and optimized machine learning models.
It predicts 5 unique main numbers (1-50) and 2 unique bonus numbers (1-12) based on
sophisticated pattern analysis and machine learning techniques.

Fixes:
- Replaced np.skew with scipy.stats.skew to correctly calculate skewness.
- Added scipy import to the script.

Updated 28-Jul-25

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
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import xgboost as xgb
import lightgbm as lgb
import configparser
from itertools import combinations
from scipy.stats import skew  # Added for skewness calculation

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("lottery_prediction.log")]
)

# Default configuration
DEFAULT_CONFIG = {
    'SETTINGS': {
        'window_size': '5',
        'main_num_range': '50',
        'bonus_num_range': '12',
        'num_main_to_predict': '5',
        'num_bonus_to_predict': '2',
        'test_size': '0.2',
        'n_splits': '5',
        'random_seed': '42'
    }
}

# Create default config file if it doesn't exist
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
config = create_default_config()
config.read("config.ini")

# Global variables to store feature dimensions for consistency
MAIN_FEATURE_DIM = None
BONUS_FEATURE_DIM = None

def get_data_file_path():
    """Get the path to the data file. Assumes data.txt is in the script's directory if no arg given."""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not os.path.isfile(file_path):
            logging.error(f"Data file not found at {file_path}")
            sys.exit(1)
        return file_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "data.txt")
        if not os.path.isfile(file_path):
            logging.error(f"data.txt not found in script directory: {script_dir}")
            sys.exit(1)
        return file_path

def load_data(file_path):
    """Load and parse the data file with enhanced validation and date parsing."""
    main_number_sets = []
    bonus_number_sets = []
    processed_lines = set()  # For deduplication
    dates = []  # Store dates for temporal features

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        sys.exit(1)

    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()
        if not stripped_line:
            logging.debug(f"Skipping empty line {line_num}")
            continue

        if stripped_line in processed_lines:
            logging.warning(f"Skipping duplicate line {line_num}: {stripped_line}")
            continue
        processed_lines.add(stripped_line)

        try:
            # Assuming format: [date (YYYY-MM-DD, optional)] num1 num2 num3 num4 num5 bonus1 bonus2
            parts = stripped_line.split()
            date = None
            if len(parts) > 7 and '-' in parts[0]:
                try:
                    date = datetime.strptime(parts[0], "%Y-%m-%d")
                    parts = parts[1:]
                except ValueError:
                    logging.warning(f"Invalid date format in line {line_num}: {parts[0]}")
                    date = None

            if len(parts) != 7:
                logging.warning(f"Skipping line {line_num} due to incorrect number count (expected 7): {stripped_line}")
                continue

            numbers = [int(num) for num in parts]
            main_nums = numbers[:5]
            bonus_nums = numbers[5:]

            # Validate ranges and uniqueness
            main_range = config['SETTINGS'].getint('main_num_range')
            bonus_range = config['SETTINGS'].getint('bonus_num_range')
            if not all(1 <= num <= main_range for num in main_nums):
                logging.warning(f"Main numbers out of range (1-{main_range}) in line {line_num}: {main_nums}")
                continue
            if not all(1 <= num <= bonus_range for num in bonus_nums):
                logging.warning(f"Bonus numbers out of range (1-{bonus_range}) in line {line_num}: {bonus_nums}")
                continue
            if len(set(main_nums)) != 5:
                logging.warning(f"Duplicate main numbers in line {line_num}: {main_nums}")
                continue
            if len(set(bonus_nums)) != 2:
                logging.warning(f"Duplicate bonus numbers in line {line_num}: {bonus_nums}")
                continue

            main_number_sets.append(main_nums)
            bonus_number_sets.append(bonus_nums)
            dates.append(date)
        except ValueError:
            logging.error(f"Invalid number format in line {line_num}: {stripped_line}")
            continue

    if not main_number_sets:
        logging.error("No valid data loaded from file")
        sys.exit(1)

    logging.info(f"Loaded {len(main_number_sets)} valid draws")
    return main_number_sets, bonus_number_sets, dates

def create_enhanced_features(number_sets, window_size, is_main=True, dates=None):
    """Create advanced features for machine learning prediction with additional feature engineering."""
    global MAIN_FEATURE_DIM, BONUS_FEATURE_DIM

    if len(number_sets) <= window_size:
        logging.warning(f"Insufficient draws ({len(number_sets)}) for feature creation. Need > {window_size}")
        return np.array([]), np.array([])

    features = []
    targets = []
    num_range = config['SETTINGS'].getint('main_num_range') if is_main else config['SETTINGS'].getint('bonus_num_range')

    for i in range(len(number_sets) - window_size):
        window = number_sets[i:i + window_size]
        feature_vector = []

        # 1. Basic features (flattened window)
        basic_features = [num for subset in window for num in subset]
        feature_vector.extend(basic_features)

        # 2. Statistical features
        flat_window = np.array(basic_features)
        feature_vector.extend([
            np.mean(flat_window),
            np.std(flat_window),
            np.median(flat_window),
            np.min(flat_window),
            np.max(flat_window),
            np.percentile(flat_window, 25),
            np.percentile(flat_window, 75),
            np.ptp(flat_window),  # Range (max - min)
            np.var(flat_window),  # Variance
            skew(flat_window) if np.std(flat_window) != 0 else 0,  # Skewness
        ])

        # 3. Frequency-based features
        counter = Counter(flat_window)
        frequencies = [counter.get(num, 0) / len(flat_window) for num in range(1, num_range + 1)]
        feature_vector.extend(frequencies)

        # 4. Pattern-based features
        # 4.1 Consecutive numbers
        sorted_window = sorted(flat_window)
        consecutive_count = sum(1 for j in range(len(sorted_window) - 1) if sorted_window[j] + 1 == sorted_window[j + 1])
        feature_vector.append(consecutive_count)

        # 4.2 Even/Odd ratio
        even_count = sum(1 for num in flat_window if num % 2 == 0)
        feature_vector.append(even_count / len(flat_window) if len(flat_window) > 0 else 0)

        # 4.3 Low/High ratio
        mid_point = num_range // 2
        low_count = sum(1 for num in flat_window if num <= mid_point)
        feature_vector.append(low_count / len(flat_window) if len(flat_window) > 0 else 0)

        # 4.4 Sum and product features
        feature_vector.append(sum(flat_window))
        feature_vector.append(np.prod(flat_window))

        # 4.5 Pairwise differences within each draw
        for draw in window:
            pairs = list(combinations(draw, 2))
            diffs = [abs(p[0] - p[1]) for p in pairs]
            feature_vector.extend([np.mean(diffs), np.std(diffs) if len(diffs) > 1 else 0])

        # 5. Lag features
        for j in range(window_size):
            feature_vector.extend(window[j])

        # 6. Difference features between draws
        for j in range(window_size - 1):
            diffs = [window[j + 1][k] - window[j][k] for k in range(len(window[j]))]
            feature_vector.extend([np.mean(diffs), np.std(diffs) if len(diffs) > 1 else 0])

        # 7. Time-based features
        if dates and dates[i] is not None:
            try:
                date = dates[i]
                feature_vector.extend([
                    date.weekday(),
                    date.day,
                    date.month,
                    date.year,
                    (date - datetime(1970, 1, 1)).days  # Days since epoch
                ])
            except:
                feature_vector.extend([0] * 5)
        else:
            feature_vector.extend([0] * 5)

        # 8. Polynomial features for key statistics
        key_stats = [np.mean(flat_window), np.std(flat_window), sum(flat_window)]
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform([key_stats])[0]
        feature_vector.extend(poly_features)

        targets.append(number_sets[i + window_size])
        features.append(feature_vector)

    features_array = np.array(features)
    targets_array = np.array(targets)

    # Apply PCA to reduce dimensionality if needed
    if features_array.shape[1] > 100:
        pca = PCA(n_components=0.95)  # Retain 95% of variance
        features_array = pca.fit_transform(features_array)
        logging.info(f"Applied PCA, reduced features from {features_array.shape[1]} to {pca.n_components_}")

    if is_main:
        MAIN_FEATURE_DIM = features_array.shape[1]
    else:
        BONUS_FEATURE_DIM = features_array.shape[1]

    return features_array, targets_array

def train_traditional_model(features, targets, model_type="xgboost"):
    """Train a traditional ML model with hyperparameter tuning."""
    if features.size == 0 or targets.size == 0 or len(features) < 2:
        logging.warning(f"Insufficient samples ({len(features)}) for {model_type} model training")
        return [], None

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    models = []
    for i in range(targets.shape[1]):
        try:
            if model_type == "xgboost":
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.7, 0.8],
                    'colsample_bytree': [0.7, 0.8]
                }
                base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=config['SETTINGS'].getint('random_seed'))
            elif model_type == "lightgbm":
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'num_leaves': [15, 31],
                    'max_depth': [3, 5],
                    'subsample': [0.7, 0.8]
                }
                base_model = lgb.LGBMRegressor(random_state=config['SETTINGS'].getint('random_seed'))
            elif model_type == "random_forest":
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                base_model = RandomForestRegressor(random_state=config['SETTINGS'].getint('random_seed'))
            elif model_type == "gradient_boosting":
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.7, 0.8]
                }
                base_model = GradientBoostingRegressor(random_state=config['SETTINGS'].getint('random_seed'))
            elif model_type == "bayesian_ridge":
                param_grid = {
                    'alpha_1': [1e-6, 1e-5],
                    'alpha_2': [1e-6, 1e-5],
                    'lambda_1': [1e-6, 1e-5],
                    'lambda_2': [1e-6, 1e-5]
                }
                base_model = BayesianRidge()
            else:
                logging.warning(f"Unsupported model type: {model_type}")
                models.append(None)
                continue

            grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid_search.fit(features_scaled, targets[:, i])
            models.append(grid_search.best_estimator_)
            logging.info(f"Best params for {model_type} (target {i+1}): {grid_search.best_params_}")
        except Exception as e:
            logging.warning(f"Error training {model_type} for target {i+1}: {str(e)}")
            models.append(None)
    return models, scaler

def train_stacking_ensemble(features, targets):
    """Train a stacking ensemble model combining multiple base models."""
    if features.size == 0 or targets.size == 0 or len(features) < 2:
        logging.warning(f"Insufficient samples ({len(features)}) for stacking ensemble")
        return [], None

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    models = []
    for i in range(targets.shape[1]):
        try:
            estimators = [
                ('xgb', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, random_state=config['SETTINGS'].getint('random_seed'))),
                ('lgb', lgb.LGBMRegressor(n_estimators=100, max_depth=3, random_state=config['SETTINGS'].getint('random_seed'))),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=3, random_state=config['SETTINGS'].getint('random_seed')))
            ]
            stacking_model = StackingRegressor(
                estimators=estimators,
                final_estimator=BayesianRidge(),
                cv=3
            )
            stacking_model.fit(features_scaled, targets[:, i])
            models.append(stacking_model)
        except Exception as e:
            logging.warning(f"Error training stacking ensemble for target {i+1}: {str(e)}")
            models.append(None)
    return models, scaler

def evaluate_model(models, scaler, features, targets, model_type="xgboost"):
    """Evaluate the trained model with multiple metrics."""
    if not models or features.size == 0 or targets.size == 0:
        return {f'{model_type}_mae': float('inf'), f'{model_type}_rmse': float('inf'), f'{model_type}_r2': float('-inf')}

    predictions = []
    features_scaled = scaler.transform(features)

    valid_predictions = False
    for i, model in enumerate(models):
        if model is None:
            logging.warning(f"Skipping evaluation for {model_type} model {i+1} (None)")
            pred = np.zeros(len(features))
        else:
            try:
                pred = model.predict(features_scaled)
                valid_predictions = True
            except Exception as e:
                logging.warning(f"Error predicting with {model_type} model {i+1}: {str(e)}")
                pred = np.zeros(len(features))
        predictions.append(pred)

    if not valid_predictions:
        return {f'{model_type}_mae': float('inf'), f'{model_type}_rmse': float('inf'), f'{model_type}_r2': float('-inf')}

    predictions = np.array(predictions).T
    try:
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
    except Exception as e:
        logging.warning(f"Error calculating metrics for {model_type}: {str(e)}")
        mae = rmse = float('inf')
        r2 = float('-inf')

    return {f'{model_type}_mae': mae, f'{model_type}_rmse': rmse, f'{model_type}_r2': r2}

def cross_validate_model(features, targets, model_type="xgboost", n_splits=5):
    """Perform time-series cross-validation with dynamic split adjustment."""
    if len(features) < 3:
        logging.warning(f"Insufficient samples ({len(features)}) for {model_type} cross-validation")
        return {f'{model_type}_cv_mae': float('inf'), f'{model_type}_cv_rmse': float('inf'), f'{model_type}_cv_r2': float('-inf')}

    actual_n_splits = min(n_splits, len(features) - 1)
    if actual_n_splits < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=config['SETTINGS'].getint('random_seed')
        )
        models, scaler = train_traditional_model(X_train, y_train, model_type=model_type)
        if models and scaler:
            evaluation_results = evaluate_model(models, scaler, X_test, y_test, model_type=model_type)
            return {
                f'{model_type}_cv_mae': evaluation_results[f'{model_type}_mae'],
                f'{model_type}_cv_rmse': evaluation_results[f'{model_type}_rmse'],
                f'{model_type}_cv_r2': evaluation_results[f'{model_type}_r2']
            }
        return {f'{model_type}_cv_mae': float('inf'), f'{model_type}_cv_rmse': float('inf'), f'{model_type}_cv_r2': float('-inf')}

    tscv = TimeSeriesSplit(n_splits=actual_n_splits)
    mae_scores, rmse_scores, r2_scores = [], [], []

    for train_index, test_index in tscv.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        if len(X_train) < 2 or len(X_test) < 1:
            logging.warning(f"Skipping fold with insufficient samples (train: {len(X_train)}, test: {len(X_test)})")
            continue

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
    return {f'{model_type}_cv_mae': float('inf'), f'{model_type}_cv_rmse': float('inf'), f'{model_type}_cv_r2': float('-inf')}

def advanced_ensemble_predict(all_predictions, model_evaluations, num_range, num_to_predict):
    """Combine predictions using weighted averaging and ensure valid, unique numbers."""
    if not all_predictions:
        logging.warning("No predictions available for ensemble. Using random numbers")
        return sorted(random.sample(range(1, num_range + 1), num_to_predict))

    model_types = list(all_predictions.keys())
    weights = {}
    for model_type in model_types:
        mae_key = f"{model_type}_mae"
        weights[model_type] = 1.0 / model_evaluations.get(mae_key, float('inf')) if model_evaluations.get(mae_key, float('inf')) > 0 else 1.0

    total_weight = sum(weights.values())
    if total_weight > 0:
        for model_type in weights:
            weights[model_type] /= total_weight

    weighted_predictions = {}
    for model_type in model_types:
        prediction = all_predictions[model_type]
        if len(prediction) == num_to_predict:
            for i, num in enumerate(prediction):
                if i not in weighted_predictions:
                    weighted_predictions[i] = 0
                weighted_predictions[i] += num * weights[model_type]

    predicted_set = set()
    for i in sorted(weighted_predictions.keys()):
        rounded_num = max(1, min(num_range, round(weighted_predictions[i])))
        predicted_set.add(rounded_num)

    # Ensure enough unique numbers
    while len(predicted_set) < num_to_predict:
        num = random.randint(1, num_range)
        if num not in predicted_set:
            predicted_set.add(num)

    return sorted(list(predicted_set)[:num_to_predict])

def create_prediction_features(number_sets, window_size, is_main=True):
    """Create feature vector for prediction with consistent dimensions."""
    global MAIN_FEATURE_DIM, BONUS_FEATURE_DIM
    num_range = config['SETTINGS'].getint('main_num_range') if is_main else config['SETTINGS'].getint('bonus_num_range')
    expected_dim = MAIN_FEATURE_DIM if is_main else BONUS_FEATURE_DIM

    if len(number_sets) < window_size:
        logging.error(f"Not enough historical data. Need at least {window_size} draws")
        return None

    recent_window = number_sets[-window_size:]
    feature_vector = []

    # 1. Basic features
    basic_features = [num for subset in recent_window for num in subset]
    feature_vector.extend(basic_features)

    # 2. Statistical features
    flat_window = np.array(basic_features)
    feature_vector.extend([
        np.mean(flat_window),
        np.std(flat_window),
        np.median(flat_window),
        np.min(flat_window),
        np.max(flat_window),
        np.percentile(flat_window, 25),
        np.percentile(flat_window, 75),
        np.ptp(flat_window),
        np.var(flat_window),
        skew(flat_window) if np.std(flat_window) != 0 else 0,
    ])

    # 3. Frequency-based features
    counter = Counter(flat_window)
    frequencies = [counter.get(num, 0) / len(flat_window) for num in range(1, num_range + 1)]
    feature_vector.extend(frequencies)

    # 4. Pattern-based features
    sorted_window = sorted(flat_window)
    consecutive_count = sum(1 for j in range(len(sorted_window) - 1) if sorted_window[j] + 1 == sorted_window[j + 1])
    feature_vector.append(consecutive_count)
    even_count = sum(1 for num in flat_window if num % 2 == 0)
    feature_vector.append(even_count / len(flat_window) if len(flat_window) > 0 else 0)
    mid_point = num_range // 2
    low_count = sum(1 for num in flat_window if num <= mid_point)
    feature_vector.append(low_count / len(flat_window) if len(flat_window) > 0 else 0)
    feature_vector.append(sum(flat_window))
    feature_vector.append(np.prod(flat_window))

    # 4.5 Pairwise differences
    for draw in recent_window:
        pairs = list(combinations(draw, 2))
        diffs = [abs(p[0] - p[1]) for p in pairs]
        feature_vector.extend([np.mean(diffs), np.std(diffs) if len(diffs) > 1 else 0])

    # 5. Lag features
    for j in range(window_size):
        feature_vector.extend(recent_window[j])

    # 6. Difference features
    for j in range(window_size - 1):
        diffs = [recent_window[j + 1][k] - recent_window[j][k] for k in range(len(recent_window[j]))]
        feature_vector.extend([np.mean(diffs), np.std(diffs) if len(diffs) > 1 else 0])

    # 7. Time-based features
    feature_vector.extend([0] * 5)

    # 8. Polynomial features
    key_stats = [np.mean(flat_window), np.std(flat_window), sum(flat_window)]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform([key_stats])[0]
    feature_vector.extend(poly_features)

    # Adjust dimensions
    if expected_dim is not None:
        current_dim = len(feature_vector)
        if current_dim < expected_dim:
            logging.warning(f"Feature vector too short ({current_dim} vs {expected_dim}). Padding with zeros")
            feature_vector.extend([0] * (expected_dim - current_dim))
        elif current_dim > expected_dim:
            logging.warning(f"Feature vector too long ({current_dim} vs {expected_dim}). Truncating")
            feature_vector = feature_vector[:expected_dim]

    # Apply PCA if used during training
    if expected_dim < len(feature_vector):
        pca = PCA(n_components=expected_dim)
        feature_vector = pca.fit_transform([feature_vector])[0]

    return np.array([feature_vector])

def predict_with_traditional_model(number_sets, num_range, num_to_predict, models, scaler, window_size, is_main=True, model_type="xgboost"):
    """Predict the next numbers with post-processing for uniqueness."""
    predicted_set = set()
    if models and scaler and len(number_sets) >= window_size:
        feature_for_prediction = create_prediction_features(number_sets, window_size, is_main)
        if feature_for_prediction is None:
            logging.warning(f"Failed to create prediction features for {model_type}")
            return sorted(random.sample(range(1, num_range + 1), num_to_predict))

        try:
            feature_scaled = scaler.transform(feature_for_prediction)
            for model in models:
                if model is None:
                    continue
                prediction = model.predict(feature_scaled)[0]
                rounded = max(1, min(num_range, round(prediction)))
                predicted_set.add(rounded)
        except Exception as e:
            logging.warning(f"Prediction error with {model_type}: {str(e)}")

    while len(predicted_set) < num_to_predict:
        num = random.randint(1, num_range)
        predicted_set.add(num)

    return sorted(list(predicted_set)[:num_to_predict])

def plot_frequency(number_sets, title, filename, num_range):
    """Plot number frequency with enhanced visualization."""
    all_numbers = [num for sublist in number_sets for num in sublist]
    counts = Counter(all_numbers)
    numbers = list(range(1, num_range + 1))
    frequencies = [counts.get(n, 0) for n in numbers]

    plt.figure(figsize=(14, 7))
    bars = plt.bar(numbers, frequencies, color='skyblue', edgecolor='black')
    if frequencies:
        max_freq = max(frequencies)
        min_freq = min(frequencies)
        for i, freq in enumerate(frequencies):
            if freq == max_freq:
                bars[i].set_color('red')
            elif freq == min_freq:
                bars[i].set_color('green')
    plt.axhline(y=np.mean(frequencies), color='orange', linestyle='--', label=f'Mean: {np.mean(frequencies):.2f}')
    plt.xlabel('Number', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(numbers, rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved frequency plot to {filename}")

def plot_model_comparison(evaluations, metric='mae', title='Model Comparison', filename='model_comparison.png'):
    """Plot model performance comparison."""
    models, values = [], []
    for key, value in evaluations.items():
        if key.endswith(f'_{metric}'):
            model_name = key.replace(f'_{metric}', '')
            models.append(model_name)
            values.append(value)

    if not values:
        logging.warning(f"No {metric} data available for plotting")
        return

    if metric in ['mae', 'rmse']:
        sorted_indices = np.argsort(values)
    else:
        sorted_indices = np.argsort(values)[::-1]

    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_models, sorted_values, color='skyblue', edgecolor='black')
    if sorted_values:
        bars[0].set_color('green')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.title(f'{title} ({metric.upper()})', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved model comparison plot to {filename}")

def main():
    """Main function to execute the lottery prediction pipeline."""
    logging.info("Starting lottery number prediction")
    file_path = get_data_file_path()
    main_number_sets, bonus_number_sets, dates = load_data(file_path)

    window_size = config['SETTINGS'].getint('window_size')
    main_num_range = config['SETTINGS'].getint('main_num_range')
    bonus_num_range = config['SETTINGS'].getint('bonus_num_range')
    num_main_to_predict = config['SETTINGS'].getint('num_main_to_predict')
    num_bonus_to_predict = config['SETTINGS'].getint('num_bonus_to_predict')
    test_size = config['SETTINGS'].getfloat('test_size')
    n_splits = config['SETTINGS'].getint('n_splits')

    model_types = ["xgboost", "lightgbm", "random_forest", "gradient_boosting", "bayesian_ridge", "stacking"]
    main_predictions, bonus_predictions = {}, {}
    main_evaluations, bonus_evaluations = {}, {}
    main_cv_evaluations, bonus_cv_evaluations = {}, {}

    # Main Numbers Prediction
    logging.info("Processing main numbers")
    main_features, main_targets = create_enhanced_features(main_number_sets, window_size, is_main=True, dates=dates)
    if main_features.size > 0 and main_targets.size > 0:
        X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(
            main_features, main_targets, test_size=test_size, random_state=config['SETTINGS'].getint('random_seed')
        )

        for model_type in model_types:
            logging.info(f"Training {model_type} model for main numbers")
            if model_type == "stacking":
                models, scaler = train_stacking_ensemble(X_train_main, y_train_main)
            else:
                models, scaler = train_traditional_model(X_train_main, y_train_main, model_type=model_type)
            prediction = predict_with_traditional_model(
                main_number_sets, main_num_range, num_main_to_predict, models, scaler, window_size, is_main=True, model_type=model_type
            )
            main_predictions[model_type] = prediction
            evaluation_results = evaluate_model(models, scaler, X_test_main, y_test_main, model_type=model_type)
            main_evaluations.update(evaluation_results)
            cv_results = cross_validate_model(main_features, main_targets, model_type=model_type, n_splits=n_splits)
            main_cv_evaluations.update(cv_results)

        main_predictions["ensemble"] = advanced_ensemble_predict(main_predictions, main_evaluations, main_num_range, num_main_to_predict)
    else:
        logging.warning("Insufficient data for main numbers model training")
        main_predictions["ensemble"] = sorted(random.sample(range(1, main_num_range + 1), num_main_to_predict))

    # Bonus Numbers Prediction
    logging.info("Processing bonus numbers")
    bonus_features, bonus_targets = create_enhanced_features(bonus_number_sets, window_size, is_main=False, dates=dates)
    if bonus_features.size > 0 and bonus_targets.size > 0:
        X_train_bonus, X_test_bonus, y_train_bonus, y_test_bonus = train_test_split(
            bonus_features, bonus_targets, test_size=test_size, random_state=config['SETTINGS'].getint('random_seed')
        )

        for model_type in model_types:
            logging.info(f"Training {model_type} model for bonus numbers")
            if model_type == "stacking":
                models, scaler = train_stacking_ensemble(X_train_bonus, y_train_bonus)
            else:
                models, scaler = train_traditional_model(X_train_bonus, y_train_bonus, model_type=model_type)
            prediction = predict_with_traditional_model(
                bonus_number_sets, bonus_num_range, num_bonus_to_predict, models, scaler, window_size, is_main=False, model_type=model_type
            )
            bonus_predictions[model_type] = prediction
            evaluation_results = evaluate_model(models, scaler, X_test_bonus, y_test_bonus, model_type=model_type)
            bonus_evaluations.update(evaluation_results)
            cv_results = cross_validate_model(bonus_features, bonus_targets, model_type=model_type, n_splits=n_splits)
            bonus_cv_evaluations.update(cv_results)

        bonus_predictions["ensemble"] = advanced_ensemble_predict(bonus_predictions, bonus_evaluations, bonus_num_range, num_bonus_to_predict)
    else:
        logging.warning("Insufficient data for bonus numbers model training")
        bonus_predictions["ensemble"] = sorted(random.sample(range(1, bonus_num_range + 1), num_bonus_to_predict))

    # Visualizations
    if main_number_sets:
        plot_frequency(main_number_sets, 'Main Numbers Frequency', 'main_numbers_frequency.png', main_num_range)
    if bonus_number_sets:
        plot_frequency(bonus_number_sets, 'Bonus Numbers Frequency', 'bonus_numbers_frequency.png', bonus_num_range)

    for metric in ['mae', 'rmse']:
        plot_model_comparison(main_evaluations, metric=metric, title=f'Main Numbers Model Comparison ({metric.upper()})',
                              filename=f'main_numbers_model_comparison_{metric}.png')
        plot_model_comparison(bonus_evaluations, metric=metric, title=f'Bonus Numbers Model Comparison ({metric.upper()})',
                              filename=f'bonus_numbers_model_comparison_{metric}.png')

    # Print Results
    print("\n--- Predictions ---")
    print("\nMain Numbers:")
    for model_type, prediction in main_predictions.items():
        print(f'{model_type.capitalize()}: {" ".join(map(str, prediction))}')

    print("\nBonus Numbers:")
    for model_type, prediction in bonus_predictions.items():
        print(f'{model_type.capitalize()}: {" ".join(map(str, prediction))}')

    print("\n--- Model Evaluations (Mean Absolute Error) ---")
    print("\nMain Numbers Models (Test Set):")
    for model_type in model_types + ["ensemble"]:
        mae_key = f'{model_type}_mae'
        if mae_key in main_evaluations:
            print(f'{model_type.capitalize()}: {main_evaluations[mae_key]:.2f}')

    print("\nMain Numbers Models (Cross-Validation):")
    for model_type in model_types:
        cv_mae_key = f'{model_type}_cv_mae'
        if cv_mae_key in main_cv_evaluations:
            print(f'{model_type.capitalize()}: {main_cv_evaluations[cv_mae_key]:.2f}')

    print("\nBonus Numbers Models (Test Set):")
    for model_type in model_types + ["ensemble"]:
        mae_key = f'{model_type}_mae'
        if mae_key in bonus_evaluations:
            print(f'{model_type.capitalize()}: {bonus_evaluations[mae_key]:.2f}')

    print("\nBonus Numbers Models (Cross-Validation):")
    for model_type in model_types:
        cv_mae_key = f'{model_type}_cv_mae'
        if cv_mae_key in bonus_cv_evaluations:
            print(f'{model_type.capitalize()}: {bonus_cv_evaluations[cv_mae_key]:.2f}')

    logging.info("Prediction pipeline completed")

if __name__ == "__main__":
    main()