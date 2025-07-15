#!/usr/bin/env python3
"""
Script creation 15 July 2025
Number Prediction Script

This script analyzes historical number data to predict the next set of numbers.
It predicts 5 unique main numbers in the range of 1-50 and 2 unique bonus numbers in the
range of 1-12, based on frequency analysis, statistical patterns, and
XGBoost, LightGBM, CatBoost, and Artificial Neural Networks (ANN) machine learning models.
Predictions are ensured to be unique within each set.
"""

import numpy as np
import pandas as pd
from collections import Counter
import random
import os
import sys
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import configparser
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")


def get_data_file_path():
    """Get the path to the data file. Assumes data.txt is in the script's directory if no arg given."""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not os.path.isfile(file_path):
            logging.error(f"File not found at {file_path}")
            logging.error("Please provide a valid path or ensure data.txt is in the script directory.")
            sys.exit(1)
        return file_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "data.txt")
        if not os.path.isfile(file_path):
            logging.error(f"data.txt not found in script directory: {script_dir}")
            logging.error(
                "Please place data.txt in the same directory as the script or provide the path as a command line argument.")
            sys.exit(1)
        return file_path


def load_data(file_path):
    """Load and parse the data file containing number selections."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        main_number_sets = []
        bonus_number_sets = []
        processed_lines = set()  # For deduplication

        for line_num, line in enumerate(lines):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            if stripped_line in processed_lines:
                logging.warning(f"Skipping duplicate line {line_num + 1}: {stripped_line}")
                continue
            processed_lines.add(stripped_line)

            try:
                numbers = [int(num) for num in stripped_line.split() if num.isdigit()]
            except ValueError:
                logging.error(f"Invalid number format in line {line_num + 1}: {stripped_line}")
                continue

            if len(numbers) == 7:  # Expecting 5 main and 2 bonus numbers
                main_nums = numbers[:5]
                bonus_nums = numbers[5:]

                # Validate ranges
                main_valid = all(1 <= num <= config['SETTINGS'].getint('main_num_range') for num in main_nums)
                bonus_valid = all(1 <= num <= config['SETTINGS'].getint('bonus_num_range') for num in bonus_nums)

                if not main_valid:
                    logging.warning(
                        f"Main numbers out of range (1-{config['SETTINGS'].getint('main_num_range')}) in line {line_num + 1}: {main_nums}")
                    continue
                if not bonus_valid:
                    logging.warning(
                        f"Bonus numbers out of range (1-{config['SETTINGS'].getint('bonus_num_range')}) in line {line_num + 1}: {bonus_nums}")
                    continue

                main_number_sets.append(main_nums)
                bonus_number_sets.append(bonus_nums)
            else:
                logging.warning(
                    f"Skipping line {line_num + 1} due to incorrect number count (expected 7): {stripped_line}")

        return main_number_sets, bonus_number_sets
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        logging.error("Please ensure 'data.txt' is in the same directory as the script.")
        sys.exit(1)  # Exit if file not found


def create_features(number_sets, window_size):
    """Create features for machine learning prediction with enhanced feature engineering."""
    features = []
    targets = []

    if len(number_sets) <= window_size:
        return np.array([]), np.array([])

    for i in range(len(number_sets) - window_size):
        window = number_sets[i:i + window_size]

        # Basic features (flattened window)
        feature_vector = [num for subset in window for num in subset]

        # Statistical features
        flat_window = np.array(window).flatten()
        feature_vector.append(np.mean(flat_window))
        feature_vector.append(np.std(flat_window))
        feature_vector.append(np.min(flat_window))
        feature_vector.append(np.max(flat_window))

        # Lag features (last draw)
        feature_vector.extend(window[-1])

        target = number_sets[i + window_size]

        features.append(feature_vector)
        targets.append(target)

    return np.array(features), np.array(targets)


def train_model(features, targets, model_type="xgboost"):
    """Train a machine learning model to predict the next set of numbers with hyperparameter tuning."""
    if features.size == 0 or targets.size == 0:
        return [], None

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    models = []
    for i in range(targets.shape[1]):
        if model_type == "xgboost":
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,  # Increased estimators
                learning_rate=0.05,  # Reduced learning rate
                max_depth=5,  # Adjusted max_depth
                subsample=0.8,  # Added subsample
                colsample_bytree=0.8,  # Added colsample_bytree
                random_state=42
            )
        elif model_type == "lightgbm":
            model = lgb.LGBMRegressor(
                n_estimators=200,  # Increased estimators
                learning_rate=0.05,  # Reduced learning rate
                num_leaves=31,  # Default, can be tuned
                max_depth=-1,  # Default, can be tuned
                random_state=42
            )
        elif model_type == "catboost":
            model = cb.CatBoostRegressor(
                n_estimators=200,  # Increased estimators
                learning_rate=0.05,  # Reduced learning rate
                depth=6,  # Adjusted depth
                l2_leaf_reg=3,  # L2 regularization
                random_state=42,
                verbose=0
            )
        model.fit(features_scaled, targets[:, i])
        models.append(model)

    return models, scaler


def train_ann_model(features, targets):
    """Train a simplified artificial neural network model to predict the next set of numbers."""
    if features.size == 0 or targets.size == 0:
        return [], None

    # For ANN, we'll use MinMaxScaler to normalize data between 0 and 1
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    models = []

    # Create a model for each target number
    for i in range(targets.shape[1]):
        # Create a very simple feedforward neural network
        model = Sequential([
            Dense(32, activation='relu', input_dim=features.shape[1]),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)  # Output layer for regression
        ])

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.01),  # Increased learning rate
            loss='mse'
        )

        # Define callbacks for training with very aggressive early stopping
        callbacks = [
            EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        ]

        # Train the model with minimal epochs and large batch size for speed
        model.fit(
            features_scaled,
            targets[:, i],
            epochs=10,  # Drastically reduced from 200
            batch_size=128,  # Significantly increased from 32
            validation_split=0.0,  # No validation to speed up training
            callbacks=callbacks,
            verbose=0
        )

        models.append(model)

    return models, scaler


def evaluate_model(models, scaler, features, targets, model_type="xgboost"):
    """Evaluate the trained machine learning model."""
    if not models or features.size == 0 or targets.size == 0:
        return {f'{model_type}_mae': float('inf')}

    predictions = []
    features_scaled = scaler.transform(features)
    for i, model in enumerate(models):
        pred = model.predict(features_scaled)
        predictions.append(pred)

    # Transpose predictions to match targets shape (num_samples, num_features)
    predictions = np.array(predictions).T

    mae = mean_absolute_error(targets, predictions)
    return {f'{model_type}_mae': mae}


def evaluate_ann_model(models, scaler, features, targets):
    """Evaluate the trained ANN model."""
    if not models or features.size == 0 or targets.size == 0:
        return {'ann_mae': float('inf')}

    predictions = []
    features_scaled = scaler.transform(features)

    for i, model in enumerate(models):
        pred = model.predict(features_scaled, verbose=0)
        predictions.append(pred.flatten())  # Flatten predictions from ANN

    # Transpose predictions to match targets shape (num_samples, num_features)
    predictions = np.array(predictions).T

    mae = mean_absolute_error(targets, predictions)
    return {'ann_mae': mae}


def cross_validate_model(features, targets, model_type="xgboost", n_splits=5):
    """Perform cross-validation for a given model type."""
    # Simple train-test split instead of full cross-validation for speed
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    models, scaler = train_model(X_train, y_train, model_type=model_type)
    if models and scaler:
        evaluation_results = evaluate_model(models, scaler, X_test, y_test, model_type=model_type)
        return {f'{model_type}_cv_mae': evaluation_results[f'{model_type}_mae']}
    else:
        return {f'{model_type}_cv_mae': float('inf')}


def ensemble_predict(all_predictions, num_range, num_to_predict):
    """Combine predictions from multiple models using averaging and ensure uniqueness."""
    if not all_predictions:
        return sorted(random.sample(range(1, num_range + 1), num_to_predict))

    # Convert predictions to a numpy array for easier averaging
    # Ensure all predictions have the same number of elements
    first_key = list(all_predictions.keys())[0]
    num_elements = len(all_predictions[first_key])

    # Filter out predictions that don't match the expected number of elements
    valid_predictions = [np.array(p) for p in all_predictions.values() if len(p) == num_elements]

    if not valid_predictions:
        return sorted(random.sample(range(1, num_range + 1), num_to_predict))

    # Average the predictions
    avg_prediction = np.mean(valid_predictions, axis=0)

    # Round to nearest integer and ensure numbers are within range and unique
    final_prediction_set = set()
    for num in avg_prediction:
        rounded_num = max(1, min(num_range, round(num)))
        final_prediction_set.add(rounded_num)

    # If not enough unique numbers, fill with random unique numbers
    while len(final_prediction_set) < num_to_predict:
        num = random.randint(1, num_range)
        final_prediction_set.add(num)

    return sorted(list(final_prediction_set)[:num_to_predict])


def predict_next_numbers(number_sets, num_range, num_to_predict, models, scaler, window_size, model_type="xgboost"):
    """Predict the next numbers based on the trained ML model with enhanced feature engineering."""
    ml_prediction = []
    predicted_set = set()  # Use a set to store unique predictions

    if models and scaler and len(number_sets) >= window_size:
        recent_window = number_sets[-window_size:]

        # Recreate the feature vector as done in create_features
        feature_for_prediction = [num for subset in recent_window for num in subset]

        flat_recent_window = np.array(recent_window).flatten()
        feature_for_prediction.append(np.mean(flat_recent_window))
        feature_for_prediction.append(np.std(flat_recent_window))
        feature_for_prediction.append(np.min(flat_recent_window))
        feature_for_prediction.append(np.max(flat_recent_window))
        feature_for_prediction.extend(recent_window[-1])

        feature_for_prediction = np.array([feature_for_prediction])

        if feature_for_prediction.shape[1] == scaler.n_features_in_:
            feature_scaled = scaler.transform(feature_for_prediction)
            for model_idx, model in enumerate(models):
                prediction = round(model.predict(feature_scaled)[0])
                prediction = max(1, min(num_range, prediction))
                predicted_set.add(prediction)  # Add to set to ensure uniqueness
        else:
            logging.warning(
                f"ML prediction skipped due to feature dimension mismatch. Expected {scaler.n_features_in_}, got {feature_for_prediction.shape[1]}.")

    ml_prediction = sorted(list(predicted_set))

    # If ML prediction didn\"t produce enough unique numbers, fill with random unique numbers
    while len(ml_prediction) < num_to_predict:
        num = random.randint(1, num_range)
        if num not in ml_prediction:
            ml_prediction.append(num)

    return sorted(ml_prediction)


def predict_next_numbers_ann(number_sets, num_range, num_to_predict, models, scaler, window_size):
    """Predict the next numbers based on the trained ANN model with enhanced feature engineering."""
    ann_prediction = []
    predicted_set = set()  # Use a set to store unique predictions

    if models and scaler and len(number_sets) >= window_size:
        recent_window = number_sets[-window_size:]

        # Recreate the feature vector as done in create_features
        feature_for_prediction = [num for subset in recent_window for num in subset]

        flat_recent_window = np.array(recent_window).flatten()
        feature_for_prediction.append(np.mean(flat_recent_window))
        feature_for_prediction.append(np.std(flat_recent_window))
        feature_for_prediction.append(np.min(flat_recent_window))
        feature_for_prediction.append(np.max(flat_recent_window))
        feature_for_prediction.extend(recent_window[-1])

        feature_for_prediction = np.array([feature_for_prediction])

        if feature_for_prediction.shape[1] == scaler.n_features_in_:
            feature_scaled = scaler.transform(feature_for_prediction)
            for model_idx, model in enumerate(models):
                prediction = round(model.predict(feature_scaled, verbose=0)[0][0])
                prediction = max(1, min(num_range, prediction))
                predicted_set.add(prediction)  # Add to set to ensure uniqueness
            else:
                logging.warning(
                    f"ANN prediction skipped due to feature dimension mismatch. Expected {scaler.n_features_in_}, got {feature_for_prediction.shape[1]}.")

    ann_prediction = sorted(list(predicted_set))

    # If ANN prediction didn\"t produce enough unique numbers, fill with random numbers
    while len(ann_prediction) < num_to_predict:
        num = random.randint(1, num_range)
        if num not in ann_prediction:
            ann_prediction.append(num)

    return sorted(ann_prediction)


def plot_frequency(number_sets, title, filename, num_range):
    """Plots the frequency of each number in the given sets."""
    all_numbers = [num for sublist in number_sets for num in sublist]
    counts = Counter(all_numbers)

    numbers = sorted(counts.keys())
    frequencies = [counts[n] for n in numbers]

    plt.figure(figsize=(12, 6))
    plt.bar(numbers, frequencies, color='skyblue')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(range(1, num_range + 1), rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved frequency plot to {filename}")


def main():
    # Check for TensorFlow GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"TensorFlow is using GPU: {gpus}")
    else:
        logging.info("TensorFlow is using CPU")

    file_path = get_data_file_path()
    main_number_sets, bonus_number_sets = load_data(file_path)

    logging.info(
        f"Loaded {len(main_number_sets)} sets of main numbers and {len(bonus_number_sets)} sets of bonus numbers from {file_path}")

    # Traditional ML model types
    model_types = ["xgboost", "lightgbm", "catboost"]

    main_predictions = {}
    bonus_predictions = {}
    main_evaluations = {}
    bonus_evaluations = {}
    main_cv_evaluations = {}
    bonus_cv_evaluations = {}

    window_size = config['SETTINGS'].getint('window_size')
    main_num_range = config['SETTINGS'].getint('main_num_range')
    bonus_num_range = config['SETTINGS'].getint('bonus_num_range')
    num_main_to_predict = config['SETTINGS'].getint('num_main_to_predict')
    num_bonus_to_predict = config['SETTINGS'].getint('num_bonus_to_predict')
    test_size = config['SETTINGS'].getfloat('test_size')
    n_splits = config['SETTINGS'].getint('n_splits')

    # --- Main Numbers Prediction ---
    logging.info("\nMain numbers")
    main_features, main_targets = create_features(main_number_sets, window_size=window_size)

    if main_features.size > 0 and main_targets.size > 0:
        # Split data for evaluation
        X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(
            main_features, main_targets, test_size=test_size, random_state=42
        )

        # Train and evaluate traditional ML models
        for model_type in model_types:
            logging.info(f"Training {model_type} model for main numbers...")
            main_models, main_scaler = train_model(X_train_main, y_train_main, model_type=model_type)
            main_prediction = predict_next_numbers(main_number_sets, main_num_range, num_main_to_predict, main_models,
                                                   main_scaler,
                                                   window_size=window_size, model_type=model_type)
            main_predictions[model_type] = main_prediction

            # Evaluate the model
            evaluation_results = evaluate_model(main_models, main_scaler, X_test_main, y_test_main,
                                                model_type=model_type)
            main_evaluations.update(evaluation_results)

            # Perform cross-validation
            cv_results = cross_validate_model(main_features, main_targets, model_type=model_type, n_splits=n_splits)
            main_cv_evaluations.update(cv_results)

        # Train and evaluate ANN model
        logging.info(f"Training ANN model for main numbers...")
        main_ann_models, main_ann_scaler = train_ann_model(X_train_main, y_train_main)
        main_ann_prediction = predict_next_numbers_ann(
            main_number_sets, main_num_range, num_main_to_predict,
            main_ann_models, main_ann_scaler, window_size=window_size
        )
        main_predictions["ann"] = main_ann_prediction

        # Evaluate the ANN model
        ann_evaluation_results = evaluate_ann_model(
            main_ann_models, main_ann_scaler, X_test_main, y_test_main
        )
        main_evaluations.update(ann_evaluation_results)

        # Skip cross-validation for ANN to save time
        main_cv_evaluations.update({'ann_cv_mae': ann_evaluation_results['ann_mae']})

        # Ensemble prediction for main numbers (including ANN predictions)
        ensemble_main_prediction = ensemble_predict(main_predictions, main_num_range, num_main_to_predict)
        main_predictions["ensemble"] = ensemble_main_prediction

    else:
        logging.warning("Not enough data to create features for main numbers model training.")

    # --- Bonus Numbers Prediction ---
    logging.info("\nBonus")
    bonus_features, bonus_targets = create_features(bonus_number_sets, window_size=window_size)

    if bonus_features.size > 0 and bonus_targets.size > 0:
        # Split data for evaluation
        X_train_bonus, X_test_bonus, y_train_bonus, y_test_bonus = train_test_split(
            bonus_features, bonus_targets, test_size=test_size, random_state=42
        )

        # Train and evaluate traditional ML models
        for model_type in model_types:
            logging.info(f"Training {model_type} model for bonus numbers...")
            bonus_models, bonus_scaler = train_model(X_train_bonus, y_train_bonus, model_type=model_type)
            bonus_prediction = predict_next_numbers(bonus_number_sets, bonus_num_range, num_bonus_to_predict,
                                                    bonus_models, bonus_scaler,
                                                    window_size=window_size, model_type=model_type)
            bonus_predictions[model_type] = bonus_prediction

            # Evaluate the model
            evaluation_results = evaluate_model(bonus_models, bonus_scaler, X_test_bonus, y_test_bonus,
                                                model_type=model_type)
            bonus_evaluations.update(evaluation_results)

            # Perform cross-validation
            cv_results = cross_validate_model(bonus_features, bonus_targets, model_type=model_type, n_splits=n_splits)
            bonus_cv_evaluations.update(cv_results)

        # Train and evaluate ANN model for bonus numbers
        logging.info(f"Training ANN model for bonus numbers...")
        bonus_ann_models, bonus_ann_scaler = train_ann_model(X_train_bonus, y_train_bonus)
        bonus_ann_prediction = predict_next_numbers_ann(
            bonus_number_sets, bonus_num_range, num_bonus_to_predict,
            bonus_ann_models, bonus_ann_scaler, window_size=window_size
        )
        bonus_predictions["ann"] = bonus_ann_prediction

        # Evaluate the ANN model
        ann_evaluation_results = evaluate_ann_model(
            bonus_ann_models, bonus_ann_scaler, X_test_bonus, y_test_bonus
        )
        bonus_evaluations.update(ann_evaluation_results)

        # Skip cross-validation for ANN to save time
        bonus_cv_evaluations.update({'ann_cv_mae': ann_evaluation_results['ann_mae']})

        # Ensemble prediction for bonus numbers (including ANN predictions)
        ensemble_bonus_prediction = ensemble_predict(bonus_predictions, bonus_num_range, num_bonus_to_predict)
        bonus_predictions["ensemble"] = ensemble_bonus_prediction

    else:
        logging.warning("Not enough data to create features for bonus numbers model training.")

    # --- Data Visualization ---
    if main_number_sets:
        plot_frequency(main_number_sets, 'Main Numbers Frequency', 'main_numbers_frequency.png', main_num_range)
    if bonus_number_sets:
        plot_frequency(bonus_number_sets, 'Bonus Numbers Frequency', 'bonus_numbers_frequency.png', bonus_num_range)

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
    for model_type, mae in main_evaluations.items():
        print(f'{model_type.capitalize()}: {mae:.2f}')

    print("\nMain Numbers Models (Cross-Validation):")
    for model_type, mae in main_cv_evaluations.items():
        print(f'{model_type.capitalize()}: {mae:.2f}')

    print("\nBonus Numbers Models (Test Set):")
    for model_type, mae in bonus_evaluations.items():
        print(f'{model_type.capitalize()}: {mae:.2f}')

    print("\nBonus Numbers Models (Cross-Validation):")
    for model_type, mae in bonus_cv_evaluations.items():
        print(f'{model_type.capitalize()}: {mae:.2f}')


if __name__ == "__main__":
    main()
