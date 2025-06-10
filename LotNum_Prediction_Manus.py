#!/usr/bin/env python3
"""
Number Prediction Script

This script analyzes historical number data to predict the next set of numbers.
It predicts 5 main numbers in the range of 1-50 and 2 bonus numbers in the
range of 1-12, based on frequency analysis, statistical patterns, and
XGBoost, LightGBM, and CatBoost machine learning models.
"""

import numpy as np
import pandas as pd
from collections import Counter
import random
import os
import sys
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


def load_data(file_path):
    """Load and parse the data file containing number selections."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        main_number_sets = []
        bonus_number_sets = []
        for line in lines:
            numbers = [int(num) for num in line.strip().split() if num.isdigit()]
            if len(numbers) == 7:  # Expecting 5 main and 2 bonus numbers
                main_number_sets.append(numbers[:5])
                bonus_number_sets.append(numbers[5:])
            elif numbers:  # Only add non-empty sets if not 7 numbers
                print(f"Warning: Skipping line due to incorrect number count: {line.strip()}")

        return main_number_sets, bonus_number_sets
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please ensure 'data.txt' is in the same directory as the script.")
        sys.exit(1)  # Exit if file not found


def create_features(number_sets, window_size=5):
    """Create features for machine learning prediction."""
    features = []
    targets = []

    if len(number_sets) <= window_size:
        return np.array([]), np.array([])

    for i in range(len(number_sets) - window_size):
        window = number_sets[i:i + window_size]
        feature = [num for subset in window for num in subset]
        target = number_sets[i + window_size]

        features.append(feature)
        targets.append(target)

    return np.array(features), np.array(targets)


def train_model(features, targets, model_type="xgboost"):
    """Train a machine learning model to predict the next set of numbers."""
    if features.size == 0 or targets.size == 0:
        return [], None

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    models = []
    for i in range(targets.shape[1]):
        if model_type == "xgboost":
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        elif model_type == "lightgbm":
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        elif model_type == "catboost":
            model = cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)
        model.fit(features_scaled, targets[:, i])
        models.append(model)

    return models, scaler


def predict_next_numbers(number_sets, num_range, num_to_predict, models, scaler, window_size=5):
    """Predict the next numbers based on the trained ML model."""
    ml_prediction = []
    if models and scaler and len(number_sets) >= window_size:
        recent_window = number_sets[-window_size:]
        flat_recent_window = [num for subset in recent_window for num in subset]

        if flat_recent_window:
            feature_for_prediction = np.array([flat_recent_window])

            if feature_for_prediction.shape[1] == scaler.n_features_in_:
                feature_scaled = scaler.transform(feature_for_prediction)
                for model_idx, model in enumerate(models):
                    prediction = round(model.predict(feature_scaled)[0])
                    prediction = max(1, min(num_range, prediction))
                    ml_prediction.append(prediction)
            else:
                print(
                    f"Warning: ML prediction skipped due to feature dimension mismatch. Expected {scaler.n_features_in_}, got {feature_for_prediction.shape[1]}.")
                ml_prediction = []

    # If ML prediction didn't produce enough numbers (e.g., due to insufficient data),
    # fill with random numbers within the range to meet num_to_predict.
    while len(ml_prediction) < num_to_predict:
        num = random.randint(1, num_range)
        if num not in ml_prediction:
            ml_prediction.append(num)

    return sorted(ml_prediction)


def get_data_file_path():
    """Get the path to the data file. Assumes data.txt is in the script's directory if no arg given."""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not os.path.isfile(file_path):
            print(f"Error: File not found at {file_path}")
            print("Please provide a valid path or ensure data.txt is in the script directory.")
            sys.exit(1)
        return file_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "data.txt")
        if not os.path.isfile(file_path):
            print(f"Error: data.txt not found in script directory: {script_dir}")
            print(
                "Please place data.txt in the same directory as the script or provide the path as a command line argument.")
            sys.exit(1)
        return file_path


def main():
    file_path = get_data_file_path()
    main_number_sets, bonus_number_sets = load_data(file_path)

    print(
        f"Loaded {len(main_number_sets)} sets of main numbers and {len(bonus_number_sets)} sets of bonus numbers from {file_path}")

    model_types = ["xgboost", "lightgbm", "catboost"]

    # --- Main Numbers Prediction ---
    print("\nMain numbers")
    main_features, main_targets = create_features(main_number_sets, window_size=5)

    if main_features.size > 0 and main_targets.size > 0:
        for model_type in model_types:
            print(f"Training {model_type} model for main numbers...")
            main_models, main_scaler = train_model(main_features, main_targets, model_type=model_type)
            main_prediction = predict_next_numbers(main_number_sets, 50, 5, main_models, main_scaler)
            print(f'{model_type.capitalize()}: {" ".join(map(str, main_prediction))}')
    else:
        print("Not enough data to create features for main numbers model training.")

    # --- Bonus Numbers Prediction ---
    print("\nBonus")
    bonus_features, bonus_targets = create_features(bonus_number_sets, window_size=5)

    if bonus_features.size > 0 and bonus_targets.size > 0:
        for model_type in model_types:
            print(f"Training {model_type} model for bonus numbers...")
            bonus_models, bonus_scaler = train_model(bonus_features, bonus_targets, model_type=model_type)
            bonus_prediction = predict_next_numbers(bonus_number_sets, 12, 2, bonus_models, bonus_scaler)
            print(f'{model_type.capitalize()}: {" ".join(map(str, bonus_prediction))}')
    else:
        print("Not enough data to create features for bonus numbers model training.")


if __name__ == "__main__":
    main()


