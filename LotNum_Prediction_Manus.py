#!/usr/bin/env python3
"""
Lottery Number Prediction Script

This script analyzes historical lottery number data and predicts the next set of numbers
based on frequency analysis and statistical patterns.
- Main numbers: 5 numbers in the range of 1-50
- Bonus numbers: 2 numbers in the range of 1-12

The script automatically uses data.txt in the same directory as the script.
"""

import numpy as np
import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load and parse the data file containing lottery number selections.
    Each line should contain 7 numbers: 5 main numbers (1-50) followed by 2 bonus numbers (1-12).
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Parse each line into main numbers and bonus numbers
        main_number_sets = []
        bonus_number_sets = []

        for line in lines:
            numbers = [int(num) for num in line.strip().split() if num.isdigit()]
            if len(numbers) >= 7:  # Ensure we have at least 7 numbers
                main_numbers = numbers[:5]  # First 5 are main numbers
                bonus_numbers = numbers[5:7]  # Last 2 are bonus numbers

                # Validate ranges
                if all(1 <= n <= 50 for n in main_numbers) and all(1 <= n <= 12 for n in bonus_numbers):
                    main_number_sets.append(main_numbers)
                    bonus_number_sets.append(bonus_numbers)
                else:
                    print(f"Warning: Line with invalid number ranges skipped: {numbers}")
            else:
                print(f"Warning: Line with insufficient numbers skipped: {numbers}")

        return main_number_sets, bonus_number_sets
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)


def analyze_frequency(number_sets, max_number):
    """
    Analyze the frequency of each number in the dataset.

    Args:
        number_sets: List of number sets to analyze
        max_number: Maximum number in the range (50 for main, 12 for bonus)

    Returns:
        Dictionary with number frequencies
    """
    # Flatten the list of lists to get all numbers
    all_numbers = [num for subset in number_sets for num in subset]

    # Count frequency of each number
    counter = Counter(all_numbers)

    # Create a frequency dictionary for all numbers in the range
    frequency = {i: counter.get(i, 0) for i in range(1, max_number + 1)}

    return frequency


def analyze_patterns(number_sets):
    """
    Analyze patterns in the data such as common pairs and sequences.

    Args:
        number_sets: List of number sets to analyze

    Returns:
        List of common pairs with their frequencies
    """
    pairs = []
    for subset in number_sets:
        # Generate all possible pairs in each subset
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                pairs.append((min(subset[i], subset[j]), max(subset[i], subset[j])))

    # Count frequency of each pair
    pair_counter = Counter(pairs)

    # Get the most common pairs
    common_pairs = pair_counter.most_common(10)

    return common_pairs


def calculate_statistics(number_sets):
    """
    Calculate various statistics about the number sets.

    Args:
        number_sets: List of number sets to analyze

    Returns:
        Dictionary with statistical measures
    """
    # Convert to numpy array for easier calculations
    data = np.array(number_sets)

    stats = {
        'mean': np.mean(data, axis=0),
        'median': np.median(data, axis=0),
        'std': np.std(data, axis=0),
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }

    return stats


def create_features(number_sets, window_size=5):
    """
    Create features for machine learning prediction.

    Args:
        number_sets: List of number sets to analyze
        window_size: Number of previous draws to use as features

    Returns:
        Features and targets for machine learning
    """
    features = []
    targets = []

    for i in range(len(number_sets) - window_size):
        # Flatten the window of sets into a feature vector
        window = number_sets[i:i + window_size]
        feature = [num for subset in window for num in subset]

        # The target is the next set after the window
        target = number_sets[i + window_size]

        features.append(feature)
        targets.append(target)

    return np.array(features), np.array(targets)


def train_model(features, targets):
    """
    Train a machine learning model to predict the next set of numbers.

    Args:
        features: Feature matrix for training
        targets: Target values for training

    Returns:
        Trained models and feature scaler
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train a random forest regressor for each position in the target
    models = []
    for i in range(targets.shape[1]):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features_scaled, targets[:, i])
        models.append(model)

    return models, scaler


def predict_next_numbers(number_sets, frequency, common_pairs, max_number, count, models=None, scaler=None,
                         window_size=5):
    """
    Predict the next set of numbers based on various methods.

    Args:
        number_sets: Historical number sets
        frequency: Dictionary of number frequencies
        common_pairs: List of common number pairs
        max_number: Maximum number in the range
        count: Number of numbers to predict
        models: Trained machine learning models (optional)
        scaler: Feature scaler for ML models (optional)
        window_size: Window size for ML prediction (optional)

    Returns:
        Dictionary with predictions from different methods
    """
    # Method 1: Frequency-based prediction
    most_frequent = [num for num, _ in sorted(frequency.items(), key=lambda x: x[1], reverse=True)]

    # Method 2: Recent trend analysis
    recent_sets = number_sets[-5:]
    recent_numbers = [num for subset in recent_sets for num in subset]
    recent_frequency = Counter(recent_numbers)
    trending = [num for num, _ in sorted(recent_frequency.items(), key=lambda x: x[1], reverse=True)]

    # Method 3: Machine learning prediction (if models are provided)
    ml_prediction = []
    if models and scaler and len(number_sets) >= window_size:
        # Create feature from the most recent window
        recent_window = number_sets[-window_size:]
        feature = np.array([[num for subset in recent_window for num in subset]])
        feature_scaled = scaler.transform(feature)

        # Predict using each model
        for model in models:
            prediction = round(model.predict(feature_scaled)[0])
            # Ensure prediction is within range
            prediction = max(1, min(max_number, prediction))
            ml_prediction.append(prediction)

    # Method 4: Statistical sampling based on frequency
    statistical_sample = random.choices(
        population=list(range(1, max_number + 1)),
        weights=[frequency.get(i, 0) + 1 for i in range(1, max_number + 1)],  # Add 1 to avoid zero weights
        k=count
    )

    # Method 5: Pair-based prediction
    pair_based = []
    if common_pairs:
        # Extract numbers from common pairs
        pair_numbers = set()
        for pair, _ in common_pairs:
            pair_numbers.update(pair)

        # Select from these numbers
        pair_based = list(pair_numbers)[:count]

        # If we don't have enough numbers, fill with most frequent
        if len(pair_based) < count:
            for num in most_frequent:
                if num not in pair_based:
                    pair_based.append(num)
                    if len(pair_based) == count:
                        break

    # Combine predictions from different methods
    combined_prediction = []

    # Add some numbers from frequency analysis
    for num in most_frequent[:max(1, count // 3)]:
        if num not in combined_prediction and len(combined_prediction) < count:
            combined_prediction.append(num)

    # Add some trending numbers
    for num in trending[:max(1, count // 3)]:
        if num not in combined_prediction and len(combined_prediction) < count:
            combined_prediction.append(num)

    # Add some numbers from ML prediction if available
    if ml_prediction:
        for num in ml_prediction:
            if num not in combined_prediction and len(combined_prediction) < count:
                combined_prediction.append(num)

    # Fill remaining slots with statistical sampling
    for num in statistical_sample:
        if num not in combined_prediction and len(combined_prediction) < count:
            combined_prediction.append(num)

    # Ensure we have exactly the required number of predictions
    while len(combined_prediction) < count:
        num = random.randint(1, max_number)
        if num not in combined_prediction:
            combined_prediction.append(num)

    # Sort the prediction
    combined_prediction.sort()

    return {
        'frequency_based': sorted(most_frequent[:count]),
        'trend_based': sorted(trending[:count]),
        'ml_based': sorted(ml_prediction) if ml_prediction else None,
        'statistical_sample': sorted(statistical_sample),
        'pair_based': sorted(pair_based) if pair_based else None,
        'combined_prediction': combined_prediction
    }


def visualize_frequency(main_frequency, bonus_frequency, output_dir='.'):
    """
    Create visualizations of number frequency for both main and bonus numbers.

    Args:
        main_frequency: Dictionary of main number frequencies
        bonus_frequency: Dictionary of bonus number frequencies
        output_dir: Directory to save the output files
    """
    # Visualize main numbers
    plt.figure(figsize=(12, 6))
    numbers = list(range(1, 51))
    frequencies = [main_frequency.get(num, 0) for num in numbers]

    plt.bar(numbers, frequencies)
    plt.xlabel('Main Number')
    plt.ylabel('Frequency')
    plt.title('Frequency of Main Numbers (1-50) in Historical Data')
    plt.xticks(range(0, 51, 5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    main_output_file = os.path.join(output_dir, 'main_number_frequency.png')
    plt.savefig(main_output_file)
    print(f"\nMain number frequency visualization saved as '{main_output_file}'")
    plt.close()

    # Visualize bonus numbers
    plt.figure(figsize=(10, 5))
    numbers = list(range(1, 13))
    frequencies = [bonus_frequency.get(num, 0) for num in numbers]

    plt.bar(numbers, frequencies)
    plt.xlabel('Bonus Number')
    plt.ylabel('Frequency')
    plt.title('Frequency of Bonus Numbers (1-12) in Historical Data')
    plt.xticks(range(1, 13))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    bonus_output_file = os.path.join(output_dir, 'bonus_number_frequency.png')
    plt.savefig(bonus_output_file)
    print(f"Bonus number frequency visualization saved as '{bonus_output_file}'")
    plt.close()


def main():
    # Use data.txt in the same directory as the script by default
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_file = os.path.join(script_dir, 'data.txt')

    # Check if a command line argument was provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_data_file

    print(f"Using data file: {file_path}")

    # Load and analyze the data
    main_number_sets, bonus_number_sets = load_data(file_path)

    print(f"Loaded {len(main_number_sets)} sets of lottery numbers from {file_path}")

    # Analyze frequency for main numbers
    main_frequency = analyze_frequency(main_number_sets, 50)
    print("\nTop 10 most frequent main numbers (1-50):")
    for num, freq in sorted(main_frequency.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Number {num}: {freq} occurrences")

    # Analyze frequency for bonus numbers
    bonus_frequency = analyze_frequency(bonus_number_sets, 12)
    print("\nMost frequent bonus numbers (1-12):")
    for num, freq in sorted(bonus_frequency.items(), key=lambda x: x[1], reverse=True):
        print(f"Number {num}: {freq} occurrences")

    # Analyze patterns for main numbers
    main_common_pairs = analyze_patterns(main_number_sets)
    print("\nTop 10 most common main number pairs:")
    for pair, count in main_common_pairs:
        print(f"Pair {pair}: {count} occurrences")

    # Analyze patterns for bonus numbers
    bonus_common_pairs = analyze_patterns(bonus_number_sets)
    print("\nMost common bonus number pairs:")
    for pair, count in bonus_common_pairs:
        print(f"Pair {pair}: {count} occurrences")

    # Calculate statistics for main and bonus numbers
    main_stats = calculate_statistics(main_number_sets)
    print("\nStatistics of main number sets:")
    for stat_name, stat_values in main_stats.items():
        print(f"{stat_name}: {stat_values}")

    bonus_stats = calculate_statistics(bonus_number_sets)
    print("\nStatistics of bonus number sets:")
    for stat_name, stat_values in bonus_stats.items():
        print(f"{stat_name}: {stat_values}")

    # Create features and train model for main numbers if enough data
    main_models = None
    main_scaler = None
    if len(main_number_sets) > 10:  # Need enough data for meaningful ML
        main_features, main_targets = create_features(main_number_sets)
        if len(main_features) > 0:
            print(f"\nTraining machine learning model for main numbers with {len(main_features)} samples...")
            main_models, main_scaler = train_model(main_features, main_targets)

    # Create features and train model for bonus numbers if enough data
    bonus_models = None
    bonus_scaler = None
    if len(bonus_number_sets) > 10:  # Need enough data for meaningful ML
        bonus_features, bonus_targets = create_features(bonus_number_sets)
        if len(bonus_features) > 0:
            print(f"\nTraining machine learning model for bonus numbers with {len(bonus_features)} samples...")
            bonus_models, bonus_scaler = train_model(bonus_features, bonus_targets)

    # Predict next main numbers (5 numbers in range 1-50)
    main_predictions = predict_next_numbers(
        main_number_sets,
        main_frequency,
        main_common_pairs,
        max_number=50,
        count=5,
        models=main_models,
        scaler=main_scaler
    )

    print("\nPredictions for next 5 main numbers (1-50):")
    for method, prediction in main_predictions.items():
        if prediction:
            print(f"{method}: {prediction}")

    # Predict next bonus numbers (2 numbers in range 1-12)
    bonus_predictions = predict_next_numbers(
        bonus_number_sets,
        bonus_frequency,
        bonus_common_pairs,
        max_number=12,
        count=2,
        models=bonus_models,
        scaler=bonus_scaler
    )

    print("\nPredictions for next 2 bonus numbers (1-12):")
    for method, prediction in bonus_predictions.items():
        if prediction:
            print(f"{method}: {prediction}")

    # Visualize frequency
    output_dir = os.path.dirname(os.path.abspath(file_path)) if os.path.dirname(file_path) else '.'
    visualize_frequency(main_frequency, bonus_frequency, output_dir)

    print("\nFinal prediction for next lottery draw:")
    print(f"Main numbers (1-50): {main_predictions['combined_prediction']}")
    print(f"Bonus numbers (1-12): {bonus_predictions['combined_prediction']}")


if __name__ == "__main__":
    main()
5