import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import random
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import itertools
import joblib
import os
import warnings
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_lottery_data(file_path):
    """
    Load lottery data from a text file with improved error handling and validation.
    Assumes each line contains a drawing with space-separated or comma-separated numbers.
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Process each line into a list of numbers
        drawings = []
        drawing_sizes = []
        invalid_lines = 0

        for i, line in enumerate(lines):
            # Remove whitespace and split by spaces or commas
            line = line.strip()
            if ',' in line:
                numbers = [int(num.strip()) for num in line.split(',') if num.strip().isdigit()]
            else:
                numbers = [int(num) for num in line.split() if num.isdigit()]

            if numbers:  # Only add if we have numbers
                drawings.append(numbers)
                drawing_sizes.append(len(numbers))
            else:
                invalid_lines += 1

        if invalid_lines > 0:
            logger.warning(f"Skipped {invalid_lines} invalid lines while loading data")

        if not drawings:
            logger.error("No valid drawings found in file")
            return []

        # Check consistency of drawing sizes
        most_common_size = Counter(drawing_sizes).most_common(1)[0][0]
        inconsistent_lines = sum(1 for size in drawing_sizes if size != most_common_size)

        if inconsistent_lines > 0:
            logger.warning(
                f"Found {inconsistent_lines} drawings with inconsistent size (most common: {most_common_size})")

        # Validate that all numbers are within a reasonable range
        all_numbers = [num for drawing in drawings for num in drawing]
        max_num = max(all_numbers) if all_numbers else 0
        min_num = min(all_numbers) if all_numbers else 0

        if max_num > 100:  # Arbitrary reasonable limit for lottery numbers
            logger.warning(f"Found unusually large numbers (max: {max_num})")

        if min_num < 1:
            logger.warning(f"Found numbers below 1 (min: {min_num})")

        logger.info(f"Successfully loaded {len(drawings)} drawings with typical size of {most_common_size}")
        return drawings

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []


def analyze_frequency(drawings, max_number=50, window_size=None):
    """
    Analyze the frequency of each number in the drawings with optional windowing.

    Args:
        drawings: List of drawings
        max_number: Maximum possible number in the lottery
        window_size: If provided, only analyze the last n drawings

    Returns:
        Dictionary of frequencies and normalized frequencies
    """
    if window_size and len(drawings) > window_size:
        data_to_analyze = drawings[-window_size:]
    else:
        data_to_analyze = drawings

    all_numbers = [num for drawing in data_to_analyze for num in drawing]
    freq = Counter(all_numbers)

    # Fill in zeros for numbers that never appeared
    for i in range(1, max_number + 1):
        if i not in freq:
            freq[i] = 0

    # Calculate normalized frequencies (as probabilities)
    total_nums = sum(freq.values())
    if total_nums > 0:  # Avoid division by zero
        norm_freq = {num: count / total_nums for num, count in freq.items()}
    else:
        norm_freq = {num: 0 for num in freq}

    return {'raw': freq, 'normalized': norm_freq}


def analyze_recency(drawings, max_number=50, decay_rates=[0.95, 0.9, 0.8]):
    """
    Analyze how recently each number has appeared with multiple exponential weighting factors.
    More recent occurrences get higher weight.

    Returns scores at different decay rates to capture short and longer-term patterns.
    """
    results = {}

    for decay_rate in decay_rates:
        recency_scores = {i: 0 for i in range(1, max_number + 1)}

        for i, drawing in enumerate(reversed(drawings)):
            weight = decay_rate ** i  # Exponential decay
            for num in drawing:
                if 1 <= num <= max_number:  # Ensure number is in valid range
                    recency_scores[num] += weight

        # Normalize scores
        max_score = max(recency_scores.values()) if recency_scores else 1
        if max_score > 0:
            normalized_scores = {num: score / max_score for num, score in recency_scores.items()}
        else:
            normalized_scores = recency_scores

        results[f'decay_{decay_rate}'] = normalized_scores

    return results


def analyze_gaps(drawings, max_number=50):
    """
    Analyze the gap patterns between appearances of each number with improved statistics.
    Returns the average gap, median gap, standard deviation, and current gap for each number.
    """
    # Initialize
    last_seen = {i: None for i in range(1, max_number + 1)}
    gaps = {i: [] for i in range(1, max_number + 1)}
    current_gaps = {i: 0 for i in range(1, max_number + 1)}

    # Calculate historical gaps
    for i, drawing in enumerate(drawings):
        # Increment gaps for all numbers
        for num in range(1, max_number + 1):
            if last_seen[num] is not None:
                current_gaps[num] += 1

        # Update for numbers in this drawing
        for num in drawing:
            if 1 <= num <= max_number:  # Ensure number is in valid range
                if last_seen[num] is not None:
                    gaps[num].append(current_gaps[num])
                current_gaps[num] = 0
                last_seen[num] = i

    # Calculate gap statistics
    gap_stats = {}
    for num in range(1, max_number + 1):
        if gaps[num]:
            gap_stats[num] = {
                'mean': np.mean(gaps[num]),
                'median': np.median(gaps[num]),
                'std': np.std(gaps[num]) if len(gaps[num]) > 1 else 0,
                'min': min(gaps[num]),
                'max': max(gaps[num]),
                'current': current_gaps[num],
                'count': len(gaps[num])
            }
        else:
            gap_stats[num] = {
                'mean': float('inf'),
                'median': float('inf'),
                'std': 0,
                'min': 0,
                'max': 0,
                'current': current_gaps[num],
                'count': 0
            }

    return gap_stats


def analyze_combinations(drawings, drawing_size=5, max_number=50, combo_sizes=[2, 3], max_combinations=1000):
    """
    Analyze which combinations of numbers tend to appear together.
    Improved to analyze different sizes of combinations.

    Args:
        drawings: List of drawings
        drawing_size: Typical size of a drawing
        combo_sizes: List of combination sizes to analyze
        max_combinations: Maximum number of combinations to track

    Returns:
        Dictionary with results for each combination size
    """
    results = {}

    for combo_size in combo_sizes:
        if combo_size > drawing_size:
            continue  # Skip if combo size is larger than drawing size

        combo_counts = Counter()
        for drawing in drawings:
            if len(drawing) >= combo_size:
                combos = list(itertools.combinations(sorted(drawing), combo_size))
                combo_counts.update(combos)

        # Calculate probability of each combination
        total_combos = sum(combo_counts.values())
        if total_combos > 0:
            combo_probs = {combo: count / total_combos for combo, count in combo_counts.most_common(max_combinations)}
        else:
            combo_probs = {}

        results[combo_size] = {
            'counts': dict(combo_counts.most_common(max_combinations)),
            'probabilities': combo_probs
        }

    return results


def create_enhanced_features(drawings, max_number=50, window_sizes=[5, 10, 20]):
    """
    Create enhanced feature set for machine learning prediction with multiple time windows.
    """
    if len(drawings) < max(window_sizes) + 5:
        logger.warning(f"Not enough data for feature creation (need at least {max(window_sizes) + 5} drawings)")
        return np.array([]), np.array([])

    X = []
    y = []

    # Determine the minimum window required
    min_window = min(window_sizes)

    for i in range(min_window, len(drawings)):
        features = []

        # 1. Add features for different time windows
        for window in window_sizes:
            if i >= window:
                # Frequency features in this window
                window_drawings = drawings[i - window:i]
                window_freqs = analyze_frequency(window_drawings, max_number)['normalized']
                features.extend([window_freqs.get(num, 0) for num in range(1, max_number + 1)])

                # Add some aggregate statistics from the window
                flat_window = [num for draw in window_drawings for num in draw]
                if flat_window:
                    features.append(np.mean(flat_window))
                    features.append(np.std(flat_window))
                    features.append(min(flat_window))
                    features.append(max(flat_window))
                else:
                    features.extend([0, 0, 0, 0])  # Default values if no data

        # 2. Add gap features
        gap_stats = analyze_gaps(drawings[:i], max_number)
        for num in range(1, max_number + 1):
            stats = gap_stats.get(num, {'current': 0, 'mean': 0})

            # Gap ratio (how overdue is this number)
            if stats['mean'] and stats['mean'] != float('inf'):
                gap_ratio = stats['current'] / stats['mean']
            else:
                gap_ratio = 0

            features.append(gap_ratio)

        # 3. Add recency features
        recency_data = analyze_recency(drawings[:i], max_number)
        for decay_key, scores in recency_data.items():
            # Only use one decay rate to avoid making the feature vector too large
            if decay_key == 'decay_0.9':  # Medium-term decay
                features.extend([scores.get(num, 0) for num in range(1, max_number + 1)])

        # Target is the next drawing (one-hot encoded)
        target = np.zeros(max_number)
        for num in drawings[i]:
            if 1 <= num <= max_number:
                target[num - 1] = 1

        X.append(features)
        y.append(target)

    return np.array(X), np.array(y)


def train_advanced_ml_model(X, y, max_number=50):
    """
    Train an advanced ML model with proper cross-validation and hyperparameter tuning.

    Returns a trained model for each number position.
    """
    if len(X) < 50:  # Need sufficient data for reliable training
        logger.warning("Not enough data for reliable model training")
        return None

    try:
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train a model for each number
        models = []

        for i in range(max_number):
            # Check if this number appears enough times
            if sum(y_train[:, i]) > 10:
                logger.info(f"Training model for number {i + 1}")

                # Options for model selection
                model_options = [
                    GradientBoostingRegressor(random_state=42),
                    RandomForestRegressor(random_state=42)
                ]

                best_model = None
                best_score = -np.inf

                # Simple model selection
                for model_candidate in model_options:
                    model_candidate.fit(X_train_scaled, y_train[:, i])
                    score = model_candidate.score(X_val_scaled, y_val[:, i])

                    if score > best_score:
                        best_score = score
                        best_model = model_candidate

                models.append((i + 1, best_model, best_score, scaler))
                logger.info(f"Model for number {i + 1} achieved R² score: {best_score:.4f}")
            else:
                models.append((i + 1, None, 0, scaler))

        return models

    except Exception as e:
        logger.error(f"Error training ML model: {e}")
        return None


def predict_with_advanced_ml(models, latest_features, max_number=50):
    """
    Use trained ML models to predict probabilities for each number.
    """
    if not models:
        return {}

    try:
        # Get the scaler from the first valid model
        scaler = None
        for _, model, _, scl in models:
            if model is not None:
                scaler = scl
                break

        if scaler is None:
            return {}

        # Scale the features
        latest_scaled = scaler.transform([latest_features])

        # Make predictions
        probabilities = {}

        for num, model, _, _ in models:
            if model is not None:
                prob = model.predict(latest_scaled)[0]
                # Ensure probabilities are between 0 and 1
                prob = max(0, min(1, prob))
                probabilities[num] = prob
            else:
                probabilities[num] = 0

        return probabilities

    except Exception as e:
        logger.error(f"Error making ML predictions: {e}")
        return {}


def create_latest_features(drawings, max_number=50, window_sizes=[5, 10, 20]):
    """
    Create feature vector for the latest data point using the same logic as in training.
    """
    if len(drawings) < max(window_sizes):
        return None

    features = []

    # 1. Add features for different time windows
    for window in window_sizes:
        if len(drawings) >= window:
            # Frequency features in this window
            window_drawings = drawings[-window:]
            window_freqs = analyze_frequency(window_drawings, max_number)['normalized']
            features.extend([window_freqs.get(num, 0) for num in range(1, max_number + 1)])

            # Add some aggregate statistics from the window
            flat_window = [num for draw in window_drawings for num in draw]
            if flat_window:
                features.append(np.mean(flat_window))
                features.append(np.std(flat_window))
                features.append(min(flat_window))
                features.append(max(flat_window))
            else:
                features.extend([0, 0, 0, 0])  # Default values if no data

    # 2. Add gap features
    gap_stats = analyze_gaps(drawings, max_number)
    for num in range(1, max_number + 1):
        stats = gap_stats.get(num, {'current': 0, 'mean': 0})

        # Gap ratio (how overdue is this number)
        if stats['mean'] and stats['mean'] != float('inf'):
            gap_ratio = stats['current'] / stats['mean']
        else:
            gap_ratio = 0

        features.append(gap_ratio)

    # 3. Add recency features
    recency_data = analyze_recency(drawings, max_number)
    for decay_key, scores in recency_data.items():
        # Only use one decay rate to avoid making the feature vector too large
        if decay_key == 'decay_0.9':  # Medium-term decay
            features.extend([scores.get(num, 0) for num in range(1, max_number + 1)])

    return features


def monte_carlo_simulation(drawings, drawing_size=5, max_number=50, num_simulations=10000):
    """
    Run Monte Carlo simulations to generate predictions.
    This approach simulates future drawings based on historical probabilities.
    """
    # Calculate baseline probabilities from historical data
    frequency_data = analyze_frequency(drawings, max_number)
    normalized_freq = frequency_data['normalized']

    # Create probability distribution
    probs = [normalized_freq.get(i, 0) for i in range(1, max_number + 1)]

    # Ensure probabilities sum to 1
    total_prob = sum(probs)
    if total_prob > 0:
        probs = [p / total_prob for p in probs]
    else:
        # If no historical data, use uniform distribution
        probs = [1 / max_number] * max_number

    # Run simulations
    simulation_results = []
    numbers = list(range(1, max_number + 1))

    for _ in range(num_simulations):
        # Sample without replacement based on probabilities
        simulation = []
        temp_probs = probs.copy()
        temp_numbers = numbers.copy()

        for _ in range(drawing_size):
            if not temp_numbers:
                break

            # Normalize remaining probabilities
            total = sum(temp_probs)
            if total <= 0:
                break

            norm_probs = [p / total for p in temp_probs]

            # Sample one number
            try:
                selected_idx = np.random.choice(len(temp_numbers), p=norm_probs)
                selected_num = temp_numbers[selected_idx]
                simulation.append(selected_num)

                # Remove the selected number and its probability
                temp_numbers.pop(selected_idx)
                temp_probs.pop(selected_idx)
            except Exception as e:
                logger.error(f"Error in Monte Carlo sampling: {e}")
                break

        if len(simulation) == drawing_size:
            simulation_results.append(sorted(simulation))

    # Count the frequency of each number across all simulations
    number_counts = Counter([num for sim in simulation_results for num in sim])

    # Calculate probabilities
    total_draws = len(simulation_results) * drawing_size
    if total_draws > 0:
        monte_carlo_probs = {num: count / total_draws for num, count in number_counts.items()}
    else:
        monte_carlo_probs = {}

    return monte_carlo_probs


def predict_next_numbers(drawings, drawing_size=5, max_number=50, strategy_weights=None):
    """
    Predict the next set of lottery numbers using multiple strategies with improved weighting.

    Args:
        drawings: List of historical drawings
        drawing_size: Size of each drawing (how many numbers to predict)
        max_number: Maximum possible number in the lottery
        strategy_weights: Dictionary of weights for each strategy (will use defaults if None)

    Returns:
        Predicted numbers for the next drawing
    """
    if not drawings:
        logger.warning("No historical data provided for prediction")
        return []

    # Default strategy weights
    if strategy_weights is None:
        strategy_weights = {
            'frequency': 0.10,
            'recency': 0.20,
            'gap': 0.15,
            'combination': 0.15,
            'monte_carlo': 0.20,
            'ml': 0.20
        }

    # Ensure weights sum to 1
    total_weight = sum(strategy_weights.values())
    if total_weight > 0:
        strategy_weights = {k: v / total_weight for k, v in strategy_weights.items()}

    # Strategy 1: Frequency analysis (basic probability)
    freq_data = analyze_frequency(drawings, max_number)
    normalized_freq = freq_data['normalized']

    # Strategy 2: Recency analysis (time-weighted appearance)
    recency_data = analyze_recency(drawings, max_number)
    # Use medium-term decay for primary recency score
    recency_scores = recency_data['decay_0.9']

    # Strategy 3: Gap analysis (pattern of appearances)
    gap_stats = analyze_gaps(drawings, max_number)

    # Strategy 4: Combination analysis (which numbers appear together)
    combo_data = analyze_combinations(drawings, drawing_size, max_number)

    # Strategy 5: Monte Carlo simulation
    monte_carlo_probs = monte_carlo_simulation(drawings, drawing_size, max_number)

    # Strategy 6: Machine learning prediction
    # Only use ML if we have enough data
    ml_probs = {}
    if len(drawings) >= 50:  # Arbitrary threshold for enough data
        # Train ML model
        X, y = create_enhanced_features(drawings, max_number)
        if len(X) > 0:
            models = train_advanced_ml_model(X, y, max_number)
            if models:
                # Create features for prediction
                latest_features = create_latest_features(drawings, max_number)
                if latest_features:
                    ml_probs = predict_with_advanced_ml(models, latest_features, max_number)

    # Calculate composite score for each number
    scores = {}

    for num in range(1, max_number + 1):
        # 1. Frequency score
        freq_score = normalized_freq.get(num, 0)

        # 2. Recency score
        rec_score = recency_scores.get(num, 0)

        # 3. Gap score - higher for numbers that are "due" based on their pattern
        gap_score = 0
        if num in gap_stats:
            stats = gap_stats[num]
            if stats['mean'] not in (0, float('inf')) and stats['count'] > 0:
                # How much the current gap exceeds the average (capped at a reasonable value)
                gap_ratio = min(3.0, stats['current'] / stats['mean'])
                gap_score = gap_ratio / 3.0  # Normalize to 0-1 range

        # 4. Combination score based on how often this number appears in frequent combinations
        combo_score = 0
        if 2 in combo_data:  # Look at pairs
            pair_counts = combo_data[2]['counts']
            # Sum the counts of all pairs containing this number
            pairs_with_num = sum(count for combo, count in pair_counts.items() if num in combo)
            # Normalize by the maximum pair count
            max_pair_count = max(pair_counts.values()) if pair_counts else 1
            combo_score = min(1.0, pairs_with_num / (max_pair_count * 2))

        # 5. Monte Carlo simulation score
        mc_score = monte_carlo_probs.get(num, 0)

        # 6. ML score
        ml_score = ml_probs.get(num, 0)

        # Combined weighted score
        scores[num] = (
                strategy_weights.get('frequency', 0.1) * freq_score +
                strategy_weights.get('recency', 0.2) * rec_score +
                strategy_weights.get('gap', 0.15) * gap_score +
                strategy_weights.get('combination', 0.15) * combo_score +
                strategy_weights.get('monte_carlo', 0.2) * mc_score +
                strategy_weights.get('ml', 0.2) * ml_score
        )

    # Get top numbers based on score
    top_numbers = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Apply constraints: remove numbers that would create combinations
    # that have never occurred in the history
    if len(drawings) > 10 and 3 in combo_data:
        triplet_data = combo_data[3]['counts']

        # Get preliminary selection
        preliminary = [num for num, _ in top_numbers[:drawing_size]]
        triplets_in_prediction = list(itertools.combinations(sorted(preliminary), 3))

        # Check if any of these triplets have never occurred
        never_seen_triplets = [t for t in triplets_in_prediction if t not in triplet_data]

        if never_seen_triplets and len(top_numbers) > drawing_size:
            # If we have unseen triplets, try to substitute numbers
            for triplet in never_seen_triplets:
                # Find the least confident number in this triplet
                triplet_numbers = list(triplet)
                confidences = [scores[n] for n in triplet_numbers]
                least_confident_idx = confidences.index(min(confidences))
                number_to_replace = triplet_numbers[least_confident_idx]

                # Find the next best number not in our selection
                for num, _ in top_numbers[drawing_size:]:
                    if num not in preliminary:
                        # Check if replacing would create new unseen triplets
                        new_selection = preliminary.copy()
                        new_selection.remove(number_to_replace)
                        new_selection.append(num)

                        # Check all triplets with the new number
                        new_triplets = list(itertools.combinations(sorted(new_selection), 3))
                        new_unseen = [t for t in new_triplets if t not in triplet_data]

                        if len(new_unseen) <= len(never_seen_triplets):
                            # This replacement is better or equal, so use it
                            preliminary.remove(number_to_replace)
                            preliminary.append(num)
                            break

            # Update selection after substitutions
            selection = sorted(preliminary)
        else:
            # No problematic triplets or not enough data to make changes
            selection = [num for num, _ in top_numbers[:drawing_size]]
    else:
        # Not enough data for triplet analysis
        selection = [num for num, _ in top_numbers[:drawing_size]]

    return sorted(selection)


def evaluate_predictions(predictions, actual_results, metrics=None):
    """
    Evaluate the accuracy of predictions with multiple metrics.

    Args:
        predictions: List of predicted numbers
        actual_results: List of actual numbers drawn
        metrics: List of metrics to calculate (default: all)

    Returns:
        Dictionary of evaluation metrics
    """
    if not actual_results or not predictions:
        return {'matched_count': 0, 'accuracy': 0, 'partial_match': 0}

    if metrics is None:
        metrics = ['matched_count', 'accuracy', 'partial_match', 'value_accuracy']

    results = {}

    # Convert to sets for easier comparison
    pred_set = set(predictions)
    actual_set = set(actual_results)

    # Calculate metrics
    matched = pred_set.intersection(actual_set)
    matched_count = len(matched)

    if 'matched_count' in metrics:
        results['matched_count'] = matched_count

    if 'accuracy' in metrics:
        results['accuracy'] = matched_count / min(len(predictions), len(actual_results)) * 100

    if 'partial_match' in metrics:
        # Partial match score: gives credit for numbers that are close
        partial_score = 0
        for p in predictions:
            # Find closest actual number
            if p in actual_set:
                partial_score += 1  # Exact match
            else:
                # Find minimum distance to any actual number
                min_dist = min(abs(p - a) for a in actual_results)
                # Give partial credit based on distance (adjust the formula as needed)
                closeness = max(0, 1 - (min_dist / 10))  # Linear decay up to distance of 10
                partial_score += closeness

        results['partial_match'] = partial_score / len(predictions) * 100

    if 'value_accuracy' in metrics:
        # Measure how close predictions are in terms of absolute difference
        avg_diff = sum(abs(p - a) for p in predictions for a in actual_results) / (
                    len(predictions) * len(actual_results))
        results['value_accuracy'] = 100 - min(100, avg_diff * 5)  # Higher is better

    return results


def backtest_strategy(drawings, drawing_size=5, num_backtests=None, max_number=50, strategy_weights=None):
    """
    Backtest the prediction strategy on historical data with improved statistics and
    comparison against random baseline.

    Args:
        drawings: List of historical drawings
        drawing_size: Size of each drawing
        num_backtests: Number of backtests to run (default: 30% of data)
        max_number: Maximum possible number in the lottery
        strategy_weights: Dictionary of weights for each strategy

    Returns:
        Dictionary of backtest results with detailed statistics
    """
    if len(drawings) < 20:  # Need minimum history for meaningful testing
        logger.warning("Not enough historical data for proper backtesting")
        return {'average_accuracy': 0, 'random_baseline': 0, 'improvement': 0}


def visualize_data(drawings, predictions=None, max_number=50, output_file="lottery_analysis.png"):
    """
    Create enhanced visualizations of the lottery data with predicted numbers highlighted.
    Includes more statistical insights and improved formatting.
    """
    plt.figure(figsize=(16, 14))

    # 1. Frequency chart with normalized distribution overlay
    freq_data = analyze_frequency(drawings, max_number)
    raw_freq = freq_data['raw']
    norm_freq = freq_data['normalized']

    ax1 = plt.subplot(3, 2, 1)
    numbers = list(range(1, max_number + 1))
    frequencies = [raw_freq.get(num, 0) for num in numbers]

    # Calculate expected frequency for a uniform distribution
    total_drawings = len(drawings)
    drawing_size = len(drawings[0]) if drawings else 5
    expected_freq = total_drawings * drawing_size / max_number

    # Raw frequency bars
    bars = ax1.bar(numbers, frequencies, alpha=0.7)

    # Add horizontal line for expected frequency
    ax1.axhline(y=expected_freq, color='r', linestyle='--', alpha=0.7,
                label=f'Expected ({expected_freq:.1f})')

    # Highlight predicted numbers if provided
    if predictions:
        for i, bar in enumerate(bars):
            if i + 1 in predictions:
                bar.set_color('red')
                bar.set_alpha(1.0)

    ax1.set_title("Number Frequency")
    ax1.set_xlabel("Number")
    ax1.set_ylabel("Times Drawn")
    ax1.legend()

    # 2. Frequency deviation from expected
    ax2 = plt.subplot(3, 2, 2)

    # Calculate deviation percentage
    deviations = [(freq / expected_freq - 1) * 100 if expected_freq > 0 else 0
                  for freq in frequencies]

    # Create bars with color based on positive/negative
    colors = ['green' if d >= 0 else 'red' for d in deviations]
    dev_bars = ax2.bar(numbers, deviations, color=colors, alpha=0.7)

    # Highlight predicted numbers
    if predictions:
        for i, bar in enumerate(dev_bars):
            if i + 1 in predictions:
                bar.set_alpha(1.0)
                bar.set_edgecolor('black')
                bar.set_linewidth(1.5)

    ax2.set_title("Deviation from Expected Frequency (%)")
    ax2.set_xlabel("Number")
    ax2.set_ylabel("Deviation %")
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 3. Recent drawing trends
    ax3 = plt.subplot(3, 2, 3)
    last_n = min(15, len(drawings))
    recent_drawings = drawings[-last_n:]

    # Create a heatmap-like visualization
    drawing_matrix = np.zeros((last_n, max_number))

    for i, drawing in enumerate(recent_drawings):
        for num in drawing:
            if 1 <= num <= max_number:
                drawing_matrix[i, num - 1] = 1

    # Plot as a grid
    ax3.imshow(drawing_matrix, cmap='Blues', aspect='auto', interpolation='none')

    # Add predictions at the bottom
    if predictions:
        pred_row = np.zeros(max_number)
        for num in predictions:
            if 1 <= num <= max_number:
                pred_row[num - 1] = 1

        # Create a separate subplot for predictions
        ax3.imshow(np.array([pred_row]), cmap='Reds', extent=(
            -0.5, max_number - 0.5, -0.9, -0.1), aspect='auto', interpolation='none')

    ax3.set_title("Recent Drawing Patterns")
    ax3.set_xlabel("Number")
    ax3.set_yticks(range(last_n))
    ax3.set_yticklabels([f"Draw {len(drawings) - i}" for i in range(last_n)])

    if predictions:
        ax3.set_yticks(list(ax3.get_yticks()) + [-0.5])
        ax3.set_yticklabels(list(ax3.get_yticklabels()) + ["Prediction"])

    # 4. Gap analysis
    ax4 = plt.subplot(3, 2, 4)
    gap_stats = analyze_gaps(drawings, max_number)

    # Create data for current gap vs. average gap
    gap_data = []
    for num in range(1, max_number + 1):
        if num in gap_stats and gap_stats[num]['mean'] != float('inf'):
            if gap_stats[num]['mean'] > 0:
                gap_ratio = gap_stats[num]['current'] / gap_stats[num]['mean']
            else:
                gap_ratio = 0
            gap_data.append((num, gap_ratio))

    if gap_data:
        nums, ratios = zip(*gap_data)
        bars = ax4.bar(nums, ratios, alpha=0.7)

        # Add reference line at 1.0 (where current gap equals average)
        ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5,
                    label='Current = Average')

        # Highlight predicted numbers
        if predictions:
            for i, bar in enumerate(bars):
                if nums[i] in predictions:
                    bar.set_color('red')
                    bar.set_alpha(1.0)

        ax4.set_title("Current Gap / Average Gap Ratio")
        ax4.set_xlabel("Number")
        ax4.set_ylabel("Ratio")
        ax4.set_ylim(bottom=0)
        ax4.legend()

    # 5. Heat map of pair correlations
    ax5 = plt.subplot(3, 2, 5)

    # Calculate how often each pair of numbers appears together
    pair_matrix = np.zeros((max_number, max_number))

    for drawing in drawings:
        # Count co-occurrences
        for i in drawing:
            for j in drawing:
                if 1 <= i <= max_number and 1 <= j <= max_number:
                    pair_matrix[i - 1, j - 1] += 1

    # Remove self-pairs (diagonal)
    np.fill_diagonal(pair_matrix, 0)

    # Plot as heatmap
    im = ax5.imshow(pair_matrix, cmap='viridis', interpolation='none')
    plt.colorbar(im, ax=ax5, label='Co-occurrence count')

    ax5.set_title("Number Co-occurrence Matrix")
    ax5.set_xlabel("Number")
    ax5.set_ylabel("Number")

    # Add grid lines
    ax5.set_xticks(np.arange(-.5, max_number, 5), minor=True)
    ax5.set_yticks(np.arange(-.5, max_number, 5), minor=True)
    ax5.grid(which='minor', color='w', linestyle='-', linewidth=0.3)

    # 6. Prediction score breakdown
    if predictions:
        ax6 = plt.subplot(3, 2, 6)

        # Generate detailed scoring for the predicted numbers
        freq = analyze_frequency(drawings, max_number)['normalized']
        recency = analyze_recency(drawings, max_number)['decay_0.9']
        gap_data = analyze_gaps(drawings, max_number)

        # Calculate scores for each factor
        pred_data = []

        for num in predictions:
            # Frequency score
            freq_score = freq.get(num, 0)

            # Recency score
            rec_score = recency.get(num, 0)

            # Gap score
            if num in gap_data and gap_data[num]['mean'] not in (0, float('inf')):
                gap_ratio = min(3.0, gap_data[num]['current'] / gap_data[num]['mean'])
                gap_score = gap_ratio / 3.0
            else:
                gap_score = 0

            pred_data.append((num, freq_score, rec_score, gap_score))

        # Create stacked bar chart
        labels = [str(num) for num, _, _, _ in pred_data]
        freq_scores = [fs for _, fs, _, _ in pred_data]
        rec_scores = [rs for _, _, rs, _ in pred_data]
        gap_scores = [gs for _, _, _, gs in pred_data]

        width = 0.8
        ax6.bar(labels, freq_scores, width, label='Frequency', alpha=0.7)
        ax6.bar(labels, rec_scores, width, bottom=freq_scores, label='Recency', alpha=0.7)

        # Calculate running sum for stacking
        stack_height = [f + r for f, r in zip(freq_scores, rec_scores)]
        ax6.bar(labels, gap_scores, width, bottom=stack_height, label='Gap', alpha=0.7)

        ax6.set_title("Prediction Factor Breakdown")
        ax6.set_xlabel("Predicted Number")
        ax6.set_ylabel("Factor Contribution")
        ax6.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    logger.info(f"Visualization saved to {output_file}")

    # Close the figure to free memory
    plt.close()


def optimize_strategy_weights(drawings, drawing_size=5, max_number=50, iterations=20):
    """
    Use grid search to find optimal strategy weights based on backtest performance.
    """
    if len(drawings) < 30:
        logger.warning("Not enough data for strategy optimization")
        return None

    # Define the parameter grid
    param_ranges = {
        'frequency': [0.05, 0.1, 0.15, 0.2],
        'recency': [0.1, 0.15, 0.2, 0.25],
        'gap': [0.1, 0.15, 0.2],
        'combination': [0.1, 0.15, 0.2],
        'monte_carlo': [0.1, 0.15, 0.2, 0.25],
        'ml': [0.1, 0.15, 0.2, 0.25]
    }

    # Calculate how many combinations to try (limit for performance)
    total_combinations = 1
    for values in param_ranges.values():
        total_combinations *= len(values)

    logger.info(f"Strategy optimization: testing {min(iterations, total_combinations)} combinations")

    # Generate random combinations for testing
    best_weights = None
    best_performance = -float('inf')

    for _ in range(min(iterations, total_combinations)):
        # Generate a random weight combination
        weights = {
            param: random.choice(values)
            for param, values in param_ranges.items()
        }

        # Normalize to ensure they sum to 1.0
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Run backtest with these weights
        backtest_results = backtest_strategy(
            drawings, drawing_size, num_backtests=20,
            max_number=max_number, strategy_weights=weights
        )

        # Evaluate performance (use improvement over random as the metric)
        if 'improvement' in backtest_results and 'accuracy' in backtest_results['improvement']:
            performance = backtest_results['improvement']['accuracy']

            if performance > best_performance:
                best_performance = performance
                best_weights = weights
                logger.info(f"New best weights found: {weights} with {performance:.2f}% improvement")

    return best_weights


def generate_report(drawings, predictions, backtest_results, max_number=50, output_file="lottery_report.txt"):
    """
    Generate a comprehensive report on the lottery analysis and predictions.
    """
    if not drawings:
        return

    try:
        with open(output_file, 'w') as f:
            f.write("=== LOTTERY PREDICTION REPORT ===\n\n")

            # Basic dataset information
            f.write(f"Dataset includes {len(drawings)} previous drawings\n")
            drawing_size = len(drawings[0]) if drawings else 0
            f.write(f"Drawing size: {drawing_size} numbers\n")
            f.write(f"Number range: 1 to {max_number}\n\n")

            # Prediction
            f.write("PREDICTION FOR NEXT DRAWING:\n")
            f.write(f"{predictions}\n\n")

            # Backtest results
            f.write("BACKTEST RESULTS:\n")
            if backtest_results:
                if 'strategy_metrics' in backtest_results and 'accuracy' in backtest_results['strategy_metrics']:
                    f.write(f"Average prediction accuracy: {backtest_results['strategy_metrics']['accuracy']:.2f}%\n")

                if 'random_metrics' in backtest_results and 'accuracy' in backtest_results['random_metrics']:
                    f.write(f"Random baseline accuracy: {backtest_results['random_metrics']['accuracy']:.2f}%\n")

                if 'improvement' in backtest_results and 'accuracy' in backtest_results['improvement']:
                    f.write(f"Improvement over random: {backtest_results['improvement']['accuracy']:.2f}%\n")

                if 'match_rate' in backtest_results:
                    f.write(f"Tests with at least one match: {backtest_results['match_rate']:.2f}%\n")

                if 'beat_random_rate' in backtest_results:
                    f.write(f"Tests where strategy beat random: {backtest_results['beat_random_rate']:.2f}%\n")
            else:
                f.write("No backtest results available\n")

            f.write("\n")

            # Hot and cold numbers
            f.write("HOT NUMBERS (most frequent):\n")
            freq_data = analyze_frequency(drawings, max_number)['raw']
            hot_numbers = sorted(freq_data.items(), key=lambda x: x[1], reverse=True)[:10]
            for num, freq in hot_numbers:
                f.write(f"Number {num}: drawn {freq} times\n")

            f.write("\nCOLD NUMBERS (least frequent):\n")
            cold_numbers = sorted(freq_data.items(), key=lambda x: x[1])[:10]
            for num, freq in cold_numbers:
                f.write(f"Number {num}: drawn {freq} times\n")

            f.write("\n")

            # Numbers due to appear
            f.write("NUMBERS DUE TO APPEAR (longest current gaps relative to average):\n")
            gap_stats = analyze_gaps(drawings, max_number)
            due_numbers = []

            for num in range(1, max_number + 1):
                if num in gap_stats and gap_stats[num]['mean'] not in (0, float('inf')):
                    if gap_stats[num]['mean'] > 0:
                        due_score = gap_stats[num]['current'] / gap_stats[num]['mean']
                        due_numbers.append((num, due_score, gap_stats[num]['current']))

            due_numbers.sort(key=lambda x: x[1], reverse=True)
            for num, ratio, current_gap in due_numbers[:10]:
                f.write(f"Number {num}: current gap {current_gap} drawings " +
                        f"({ratio:.2f}x average)\n")

            f.write("\n")

            # Most common pairs
            f.write("MOST COMMON PAIRS:\n")
            combo_data = analyze_combinations(drawings, drawing_size, max_number)
            if 2 in combo_data and combo_data[2]['counts']:
                pairs = list(combo_data[2]['counts'].items())
                pairs.sort(key=lambda x: x[1], reverse=True)

                for (a, b), count in pairs[:10]:
                    f.write(f"Pair ({a},{b}): appeared {count} times\n")
            else:
                f.write("No pair data available\n")

            f.write("\n")

            f.write("NOTES:\n")
            f.write("• Past performance is not indicative of future results\n")
            f.write("• Lottery drawings are random events with fixed probabilities\n")
            f.write("• No prediction system can guarantee lottery wins\n")
            f.write("• Please use this information responsibly\n")

        logger.info(f"Report saved to {output_file}")

    except Exception as e:
        logger.error(f"Error generating report: {e}")


def main():
    """
    Main function with improved flow and error handling.
    """
    try:
        # File path to data.txt
        file_path = "data.txt"
        output_dir = "lottery_results"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Configure logging to file
        file_handler = logging.FileHandler(f"{output_dir}/lottery_analysis.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Load data
        logger.info(f"Loading data from {file_path}...")
        drawings = load_lottery_data(file_path)

        if not drawings:
            logger.error("No valid data found or could not read the file.")
            return

        logger.info(f"Loaded {len(drawings)} drawings.")

        # Determine the typical drawing size and max number
        drawing_sizes = [len(d) for d in drawings]
        drawing_size = max(set(drawing_sizes), key=drawing_sizes.count)
        max_number = max([max(d) for d in drawings]) if drawings else 50

        logger.info(f"Typical drawing size: {drawing_size}")
        logger.info(f"Maximum number: {max_number}")

        # Optimize strategy weights if enough data
        if len(drawings) >= 50:
            logger.info("Optimizing strategy weights...")
            optimal_weights = optimize_strategy_weights(
                drawings, drawing_size, max_number
            )
        else:
            logger.info("Not enough data for strategy optimization, using defaults")
            optimal_weights = None

        # Backtest to check accuracy
        logger.info("Backtesting prediction strategy...")
        backtest_results = backtest_strategy(
            drawings, drawing_size, max_number=max_number,
            strategy_weights=optimal_weights
        )

        # Generate predictions for future drawings
        logger.info("Generating predictions for future drawings...")
        prediction = predict_next_numbers(
            drawings, drawing_size, max_number,
            strategy_weights=optimal_weights
        )
        logger.info(f"Prediction for next drawing: {prediction}")

        # Generate multiple future predictions
        num_future = 5
        future_predictions = []
        temp_drawings = drawings.copy()

        for i in range(num_future):
            next_prediction = predict_next_numbers(
                temp_drawings, drawing_size, max_number,
                strategy_weights=optimal_weights
            )
            future_predictions.append(next_prediction)
            # Add the prediction to simulate a new drawing
            temp_drawings.append(next_prediction)

        logger.info(f"Generated {num_future} future predictions")
        for i, pred in enumerate(future_predictions):
            logger.info(f"Drawing {i + 1}: {pred}")

        # Create visualizations
        logger.info("Creating data visualizations...")
        vis_file = f"{output_dir}/lottery_analysis.png"
        visualize_data(drawings, prediction, max_number, vis_file)

        # Generate report
        logger.info("Generating comprehensive report...")
        report_file = f"{output_dir}/lottery_report.txt"
        generate_report(drawings, prediction, backtest_results, max_number, report_file)

        logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"Error in main process: {e}")


def disclaimer():
    """
    Print important disclaimer about lottery predictions.
    """
    print("\n" + "=" * 80)
    print("IMPORTANT DISCLAIMER".center(80))
    print("=" * 80)
    print("\nThis lottery prediction software is provided for EDUCATIONAL PURPOSES ONLY.")
    print("\nKey points to understand:")
    print("• Lottery drawings are random events with fixed probabilities")
    print("• No algorithm can predict random events with certainty")
    print("• Each number has the same mathematical probability in a fair lottery")
    print("• Past results do not influence future drawings")
    print("• Statistical analysis can identify patterns but cannot guarantee wins")
    print("• The expected value of lottery tickets is typically negative")
    print("\nThe authors of this software make NO CLAIMS about improving your")
    print("chances of winning and accept NO RESPONSIBILITY for any financial")
    print("decisions made based on its output.")
    print("\nPlease use responsibly and consider gambling addiction resources if needed.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    disclaimer()
    main()

    # If num_backtests not specified, use 30% of data or 50, whichever is smaller
    if num_backtests is None:
        num_backtests = min(50, int(len(drawings) * 0.3))

    # Ensure we have enough data for training even after leaving some for backtesting
    if len(drawings) - num_backtests < 10:
        num_backtests = len(drawings) - 10

    if num_backtests <= 0:
        logger.warning("Not enough data for backtesting after reserving training data")
        return {'average_accuracy': 0, 'random_baseline': 0, 'improvement': 0}

    logger.info(f"Running {num_backtests} backtests")

    # Track results
    strategy_results = []
    random_results = []

    # Generate random predictions for baseline comparison
    random_predictions = []
    for _ in range(num_backtests):
        # Generate random selection without replacement
        random_pred = sorted(random.sample(range(1, max_number + 1), drawing_size))
        random_predictions.append(random_pred)

    # Run backtests
    for i in tqdm(range(num_backtests)):
        # Select a cutoff point
        cutoff = len(drawings) - i - 1
        if cutoff < 10:  # Need minimum history
            break

        # Use data up to the cutoff
        training_data = drawings[:cutoff]

        # Make prediction with our strategy
        prediction = predict_next_numbers(
            training_data, drawing_size, max_number, strategy_weights
        )

        # Compare with actual
        actual = drawings[cutoff]

        # Evaluate our strategy
        accuracy = evaluate_predictions(prediction, actual)
        strategy_results.append(accuracy)

        # Evaluate random baseline
        random_accuracy = evaluate_predictions(random_predictions[i], actual)
        random_results.append(random_accuracy)

        logger.info(f"Backtest {i + 1}: predicted {prediction}, actual {actual}, " +
                    f"matched {accuracy['matched_count']} numbers " +
                    f"({accuracy['accuracy']:.2f}% accuracy)")

    # Calculate summary statistics
    if strategy_results:
        # Calculate average metrics for our strategy
        avg_metrics = {}
        for metric in strategy_results[0].keys():
            avg_metrics[metric] = sum(r[metric] for r in strategy_results) / len(strategy_results)

        # Calculate average metrics for random baseline
        random_avg_metrics = {}
        for metric in random_results[0].keys():
            random_avg_metrics[metric] = sum(r[metric] for r in random_results) / len(random_results)

        # Calculate improvement over random baseline
        improvement = {}
        for metric in avg_metrics.keys():
            if metric in random_avg_metrics and random_avg_metrics[metric] > 0:
                improvement[metric] = (avg_metrics[metric] / random_avg_metrics[metric] - 1) * 100
            else:
                improvement[metric] = 0

        # Count how many backtests had at least one match
        matches_count = sum(1 for r in strategy_results if r['matched_count'] > 0)
        match_rate = matches_count / len(strategy_results) * 100

        # Count how many backtests beat random
        beat_random_count = sum(1 for i in range(len(strategy_results))
                                if strategy_results[i]['matched_count'] >
                                random_results[i]['matched_count'])
        beat_random_rate = beat_random_count / len(strategy_results) * 100

        logger.info(f"Average accuracy across {len(strategy_results)} backtests: " +
                    f"{avg_metrics['accuracy']:.2f}%")
        logger.info(f"Random baseline accuracy: {random_avg_metrics['accuracy']:.2f}%")
        logger.info(f"Improvement over random: {improvement['accuracy']:.2f}%")
        logger.info(f"Match rate (% of tests with at least one match): {match_rate:.2f}%")
        logger.info(f"Beat random rate: {beat_random_rate:.2f}%")

        return {
            'strategy_metrics': avg_metrics,
            'random_metrics': random_avg_metrics,
            'improvement': improvement,
            'match_rate': match_rate,
            'beat_random_rate': beat_random_rate,
            'num_backtests': len(strategy_results)
        }
    else:
        logger.warning("No backtest results were collected")
        return {'average_accuracy': 0, 'random_baseline': 0, 'improvement': 0}