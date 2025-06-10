import random
from collections import Counter
import numpy as np


def read_data(file_path):
    """Reads lottery data from a file."""
    all_numbers = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                numbers = [int(num) for num in line.strip().split() if num.isdigit() and 1 <= int(num) <= 50]
                if len(numbers) == 5:  # Ensure each line has 5 numbers
                    all_numbers.append(numbers)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except ValueError:
        print(f"Error: Non-numeric values found in data file")
        return []
    return all_numbers


def calculate_co_occurrence(data, time_window=20):
    """Calculates co-occurrence frequency of number pairs."""
    co_occurrence = Counter()
    for draw in data[-time_window:]:  # Iterate through draws, in the time window.
        for i in range(len(draw)):
            for j in range(i + 1, len(draw)):
                pair = tuple(sorted((draw[i], draw[j])))
                co_occurrence[pair] += 1
    return co_occurrence


def suggest_numbers(data, params, num_suggestions=5):
    """Suggests lottery numbers using a hybrid approach with adjustable parameters."""
    if not data:
        return random.sample(range(1, 51), num_suggestions)

    # Extract parameters
    co_occurrence_time_window = params["co_occurrence_time_window"]
    recency_decay_rate = params["recency_decay_rate"]
    freq_range_boost = params["freq_range_boost"]
    pattern_boost_factor = params["pattern_boost_factor"]
    recency_weight = params["recency_weight"]
    frequency_weight = params["frequency_weight"]
    hot_cold_weight = params["hot_cold_weight"]
    hot_cold_length = params["hot_cold_length"]

    # Time Windows for Frequency
    all_numbers = [number for draw in data for number in draw]
    all_numbers_recent = [number for draw in data[-co_occurrence_time_window:] for number in draw]
    frequency = Counter(all_numbers)
    frequency_recent = Counter(all_numbers_recent)

    co_occurrence = calculate_co_occurrence(data, time_window=co_occurrence_time_window)

    # Recency Analysis with Exponential Decay
    number_last_seen = {}
    draw_count = 0
    for draw in data:
        draw_count += 1
        for num in draw:
            number_last_seen[num] = draw_count

    max_draw = draw_count
    recency_scores = {
        num: np.exp(-(max_draw - last_seen) / (max_draw / recency_decay_rate)) if last_seen != max_draw else 10
        for num, last_seen in number_last_seen.items()
    }

    # Frequency with Cutoff and Ranges
    freq_scores = {}
    max_freq = max(frequency.values()) if frequency else 1
    max_freq_recent = max(frequency_recent.values()) if frequency_recent else 1
    for num in range(1, 51):
        freq_scores[num] = 0.0
        if num in frequency:
            freq_scores[num] = frequency[num] / max_freq
        if num in frequency_recent:
            freq_scores[num] += frequency_recent[num] / max_freq_recent

        # Add frequency range boost
        if num in frequency:
            if frequency[num] > max_freq / 3:
                freq_scores[num] *= freq_range_boost
            elif frequency[num] < max_freq / 10:
                freq_scores[num] *= 0.5

    # Hot/Cold Analysis (Improved)
    hot_numbers = set([num for num, _ in frequency.most_common(hot_cold_length)])
    cold_numbers = set([num for num, _ in frequency.most_common()[:(-hot_cold_length - 1):-1]])
    hot_cold_scores = {}
    for num in range(1, 51):
        if num in hot_numbers:
            hot_cold_scores[num] = 1.0  # Hot number
        elif num in cold_numbers:
            hot_cold_scores[num] = -1.0  # Cold number
        else:
            hot_cold_scores[num] = 0.0  # Neutral

    # Combined score
    combined_scores = {}
    for num in range(1, 51):
        recency_component = recency_scores.get(num, 0) * recency_weight
        frequency_component = freq_scores.get(num, 0) * frequency_weight
        hot_cold_component = hot_cold_scores.get(num, 0) * hot_cold_weight

        combined_scores[num] = recency_component + frequency_component + hot_cold_component

    # Pattern Boost
    for num1 in combined_scores:
        partner_boost = 0
        for num2 in range(1, 51):
            if num1 != num2:
                pair = tuple(sorted((num1, num2)))
                partner_boost += co_occurrence[pair]
        combined_scores[num1] += partner_boost * pattern_boost_factor

    # Select numbers with some bias
    # split the numbers into 5 groups, and pick from each
    sorted_by_combined = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    groups = np.array_split(sorted_by_combined, 5)

    predictions = []
    for group in groups:
        if len(group) > 0:
            predictions.extend(random.sample([num for num, _ in group], min(1, len(group))))

    # if we dont have the right number of predictions then pick at random.
    while len(predictions) < num_suggestions:
        predictions.append(random.randint(1, 50))

    return predictions


def backtest_predictions(data, params):
    """Backtests the prediction accuracy against all draws in the data."""
    if not data:
        print("No data provided for backtesting.")
        return 0.0  # Return 0 if no data

    if len(data) < 2:
        print(f"Not enough data to run backtest. Need at least 2 draws.")
        return 0.0

    correct_predictions = []
    total_predictions = 0
    total_correct = 0

    for i in range(len(data)):  # Start from the first draw

        if i == len(data) - 1:
            # this is the last line, just test against this.
            train_data = data[:i]
            test_data = data[i]
        else:
            train_data = data[:i + 1]  # Data up to and including the current draw
            test_data = data[i + 1]  # The next draw

        # Generate predictions based on previous draws
        predictions = suggest_numbers(train_data, params)

        # Check how many predicted numbers were in the test draw
        correct_count = sum(1 for num in predictions if num in test_data)
        correct_predictions.append(correct_count)
        total_predictions += len(predictions)
        total_correct += correct_count

    # calculate the accuracy
    accuracy = (total_correct / total_predictions) * 100
    return accuracy


def optimize_parameters(data, initial_params, target_accuracy=90.0, max_iterations=100):
    """Optimizes parameters iteratively to reach the target accuracy."""
    best_accuracy = 0.0
    best_params = initial_params.copy()
    current_params = initial_params.copy()

    for iteration in range(max_iterations):
        accuracy = backtest_predictions(data, current_params)
        print(f"Iteration {iteration + 1}: Accuracy = {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = current_params.copy()
        if accuracy >= target_accuracy:
            print(f"\nReached target accuracy of {target_accuracy:.2f}% in {iteration + 1} iterations.")
            return current_params, accuracy

        # Adjust parameters based on accuracy - Improved Logic
        if accuracy < target_accuracy / 3:
            # Low Accuracy: Increase exploration, emphasize frequency
            current_params["co_occurrence_time_window"] += 3
            current_params["recency_decay_rate"] -= 0.1
            current_params["freq_range_boost"] += 0.15
            current_params["pattern_boost_factor"] += 0.001
            current_params["recency_weight"] -= 0.05
            current_params["frequency_weight"] += 0.05
            current_params["hot_cold_weight"] -= 0.02
            current_params["hot_cold_length"] += 1
        elif accuracy < target_accuracy * 2 / 3:
            # Medium Accuracy: Balance exploration/exploitation, fine-tune
            current_params["co_occurrence_time_window"] += 2
            current_params["recency_decay_rate"] -= 0.05
            current_params["freq_range_boost"] += 0.075
            current_params["pattern_boost_factor"] += 0.0005
            current_params["recency_weight"] -= 0.025
            current_params["frequency_weight"] += 0.025
            current_params["hot_cold_weight"] += 0.01
            current_params["hot_cold_length"] += 0.5

        elif accuracy < target_accuracy:
            # Approaching Target: Refine, emphasize recency
            current_params["co_occurrence_time_window"] -= 1
            current_params["recency_decay_rate"] += 0.025
            current_params["freq_range_boost"] -= 0.025
            current_params["pattern_boost_factor"] -= 0.0002
            current_params["recency_weight"] += 0.01
            current_params["frequency_weight"] -= 0.01
            current_params["hot_cold_weight"] += 0.005
            current_params["hot_cold_length"] -= 0.25

        else:
            # High accuracy, refine further.
            current_params["co_occurrence_time_window"] -= 2
            current_params["recency_decay_rate"] += 0.05
            current_params["freq_range_boost"] -= 0.05
            current_params["pattern_boost_factor"] -= 0.0004
            current_params["recency_weight"] += 0.025
            current_params["frequency_weight"] -= 0.025
            current_params["hot_cold_weight"] -= 0.01
            current_params["hot_cold_length"] -= 0.5

        # Ensure parameters stay within bounds
        current_params["co_occurrence_time_window"] = max(1, int(current_params["co_occurrence_time_window"]))
        current_params["recency_decay_rate"] = max(1, current_params["recency_decay_rate"])
        current_params["freq_range_boost"] = max(0, current_params["freq_range_boost"])
        current_params["pattern_boost_factor"] = max(0.0001, current_params["pattern_boost_factor"])
        current_params["recency_weight"] = max(0.1, min(0.9, current_params["recency_weight"]))
        current_params["frequency_weight"] = max(0.1, min(0.9, current_params["frequency_weight"]))
        current_params["hot_cold_weight"] = min(0.1, max(-0.1, current_params["hot_cold_weight"]))
        current_params["hot_cold_length"] = max(2, int(current_params["hot_cold_length"]))

        # print out the results of the interation.
        print(f"New parameters : {current_params}")
        print(f"Best parameters : {best_params}")
        print(f"Best accuracy : {best_accuracy}")

    print(f"\nOptimization finished. Best accuracy achieved: {best_accuracy:.2f}%")
    print(f"Best parameters: {best_params}")
    return best_params, best_accuracy


def main():
    file_path = 'data.txt'
    historical_data = read_data(file_path)

    if not historical_data:
        print("No data loaded. Exiting.")
        return

    initial_params = {
        "co_occurrence_time_window": 20,
        "recency_decay_rate": 5,
        "freq_range_boost": 1.5,
        "pattern_boost_factor": 0.01,
        "recency_weight": 0.5,
        "frequency_weight": 0.3,
        "hot_cold_weight": 0.0,
        "hot_cold_length": 10
    }

    best_params, best_accuracy = optimize_parameters(historical_data, initial_params, max_iterations=200)

    suggested_numbers = suggest_numbers(historical_data, best_params)
    print(f"Suggested numbers for the next lottery draw using optimized parameters: {suggested_numbers}")

    backtest_accuracy = backtest_predictions(historical_data, best_params)
    print(f"\nBacktesting over {len(historical_data)} draws with best parameters:")
    print(f"Prediction accuracy: {backtest_accuracy:.2f}%")


if __name__ == "__main__":
    main()