import pandas as pd
import numpy as np
import random
from collections import Counter


def load_data(file_path):
    """
    Load space-separated lottery data
    """
    try:
        # Read the file line by line
        with open(file_path, 'r') as file:
            draws = []
            for line in file:
                # Split by whitespace and convert to integers
                numbers = [int(num) for num in line.strip().split() if num.isdigit()]
                if numbers:  # Only add non-empty lines
                    draws.append(numbers)

        print(f"Successfully loaded {len(draws)} draws from {file_path}")

        # Check if we have valid data (EuroMillions has 5 main + 2 star numbers)
        # Your data seems to have just 5 numbers per line
        expected_count = 5
        valid_draws = [draw for draw in draws if len(draw) == expected_count]

        if len(valid_draws) < len(draws):
            print(
                f"Warning: Found {len(draws) - len(valid_draws)} invalid draws that don't have {expected_count} numbers")

        print(f"Working with {len(valid_draws)} valid draws")
        return valid_draws

    except Exception as e:
        print(f"Error loading data: {e}")
        return []


def analyze_frequency(draws):
    """
    Analyze the frequency of each number across all draws
    """
    # Flatten the list of draws to get all numbers
    all_numbers = [num for draw in draws for num in draw]

    # Count occurrences of each number
    number_counts = Counter(all_numbers)

    # Get the range of possible numbers
    min_number = min(all_numbers)
    max_number = max(all_numbers)

    # Fill in missing numbers with zero counts
    for num in range(min_number, max_number + 1):
        if num not in number_counts:
            number_counts[num] = 0

    # Sort by frequency
    sorted_counts = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)

    print("\nMost frequent numbers:")
    for num, count in sorted_counts[:10]:
        print(f"Number {num}: appeared {count} times")

    return number_counts


def analyze_patterns(draws):
    """
    Look for patterns in the data
    """
    results = {}

    # Check for consecutive draws with repeating numbers
    overlaps = []
    for i in range(1, len(draws)):
        previous = set(draws[i - 1])
        current = set(draws[i])
        overlap = previous.intersection(current)
        overlaps.append(len(overlap))

    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    results['average_overlap'] = avg_overlap
    print(f"\nAverage number overlap between consecutive draws: {avg_overlap:.2f}")

    # Check if certain positions tend to have certain number ranges
    positions = {i: [] for i in range(5)}  # Assuming 5 numbers per draw

    for draw in draws:
        for i, num in enumerate(draw):
            if i < 5:  # Ensure we only look at the main numbers
                positions[i].append(num)

    position_averages = {pos: sum(nums) / len(nums) for pos, nums in positions.items()}
    results['position_averages'] = position_averages

    print("\nAverage value by position:")
    for pos, avg in position_averages.items():
        print(f"Position {pos + 1}: {avg:.2f}")

    # Check for hot and cold numbers
    recent_draws = draws[-10:]  # Last 10 draws
    recent_numbers = [num for draw in recent_draws for num in draw]
    recent_counts = Counter(recent_numbers)

    # Hot numbers (appeared multiple times recently)
    hot_numbers = [num for num, count in recent_counts.items() if count > 1]
    results['hot_numbers'] = hot_numbers

    # Cold numbers (haven't appeared recently)
    # Assuming numbers range from 1 to 50
    all_possible = set(range(1, 51))
    recent_set = set(recent_numbers)
    cold_numbers = list(all_possible - recent_set)
    results['cold_numbers'] = cold_numbers

    print(f"\nHot numbers (appeared multiple times recently): {hot_numbers}")
    print(f"Cold numbers (haven't appeared in last 10 draws): {cold_numbers[:10]}...")

    return results


def generate_predictions(draws, frequency_data, pattern_data):
    """
    Generate 5 sets of predictions based on different strategies
    """
    predictions = []

    # Strategy 1: Frequency-based
    sorted_freq = sorted(frequency_data.items(), key=lambda x: x[1], reverse=True)
    top_numbers = [num for num, _ in sorted_freq[:15]]
    prediction1 = sorted(random.sample(top_numbers, 5))
    predictions.append((prediction1, "Based on most frequent numbers historically"))

    # Strategy 2: Position-based
    position_based = []
    for pos in range(5):
        pos_numbers = [draw[pos] for draw in draws if len(draw) > pos]
        pos_counts = Counter(pos_numbers)
        most_common = pos_counts.most_common(3)
        position_based.append(most_common[0][0] if most_common else random.randint(1, 50))
    predictions.append((sorted(position_based), "Based on most common numbers by position"))

    # Strategy 3: Hot + Cold combination
    hot_numbers = pattern_data.get('hot_numbers', [])
    cold_numbers = pattern_data.get('cold_numbers', [])

    # If we have enough hot numbers, use 2 of them, otherwise fill with top frequent
    hot_selection = random.sample(hot_numbers, min(2, len(hot_numbers))) if hot_numbers else []

    # Fill the rest with cold numbers
    remaining = 5 - len(hot_selection)
    cold_selection = random.sample(cold_numbers, min(remaining, len(cold_numbers))) if cold_numbers else []

    # If we still need more, use top frequent
    if len(hot_selection) + len(cold_selection) < 5:
        need = 5 - len(hot_selection) - len(cold_selection)
        freq_selection = [num for num, _ in sorted_freq[:20]
                          if num not in hot_selection and num not in cold_selection]
        freq_selection = random.sample(freq_selection, min(need, len(freq_selection)))
    else:
        freq_selection = []

    combined = hot_selection + cold_selection + freq_selection

    # Ensure we have exactly 5 numbers
    while len(combined) < 5:
        new_num = random.randint(1, 50)
        if new_num not in combined:
            combined.append(new_num)

    predictions.append((sorted(combined[:5]), "Combination of hot and cold numbers"))

    # Strategy 4: Last draw + history pattern
    if draws:
        last_draw = draws[-1]
        avg_overlap = pattern_data.get('average_overlap', 0)

        # Round to nearest integer
        expected_overlap = round(avg_overlap)

        # Select n numbers from the last draw
        last_draw_selection = random.sample(last_draw, min(expected_overlap, len(last_draw)))

        # Fill the rest with frequent numbers not in the last draw
        rest_needed = 5 - len(last_draw_selection)
        rest_pool = [num for num, _ in sorted_freq if num not in last_draw_selection]
        rest_selection = random.sample(rest_pool, min(rest_needed, len(rest_pool)))

        pattern_based = last_draw_selection + rest_selection

        # Ensure we have exactly 5 numbers
        while len(pattern_based) < 5:
            new_num = random.randint(1, 50)
            if new_num not in pattern_based:
                pattern_based.append(new_num)

        predictions.append((sorted(pattern_based[:5]), "Based on pattern from last draw"))
    else:
        # Fallback if no last draw
        predictions.append((sorted(random.sample(range(1, 51), 5)), "Random selection (no last draw data)"))

    # Strategy 5: Pure mathematical approach (choosing numbers with specific spacing)
    # This is still random but follows a pattern some players use
    start_num = random.randint(1, 20)  # Start with a low number
    spacing = random.randint(5, 10)  # Choose some spacing

    math_based = [start_num]
    current = start_num

    # Generate 4 more numbers with spacing
    for _ in range(4):
        current += spacing
        if current > 50:  # Wrap around if we exceed the maximum
            current = current % 50
            if current == 0:
                current = 50
        math_based.append(current)

    predictions.append((sorted(math_based), "Mathematical approach with number spacing"))

    return predictions


def main():
    file_path = "data.txt"

    # Load the data
    draws = load_data(file_path)

    if not draws:
        print("No valid data found. Cannot continue.")
        return

    # Analyze frequency
    print("\nAnalyzing number frequencies...")
    frequency_data = analyze_frequency(draws)

    # Analyze patterns
    print("\nAnalyzing patterns in the data...")
    pattern_data = analyze_patterns(draws)

    # Generate predictions
    print("\nGenerating suggestions for next 5 draws...")
    predictions = generate_predictions(draws, frequency_data, pattern_data)

    # Display predictions
    print("\n--- Next 5 Draw Suggestions ---")
    for i, (numbers, method) in enumerate(predictions):
        print(f"\nDraw {i + 1}: {numbers}")
        print(f"Method: {method}")

    print("\nIMPORTANT: These are statistical suggestions based on your data analysis,")
    print("but lottery draws are designed to be random and unpredictable.")
    print("No method can predict future lottery numbers with 100% accuracy.")


if __name__ == "__main__":
    main()