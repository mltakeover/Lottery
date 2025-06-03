import random
import numpy as np
from collections import Counter
from scipy.stats import norm


def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    numbers = []
    for line in lines:
        numbers.extend([int(num) for num in line.strip().split('\t')])

    return numbers


def calculate_statistics(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return mean, std_dev


def predict_next_numbers(data, num_predictions=5):
    # Calculate frequency and probability
    frequency = Counter(data)
    total = len(data)
    probabilities = {num: count / total for num, count in frequency.items()}

    # Calculate statistics
    mean, std_dev = calculate_statistics(data)

    # Create a scoring system
    scores = {}
    for num in range(1, 51):
        # Combine frequency-based and statistical approaches
        freq_score = probabilities.get(num, 0)
        stat_score = norm.pdf(num, mean, std_dev)
        scores[num] = 0.7 * freq_score + 0.3 * stat_score

    # Sort numbers by their score (descending order)
    sorted_numbers = sorted(scores, key=scores.get, reverse=True)

    # Select the top 'num_predictions' numbers
    predictions = sorted_numbers[:num_predictions]

    # If we don't have enough unique numbers, fill the rest using a weighted random approach
    while len(predictions) < num_predictions:
        remaining_numbers = [n for n in range(1, 51) if n not in predictions]
        remaining_scores = [scores[n] for n in remaining_numbers]
        new_num = random.choices(remaining_numbers, weights=remaining_scores)[0]
        predictions.append(new_num)

    return predictions


def analyze_data(data):
    mean, std_dev = calculate_statistics(data)
    most_common = Counter(data).most_common(5)
    least_common = Counter(data).most_common()[:-6:-1]

    print(f"Mean: {mean:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Most common numbers: {most_common}")
    print(f"Least common numbers: {least_common}")


# Main execution
filename = 'data.txt'
data = read_data(filename)

print("Data analysis:")
analyze_data(data)

predictions = predict_next_numbers(data)
print("\nPredicted next 5 numbers:", predictions)
