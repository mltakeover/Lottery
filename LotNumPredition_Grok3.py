import random
from collections import Counter


def predict_next_numbers(filename):
    number_data = {i: {'freq': 0, 'last_seen': 0} for i in range(1, 51)}
    draw_count = 0

    # Read and count numbers from file
    try:
        with open(filename, 'r') as file:
            for line in file:
                draw_count += 1
                numbers = [int(num) for num in line.strip().split('\t') if num.isdigit() and 1 <= int(num) <= 50]

                for num in numbers:
                    number_data[num]['freq'] += 1
                    number_data[num]['last_seen'] = draw_count
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return []

    # Classify numbers as hot or cold
    hot_numbers = sorted(number_data.items(),
                         key=lambda x: x[1]['freq'], reverse=True)[:10]  # Top 10 most frequent numbers
    cold_numbers = sorted(number_data.items(),
                          key=lambda x: x[1]['freq'])[:10]  # Bottom 10 least frequent numbers

    # Combine hot, cold, and some randomness
    predictions = []

    # 2 from hot numbers
    predictions.extend(random.sample([num for num, _ in hot_numbers], 2))
    # 2 from cold numbers
    predictions.extend(random.sample([num for num, _ in cold_numbers], 2))

    # Fill the rest with random numbers within the range, excluding already chosen ones
    all_numbers = list(range(1, 51))
    remaining_numbers = [num for num in all_numbers if num not in predictions]
    predictions.extend(random.sample(remaining_numbers, 1))  # Adds 1 more to make it 5

    return predictions


# Use the function
predictions = predict_next_numbers('data.txt')
print("Predicted next 5 numbers:", predictions)