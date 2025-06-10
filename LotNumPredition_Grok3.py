import random
from collections import Counter


def predict_next_numbers(filename):
    # Initialize dictionaries for main numbers (1-50) and bonus numbers (1-12)
    main_data = {i: {'freq': 0, 'last_seen': 0} for i in range(1, 51)}
    bonus_data = {i: {'freq': 0, 'last_seen': 0} for i in range(1, 13)}
    draw_count = 0

    # Read and process numbers from file
    try:
        with open(filename, 'r') as file:
            for line in file:
                draw_count += 1
                numbers = [int(num) for num in line.strip().split('\t') if num.strip().isdigit()]

                # Validate and separate main and bonus numbers
                if len(numbers) != 7:
                    print(f"Warning: Skipping invalid line with {len(numbers)} numbers: {line.strip()}")
                    continue

                main_numbers = [n for n in numbers[:5] if 1 <= n <= 50]
                bonus_numbers = [n for n in numbers[5:] if 1 <= n <= 12]

                # Update main numbers stats
                for num in main_numbers:
                    main_data[num]['freq'] += 1
                    main_data[num]['last_seen'] = draw_count

                # Update bonus numbers stats
                for num in bonus_numbers:
                    bonus_data[num]['freq'] += 1
                    bonus_data[num]['last_seen'] = draw_count

    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return [], []
    except ValueError:
        print(f"Error: Invalid number format in file {filename}.")
        return [], []

    if draw_count == 0:
        print("Error: No valid draws found in the file.")
        return [], []

    # Calculate weighted scores for main numbers
    freq_weight, recency_weight = 0.6, 0.4  # Adjustable weights
    main_scores = {}
    for num, data in main_data.items():
        freq_score = data['freq'] / (draw_count * 5)  # Normalize by max possible appearances
        recency_score = 1 - (draw_count - data['last_seen']) / draw_count if data['last_seen'] > 0 else 0
        main_scores[num] = freq_weight * freq_score + recency_weight * recency_score

    # Calculate weighted scores for bonus numbers
    bonus_scores = {}
    for num, data in bonus_data.items():
        freq_score = data['freq'] / (draw_count * 2)  # Normalize by max possible appearances
        recency_score = 1 - (draw_count - data['last_seen']) / draw_count if data['last_seen'] > 0 else 0
        bonus_scores[num] = freq_weight * freq_score + recency_weight * recency_score

    # Predict main numbers: 3 from top scores, 2 random
    main_predictions = []
    sorted_main = sorted(main_scores.items(), key=lambda x: x[1], reverse=True)
    main_predictions.extend([num for num, _ in sorted_main[:3]])  # Top 3 by score
    remaining_main = [num for num in range(1, 51) if num not in main_predictions]
    main_predictions.extend(random.sample(remaining_main, 2))  # 2 random
    main_predictions.sort()  # Sort for readability

    # Predict bonus numbers: 1 from top scores, 1 random
    bonus_predictions = []
    sorted_bonus = sorted(bonus_scores.items(), key=lambda x: x[1], reverse=True)
    bonus_predictions.append(sorted_bonus[0][0])  # Top 1 by score
    remaining_bonus = [num for num in range(1, 13) if num not in bonus_predictions]
    bonus_predictions.append(random.sample(remaining_bonus, 1)[0])  # 1 random
    bonus_predictions.sort()  # Sort for readability

    return main_predictions, bonus_predictions


# Use the function
main_nums, bonus_nums = predict_next_numbers('data.txt')
print("Predicted main numbers (1-50):", main_nums)
print("Predicted bonus numbers (1-12):", bonus_nums)