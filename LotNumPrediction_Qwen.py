import numpy as np
from sklearn.linear_model import LinearRegression


# Step 1: Load the data from the file
def load_data(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into numbers (assuming tab or space separation)
            numbers = list(map(int, line.strip().split()))
            if len(numbers) == 5:  # Ensure each row has exactly 5 numbers
                sequences.append(numbers)
    return np.array(sequences)


# Step 2: Prepare the data for training
def prepare_data(sequences):
    X = np.arange(1, len(sequences) + 1).reshape(-1, 1)  # Input features (row indices)
    y = np.array(sequences)  # Target values (all rows of numbers)
    return X, y


# Step 3: Train the model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)  # Train the model on all 5 numbers per row
    return model


# Step 4: Predict the next numbers
def predict_next_numbers(model, current_length, num_predictions):
    next_indices = np.arange(current_length + 1, current_length + num_predictions + 1).reshape(-1, 1)
    predictions = model.predict(next_indices)
    return np.round(predictions).astype(int)  # Round predictions to integers


# Main function
def main():
    # File path to the data.txt file
    file_path = 'data.txt'

    # Load the sequences from the file
    sequences = load_data(file_path)

    # Prepare the data for training
    X, y = prepare_data(sequences)

    # Train the model
    model = train_model(X, y)

    # Predict the next 5 numbers (as a single row)
    next_numbers = predict_next_numbers(model, len(sequences), 1)

    print("Predicted next row of 5 numbers:", next_numbers[0])


if __name__ == "__main__":
    main()