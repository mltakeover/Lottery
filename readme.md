# Number Prediction Tool

[![GitHub stars](https://img.shields.io/github/stars/username/number-prediction-tool?style=social)](https://github.com/username/number-prediction-tool)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/username/number-prediction-tool)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated machine learning tool for predicting number sequences based on historical data. This tool uses multiple prediction models including XGBoost, LightGBM, CatBoost, and Artificial Neural Networks (ANN) to generate predictions for two sets of numbers: main numbers (1-50) and bonus numbers (1-12).

## Features

- **Multiple ML Models**: Utilizes XGBoost, LightGBM, CatBoost, and ANN for comprehensive predictions
- **Ensemble Predictions**: Combines results from all models for improved accuracy
- **Statistical Analysis**: Performs frequency analysis on historical data
- **Data Visualization**: Generates frequency plots for both main and bonus numbers
- **Model Evaluation**: Calculates Mean Absolute Error (MAE) for all models
- **Cross-Validation**: Validates model performance using train-test splits
- **GPU Support**: Automatically detects and utilizes GPU for TensorFlow operations

## Requirements

- Python 3.8 or higher
- Required Python packages:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - xgboost
  - lightgbm
  - catboost
  - tensorflow
  - configparser

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/number-prediction-tool.git
   cd number-prediction-tool
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The tool requires a `config.ini` file in the same directory as the script. Here's an example configuration:

```ini
[SETTINGS]
window_size = 5
main_num_range = 50
bonus_num_range = 12
num_main_to_predict = 5
num_bonus_to_predict = 2
test_size = 0.2
n_splits = 5
```

### Configuration Parameters

- `window_size`: Number of previous draws to consider for prediction
- `main_num_range`: Maximum value for main numbers (e.g., 50)
- `bonus_num_range`: Maximum value for bonus numbers (e.g., 12)
- `num_main_to_predict`: Number of main numbers to predict (e.g., 5)
- `num_bonus_to_predict`: Number of bonus numbers to predict (e.g., 2)
- `test_size`: Proportion of data to use for testing (e.g., 0.2 = 20%)
- `n_splits`: Number of splits for cross-validation

## Data Format

The tool expects a data file (`data.txt` by default) containing historical number draws. Each line should contain 7 numbers: 5 main numbers followed by 2 bonus numbers, separated by spaces.

Example:
```
7 9 10 15 23 3 8
12 15 28 31 44 2 12
1 4 23 24 45 5 10
```

## Usage

### Basic Usage

Run the script with the default data file location (expects `data.txt` in the same directory):

```bash
python number_prediction.py
```

### Specify Data File

You can specify a different data file path:

```bash
python number_prediction.py /path/to/your/data.txt
```

## Output

The script produces:

1. **Prediction Results**: Displays predictions from each model and the ensemble for both main and bonus numbers
2. **Model Evaluations**: Shows Mean Absolute Error (MAE) for each model on test data and cross-validation
3. **Visualization Files**: Generates frequency plots saved as PNG files:
   - `main_numbers_frequency.png`
   - `bonus_numbers_frequency.png`

Example output:
```
--- Predictions ---

Main Numbers:
Xgboost: 3 17 24 33 42
Lightgbm: 5 13 27 36 44
Catboost: 7 19 23 31 47
Ann: 4 15 22 35 41
Ensemble: 5 17 24 35 42

Bonus Numbers:
Xgboost: 3 8
Lightgbm: 2 9
Catboost: 4 7
Ann: 3 10
Ensemble: 3 8

--- Model Evaluations (Mean Absolute Error) ---

Main Numbers Models (Test Set):
Xgboost_mae: 5.23
Lightgbm_mae: 5.47
Catboost_mae: 5.31
Ann_mae: 5.62

Main Numbers Models (Cross-Validation):
Xgboost_cv_mae: 5.35
Lightgbm_cv_mae: 5.52
Catboost_cv_mae: 5.40
Ann_cv_mae: 5.62

Bonus Numbers Models (Test Set):
Xgboost_mae: 2.15
Lightgbm_mae: 2.23
Catboost_mae: 2.19
Ann_mae: 2.31

Bonus Numbers Models (Cross-Validation):
Xgboost_cv_mae: 2.18
Lightgbm_cv_mae: 2.25
Catboost_cv_mae: 2.21
Ann_cv_mae: 2.31
```

## How It Works

1. **Data Loading**: The script loads historical number data from the provided file
2. **Feature Creation**: Creates features based on the configured window size
3. **Model Training**: Trains multiple ML models on the historical data
4. **Prediction**: Each model makes predictions for the next set of numbers
5. **Ensemble**: Combines predictions from all models for final results
6. **Evaluation**: Calculates performance metrics for each model
7. **Visualization**: Generates frequency plots for historical data

## Advanced Usage

### GPU Acceleration

The script automatically detects and uses available GPUs for TensorFlow operations. No additional configuration is needed.

### Customizing Models

You can modify the script to adjust model parameters or add new models. The main model training functions are:
- `train_model()` for XGBoost, LightGBM, and CatBoost
- `train_ann_model()` for the Artificial Neural Network

## Hugging Face Integration

This tool can be deployed as a Hugging Face Space for interactive usage:

1. Fork this repository to your Hugging Face account
2. Create a new Space using the Gradio or Streamlit framework
3. Implement a simple UI that allows users to:
   - Upload their own data file
   - Configure parameters
   - View predictions and visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and entertainment purposes only. The predictions are based on historical data and statistical models, and should not be used for financial decisions or gambling.
