# LSTM Stock Price Prediction using PyTorch

A deep learning model that predicts Apple (AAPL) stock prices using Long Short-Term Memory (LSTM) neural networks implemented in PyTorch.

## Overview

This project implements a time series forecasting model that uses historical stock price data to predict future closing prices. The model uses a 60-day sliding window approach where it analyzes the past 60 days of stock prices to predict the next day's closing price.

## Features

- **LSTM Neural Network**: Multi-layer LSTM architecture for time series prediction
- **Data Preprocessing**: MinMax scaling for normalized input data
- **Sequence Generation**: Creates time-windowed sequences for training
- **GPU Support**: Automatic GPU detection and usage if available
- **Visualization**: Plots actual vs predicted prices with date indexing
- **Performance Metrics**: RMSE evaluation for model accuracy

## Requirements

```bash
pip install pandas numpy matplotlib scikit-learn torch
```

### Required Libraries:
- `pandas` - Data manipulation and CSV reading
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `scikit-learn` - Data preprocessing and metrics
- `torch` - PyTorch deep learning framework

## Data Requirements

The model expects a CSV file named `AAPL.csv` with the following structure:

```csv
Date,Open,High,Low,Close,Volume
2020-01-01,100.50,102.00,99.75,101.25,1000000
2020-01-02,101.25,103.00,100.50,102.50,1100000
...
```

## Model Architecture

```
Input Layer: 60 time steps × 1 feature (Close price)
    ↓
LSTM Layer 1: 64 hidden units
    ↓
LSTM Layer 2: 64 hidden units
    ↓
Fully Connected Layer 1: 32 units + ReLU
    ↓
Output Layer: 1 unit (Predicted price)
```

## Configuration

### Key Parameters:
- `SEQ_LEN = 60`: Number of previous days used for prediction
- `train_size = 0.8`: 80% of data used for training
- `batch_size = 64`: Training batch size
- `epochs = 20`: Number of training epochs
- `learning_rate = 0.001`: Adam optimizer learning rate
- `hidden_size = 64`: LSTM hidden layer size
- `num_layers = 2`: Number of LSTM layers

## Usage

1. **Prepare your data**: Ensure you have a CSV file with Date and Close price columns
2. **Update the filename**: Change `"AAPL.csv"` to your actual filename
3. **Run the script**:
   ```bash
   python lstm_stock_prediction.py
   ```

## Output

The script will:
1. Train the LSTM model for 20 epochs
2. Display training loss for each epoch
3. Generate predictions on test data
4. Show a plot comparing actual vs predicted prices
5. Calculate and display RMSE (Root Mean Square Error)

## Model Performance

The model's performance is evaluated using:
- **Visual Comparison**: Plot showing actual vs predicted prices over time

## Customization


### Using Different Stocks:
1. Replace `AAPL.csv` with your stock data file
2. Ensure the CSV has the same column structure
3. Update the title in the plot if desired

## Important Notes

- **Data Quality**: Ensure your CSV data is clean and has no missing values
- **Reproducibility**: Random seeds are set for consistent results across runs
- **Financial Disclaimer**: This model is for educational purposes only and should not be used for actual trading decisions

## File Structure

```
project/
├── lstm_stock_prediction.py  # Main script
├── AAPL.csv                  # Stock data (your file)
└── README.md                # This file
```


## License

This project is open source and available under the MIT License.
