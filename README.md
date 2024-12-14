# Machine Learning Final
# 12/13/2024
# Group 5

---
## Google Colab for Stock Prediction
- https://docs.google.com/document/d/1VIZK3TqRaKlOF10cOuRGj7rcpHvGt9WyEblCuq-c8uQ/edit?usp=sharing
---

## Features

### 1. **Data Collection**
- Retrieves historical stock and macroeconomic data using yfinance
- Collects data for a diverse set of tickers and macroeconomic indicators like S&P 500 (SPY) and VIX (Volatility Index).

### 2. **Data Preparation**
- Adds technical indicators like RSI, MACD, Bollinger Bands, and more to stock data.
- Merges stock data with macroeconomic data.
- Labels stocks as top performers based on future returns for supervised learning.
- Creates sequences of features for time-series modeling.

### 3. **Model Definitions**
- **Transformer-based Model**:
  - Implements a stacked Transformer encoder for binary classification.
  - Utilizes multi-head attention and feed-forward layers for efficient learning.

(Not trained)
- **LSTM-based Model**: 
  - Builds a deep LSTM model for sequential data processing.
  - Includes regularization layers for improved generalization.

### 4. **Model Training**
- Performs cross-validation using TimeSeriesSplit.
- Balances imbalanced classes using computed class weights.
- Saves the best model and its associated artifacts (e.g., scalers).

### 5. **Backtesting Framework**
- Tests model predictions on unseen data to simulate trading.
- Constructs portfolios using predicted top stocks and calculates key metrics:
  - Cumulative Return
  - Sharpe Ratio
  - Maximum Drawdown
- Generates visualizations, including equity curves and return distributions.

---

## Directory Structure

- **data**: Raw stock and macroeconomic data.
- **processed_data**: Prepared datasets with features and labels.
- **artifacts**: Trained models, scalers, and metrics.
- **backtest_results**: Backtesting results, including portfolio performance and charts.

---
## Our Outputs

### Fold Results
- **Fold 1**:  
  Cumulative Return: **276.41%**  
  Sharpe Ratio: **4.361688**

- **Fold 2**:  
  Cumulative Return: **277.84%**  
  Sharpe Ratio: **4.135574**

- **Fold 3**:  
  Cumulative Return: **262.02%**  
  Sharpe Ratio: **4.014116**

- **Fold 4**:  
  Cumulative Return: **254.42%**  
  Sharpe Ratio: **3.885298**

### Averages
- **Average Cumulative Return across all folds**: **267.67%**  
- **Average Sharpe Ratio across all folds**: **4.10**
