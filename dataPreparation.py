# dataPreparation.py

import pandas as pd
import numpy as np
import ta
import os
import json

def download_data(ticker, start_date, end_date):
    """
    Downloads historical stock data for a given ticker.

    Parameters:
    - ticker (str): Stock ticker symbol (e.g., 'AAPL').
    - start_date (str): Start date in the format 'YYYY-MM-DD'.
    - end_date (str): End date in the format 'YYYY-MM-DD'.

    Returns:
    - pd.DataFrame: A DataFrame with historical stock data.
    """
    import yfinance as yf
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def download_macro_data(start_date, end_date):
    """
    Downloads and processes macroeconomic data for SPY and VIX.

    Parameters:
    - start_date (str): Start date in the format 'YYYY-MM-DD'.
    - end_date (str): End date in the format 'YYYY-MM-DD'.

    Returns:
    - pd.DataFrame: A DataFrame containing SPY and VIX data.
    """
    spy = download_data('SPY', start_date, end_date)[['Date', 'Close']].copy()
    spy.rename(columns={'Close': 'SPY_Close'}, inplace=True)
    
    vix = download_data('^VIX', start_date, end_date)[['Date', 'Close']].copy()
    vix.rename(columns={'Close': 'VIX_Close'}, inplace=True)
    
    macro_df = pd.merge(spy, vix, on='Date', how='outer')  # Merge on 'Date'
    macro_df.sort_values('Date', inplace=True)  # Sort by date
    macro_df = macro_df.ffill().bfill()  # Fill missing values
    return macro_df

def add_technical_indicators(df):
    """
    Adds various technical indicators to the stock data.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing stock data.

    Returns:
    - pd.DataFrame: A DataFrame with added technical indicators.
    """
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['Stoch_RSI'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=14).stochrsi()
    df['TSI'] = ta.momentum.TSIIndicator(close=df['Close'], window_slow=25, window_fast=13).tsi()

    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    cci = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()
    df['CCI'] = cci

    dpo = ta.trend.DPOIndicator(close=df['Close'], window=20).dpo()
    df['DPO'] = dpo

    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr.average_true_range()

    ulcer = ta.volatility.UlcerIndex(close=df['Close'], window=14).ulcer_index()
    df['Ulcer_Index'] = ulcer

    obv = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['OBV'] = obv

    cmf = ta.volume.ChaikinMoneyFlowIndicator(
        high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20
    ).chaikin_money_flow()
    df['CMF'] = cmf

    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()

    return df

def add_performance_labels(df, n=5, k=20):
    """
    Adds performance labels to the DataFrame to indicate top performers.

    Parameters:
    - df (pd.DataFrame): DataFrame with stock data.
    - n (int): Days ahead to calculate performance.
    - k (int): Number of top performers to label.

    Returns:
    - pd.DataFrame: DataFrame with performance and top performer labels.
    """
    df['Performance'] = (df.groupby('Ticker')['Close'].shift(-n) - df['Close']) / df['Close']
    df['Top_Performer'] = df.groupby('Date')['Performance'].rank(ascending=False, method='first') <= k
    df['Top_Performer'] = df['Top_Performer'].astype(int)
    return df

def add_time_features(df):
    """
    Adds time-related features such as year, month, and day of the week.

    Parameters:
    - df (pd.DataFrame): DataFrame with stock data.

    Returns:
    - pd.DataFrame: DataFrame with added time features.
    """
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    return df

def merge_with_macro(df, macro_df):
    """
    Merges stock data with macroeconomic data on 'Date'.

    Parameters:
    - df (pd.DataFrame): DataFrame with stock data.
    - macro_df (pd.DataFrame): DataFrame with macroeconomic data.

    Returns:
    - pd.DataFrame: Merged DataFrame.
    """
    df = pd.merge(df, macro_df, on='Date', how='left')
    df.sort_values('Date', inplace=True)
    df[["SPY_Close", "VIX_Close"]] = df[["SPY_Close", "VIX_Close"]].ffill().bfill()
    return df

def prepare_dataset(df, tickers, all_tickers, window_size=60, output_dir='processed_data'):
    """
    Prepares the dataset for model training or backtesting by creating sequences and saving sample mappings.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing all data.
    - tickers (list): List of ticker symbols to include in the sequences.
    - all_tickers (list): List of all ticker symbols used during training.
    - window_size (int): Number of time steps in each sequence.
    - output_dir (str or None): Directory where the prepared data and mappings will be saved. If None, saving is skipped.

    Returns:
    - X (np.ndarray): Feature sequences.
    - y (np.ndarray): Labels.
    - features (list): List of feature names.
    """
    # One-Hot Encode the 'Ticker' column without dropping any first category
    df = pd.get_dummies(df, columns=['Ticker'], drop_first=False)

    for ticker in all_tickers:
        ticker_col = f'Ticker_{ticker}'
        if ticker_col not in df.columns:
            df[ticker_col] = 0

    # Define feature columns (excluding 'Date', 'Performance', 'Top_Performer')
    features = [
        'Close', 'SMA_10', 'EMA_10', 'RSI', 'Stoch_RSI', 'TSI', 'MACD', 'MACD_Signal',
        'CCI', 'DPO', 'BB_High', 'BB_Low', 'ATR', 'Ulcer_Index', 'OBV', 'CMF',
        'SPY_Close', 'VIX_Close', 'Year', 'Month', 'DayOfWeek'
    ]

    # Include all One-Hot Encoded Ticker columns based on all_tickers
    ticker_cols = [f'Ticker_{ticker}' for ticker in all_tickers]
    features.extend(ticker_cols)

    # Impute NaN values for technical indicators using forward-fill and backward-fill
    df[features] = df[features].ffill().bfill()

    # Drop rows with missing 'Top_Performer' labels
    df = df.dropna(subset=['Top_Performer']).copy()

    X_list, y_list = [], []
    mapping_list = []

    sample_index = 0

    # For each ticker, create sequences
    for ticker in tickers:
        ticker_col = f'Ticker_{ticker}'
        if ticker_col not in df.columns:
            continue
        ticker_data = df[df[ticker_col] == 1].sort_values('Date').reset_index(drop=True)
        ticker_features = ticker_data[features].values
        y_values = ticker_data['Top_Performer'].values
        dates = ticker_data['Date'].values
        next_day_returns = ticker_data['Performance'].values

        for i in range(window_size, len(ticker_features)):
            # Create sequence
            X_list.append(ticker_features[i - window_size:i])
            y_list.append(y_values[i])

            # Record mapping
            mapping_list.append({
                'sample_index': sample_index,
                'Date': dates[i],
                'Ticker': ticker,
                'Next_Day_Return': next_day_returns[i]
            })
            sample_index += 1

    # Convert lists to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)

    # Create sample mapping DataFrame
    sample_mapping = pd.DataFrame(mapping_list)

    # Save sample_mapping.csv if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        sample_mapping_path = os.path.join(output_dir, 'sample_mapping.csv')
        sample_mapping.to_csv(sample_mapping_path, index=False)
        print(f"Sample mapping saved to {sample_mapping_path}")
    else:
        print("Output directory not provided. Skipping saving sample mapping.")

    return X, y, features


def save_prepared_data(X, y, features, output_dir='processed_data'):
    """
    Saves prepared dataset (X and y) and feature metadata to disk.

    Parameters:
    - X (np.ndarray): Feature sequences.
    - y (np.ndarray): Labels.
    - features (list): List of feature names.
    - output_dir (str): Directory to save files.

    Saves:
    - X.npy: NumPy array of features.
    - y.npy: NumPy array of labels.
    - features.json: JSON file with feature names.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    with open(os.path.join(output_dir, 'features.json'), 'w') as f:
        json.dump(features, f, indent=4)

def load_prepared_data(input_dir='processed_data'):
    """
    Loads prepared dataset (X and y) and feature metadata from disk.

    Parameters:
    - input_dir (str): Directory to load files from.

    Returns:
    - X (np.ndarray): Feature sequences.
    - y (np.ndarray): Labels.
    - features (list): List of feature names.
    """
    X = np.load(os.path.join(input_dir, 'X.npy'))
    y = np.load(os.path.join(input_dir, 'y.npy'))
    with open(os.path.join(input_dir, 'features.json'), 'r') as f:
        features = json.load(f)
    return X, y, features