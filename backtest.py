# backtest.py

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from dataCollection import download_stock_data, download_macro_data
from dataPreparation import (
    add_technical_indicators, add_performance_labels, add_time_features,
    merge_with_macro, prepare_dataset
)

def run_backtest(
    processed_data_dir='processed_data',
    artifacts_dir='artifacts',
    backtest_output_dir='backtest_results',
    top_k=20,
    raw_data_dir='data',
    graph_start_date='2023-09-01',  # Starting date for plotting equity curve
    test_start_date='2023-09-01',   # Start date for backtest
    test_end_date='2023-09-28',      # End date for backtest
    fold=1                            # Fold number for cross-validation
):
    print(f"=== Starting Backtest for Fold {fold} ===")

    # Step [1/9]: Load Metadata
    print("[1/9] Loading metadata...")
    metadata_path = os.path.join(processed_data_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("metadata.json not found in the processed_data directory.")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    tickers = metadata.get('tickers', [])
    window_size = metadata.get('window_size', 60)
    features = metadata.get('features', [])
    print("Metadata loaded successfully.")
    print(f"Tickers: {len(tickers)} tickers")
    print(f"Window size: {window_size}")
    print(f"Number of features: {len(features)}")

    # Validate tickers list
    if not tickers:
        raise ValueError("No tickers found in metadata.json.")

    # Ensure test_start_date <= test_end_date
    test_start = pd.to_datetime(test_start_date)
    test_end = pd.to_datetime(test_end_date)
    if test_start > test_end:
        raise ValueError(f"test_start_date ({test_start_date}) is after test_end_date ({test_end_date}). Please adjust the dates accordingly.")

    # Create fold-specific output directory
    fold_output_dir = os.path.join(backtest_output_dir, f'fold_{fold}')
    os.makedirs(fold_output_dir, exist_ok=True)

    print("Downloading stock and macro data for the test period...")
    stock_data = download_stock_data(tickers, test_start_date, test_end_date)
    macro_data = download_macro_data(test_start_date, test_end_date)
    print("Data downloaded.")
    print(f"Stock data shape: {stock_data.shape}")
    print(f"Macro data shape: {macro_data.shape}")

    os.makedirs(raw_data_dir, exist_ok=True)

    print("Converting Dates to datetime...")
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    macro_data['Date'] = pd.to_datetime(macro_data['Date'])

    print("Adding technical indicators...")
    stock_data = add_technical_indicators(stock_data)
    print("Technical indicators added.")

    print("Adding time features...")
    stock_data = add_time_features(stock_data)
    print("Time features added.")

    print("Merging with macro data...")
    stock_data = merge_with_macro(stock_data, macro_data)
    print("Macro merge complete.")

    print("Adding performance labels...")
    stock_data = add_performance_labels(stock_data, n=5, k=top_k)
    print("Labels added.")

    print("Dropping NaN values...")
    before_drop = len(stock_data)
    stock_data.dropna(inplace=True)
    after_drop = len(stock_data)
    print(f"Dropped {before_drop - after_drop} rows due to NaN.")
    print(f"Remaining rows: {after_drop}")

    print("Sorting data by Date and Ticker...")
    stock_data = stock_data.sort_values(by=['Date', 'Ticker'])

    print("Preparing dataset for model inference...")
    X_test, y_test, _ = prepare_dataset(
        stock_data, 
        tickers, 
        tickers, 
        window_size=window_size, 
        output_dir=None
    )
    print("Output directory not provided. Skipping saving sample mapping.")
    print(f"Prepared test dataset. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    print("Loading model and scaler...")
    model_path = os.path.join(artifacts_dir, f'model_fold{fold}.keras')
    scaler_path = os.path.join(artifacts_dir, f'scaler_fold{fold}.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at '{scaler_path}'.")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")

    print("Scaling test data and predicting...")
    # Reshape X_test for scaler
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    y_pred_prob = model.predict(X_test_scaled).reshape(-1)
    print("Predictions complete.")

    print("Reconstructing sample mapping for predictions...")
    num_samples = X_test.shape[0]
    end_indices = np.arange(window_size - 1, window_size - 1 + num_samples)
    test_mapping = stock_data.iloc[end_indices].copy()

    test_mapping['Pred_Prob'] = y_pred_prob
    print("'Pred_Prob' assigned from model predictions.")

    if 'Performance' in test_mapping.columns:
        test_mapping['Next_Day_Return'] = test_mapping['Performance']
        print("'Next_Day_Return' assigned from 'Performance' column.")
    else:
        raise ValueError("Performance column not found in test_mapping.")

    # Debugging: Print columns in test_mapping
    print("Columns in test_mapping:", test_mapping.columns.tolist())

    print("Constructing daily portfolio and calculating performance...")
    daily_groups = test_mapping.groupby('Date')
    portfolio_values = []
    dates = []
    capital = 1.0  # Starting capital

    for date, group in daily_groups:
        if 'Pred_Prob' not in group.columns or 'Next_Day_Return' not in group.columns:
            print(f"Skipping date {date} due to missing columns.")
            continue
        # Select top_k stocks based on predicted probability
        top_stocks = group.sort_values('Pred_Prob', ascending=False).head(top_k)
        # Calculate average daily return
        daily_return = top_stocks['Next_Day_Return'].mean()
        # Update capital
        capital *= (1 + daily_return)
        portfolio_values.append(capital)
        dates.append(date)

    # Create Portfolio DataFrame
    portfolio_df = pd.DataFrame({'Date': dates, 'Portfolio_Value': portfolio_values})
    portfolio_df.set_index('Date', inplace=True)
    # Calculate Daily Returns
    portfolio_df['Daily_Return'] = portfolio_df['Portfolio_Value'].pct_change().fillna(0)
    # Calculate Cumulative Return
    cumulative_return = portfolio_df['Portfolio_Value'].iloc[-1] / portfolio_df['Portfolio_Value'].iloc[0] - 1
    # Calculate Sharpe Ratio
    daily_returns = portfolio_df['Daily_Return']
    sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else np.nan
    print(f"Cumulative Return: {cumulative_return*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    print("Saving portfolio results...")
    os.makedirs(fold_output_dir, exist_ok=True)
    portfolio_csv_path = os.path.join(fold_output_dir, 'portfolio.csv')
    portfolio_df.to_csv(portfolio_csv_path)
    print(f"Portfolio data saved to '{portfolio_csv_path}'.")

    print("Plotting portfolio equity curve...")
    normalized_portfolio = portfolio_df['Portfolio_Value'] / portfolio_df['Portfolio_Value'].iloc[0]
    plt.figure(figsize=(10, 6))
    plt.plot(normalized_portfolio.index, normalized_portfolio, label='Portfolio', color='blue')
    plt.title(f'Portfolio Equity Curve - Fold {fold}')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    equity_curve_path = os.path.join(fold_output_dir, 'portfolio_equity_curve.png')
    plt.savefig(equity_curve_path)
    plt.close()
    print(f"Portfolio equity curve chart saved to '{equity_curve_path}'.")

    print("Plotting daily returns distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(daily_returns, kde=True, color='green')
    plt.title(f'Daily Returns Distribution - Fold {fold}')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    returns_dist_path = os.path.join(fold_output_dir, 'daily_returns_distribution.png')
    plt.savefig(returns_dist_path)
    plt.close()
    print(f"Daily returns distribution chart saved to '{returns_dist_path}'.")

    print("Saving performance metrics...")
    metrics = {
        'cumulative_return': float(cumulative_return),
        'sharpe_ratio': float(sharpe_ratio)
    }
    metrics_path = os.path.join(fold_output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Performance metrics saved to '{metrics_path}'.")

    print(f"=== Backtest Completed for Fold {fold} ===")

    return metrics

if __name__ == "__main__":
    run_backtest()
