# main.py

import os
import json
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from dataCollection import download_stock_data, download_macro_data, save_raw_data
from dataPreparation import (
    add_technical_indicators, add_performance_labels, add_time_features,
    merge_with_macro, prepare_dataset, save_prepared_data
)
from TransformerModelDefinition import create_classification_model
from modelTraining import train_and_evaluate_model, save_metrics
from utils import predict_top_stocks
from backtest import run_backtest

def main():
    # Define Tickers
    tickers = list(set([
        # Original 8
        'MSFT', 'NFLX', 'AMZN', 'GOOG', 'META', 'TXN', 'NVDA', 'AAPL',

        # Software and Services
        'CRM', 'INTU', 'NOW', 'ADSK', 'WDAY', 'TEAM', 'PANW', 'FTNT',
        'CRWD', 'OKTA', 'DDOG', 'SNOW', 'ZM', 'DOCU', 'DBX', 'BOX',

        # Semiconductors and Semiconductor Equipment
        'AMD', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC', 'ADI', 'MRVL', 'ON', 
        'NXPI', 'SWKS', 'MCHP', 'ASML', 'TER',

        # Technology Hardware, Storage, and Peripherals
        'DELL', 'HPQ', 'STX', 'WDC', 'NTAP', 'PSTG', 'LOGI', 'XRX', 
        'FJTSY', 'LNVGY', 'SSNLF', 'SONY', 'RICOY', 'BRTHY', 
        'NIPNF', 'WDC',

        # Internet and Direct Marketing Retail
        'EBAY', 'BABA', 'JD', 'PDD', 'ETSY', 'W', 'SHOP', 'MELI', 'RKUNY', 
        'REAL', 'CHWY', 'CPNG', 'ZLNDY', 'ASOMY', 'BHOOY', 
        'OCDDY', 'SFIX',

        # IT Services
        'IBM', 'ACN', 'CTSH', 'INFY', 'TCS', 'WIT', 'CAPMF', 
        'DXC', 'EPAM', 'GLOB', 'DAVA', 'AEXAY', 'GIB'
    ]))

    # Define Parameters
    start_date = '2010-01-01'
    end_date = '2023-10-01'
    window_size = 60
    num_folds = 5  # Define the number of cross-validation folds

    # Define Directories
    raw_data_dir = 'data'
    prepared_data_dir = 'processed_data'
    artifacts_dir = 'artifacts'
    backtest_output_dir = 'backtest_results'
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(prepared_data_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(backtest_output_dir, exist_ok=True)

    # Step 1: Data Collection
    print("=== Starting Data Collection ===")
    stock_data = download_stock_data(tickers, start_date, end_date)
    print(f"Stock data downloaded. Shape: {stock_data.shape}")
    print("Stock data sample:")
    print(stock_data.head())

    macro_data = download_macro_data(start_date, end_date)
    print("Macro data downloaded.")
    print(f"Macro data shape: {macro_data.shape}")
    print("Macro data sample:")
    print(macro_data.head())

    # Step 2: Save Raw Data
    print("Saving raw data...")
    save_raw_data(stock_data, macro_data, output_dir=raw_data_dir)
    print(f"Raw data saved to '{raw_data_dir}' directory.")

    # Step 3: Data Preparation
    print("=== Data Preparation ===")
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    macro_data['Date'] = pd.to_datetime(macro_data['Date'])

    print("Adding technical indicators...")
    stock_data = add_technical_indicators(stock_data)
    print("Technical indicators added. Sample after indicators:")
    print(stock_data.head())

    print("Adding time features...")
    stock_data = add_time_features(stock_data)
    print("Time features added. Sample of time-related columns:")
    print(stock_data[['Date', 'Year', 'Month', 'DayOfWeek']].head())

    print("Merging with macro data...")
    stock_data = merge_with_macro(stock_data, macro_data)
    print("Merged with macro data. Sample of macro columns:")
    print(stock_data[['Date', 'SPY_Close', 'VIX_Close']].head())

    # Add Performance Labels
    print("Adding performance labels...")
    stock_data = add_performance_labels(stock_data, n=5, k=20)
    print("Performance labels added. Sample of performance and label columns:")
    print(stock_data[['Date', 'Ticker', 'Close', 'Performance', 'Top_Performer']].head(10))

    # Data Cleaning
    print("NaN count before dropping:")
    print(stock_data.isna().sum())
    before_drop = len(stock_data)
    stock_data.dropna(inplace=True)
    after_drop = len(stock_data)
    print(f"Dropped {before_drop - after_drop} rows due to NaN values. Remaining rows: {after_drop}")

    # Top Performer Distribution
    top_dist = stock_data['Top_Performer'].value_counts()
    print("Distribution of Top_Performer:")
    print(top_dist)
    if 1 in top_dist:
        pos_ratio = top_dist[1] / (top_dist[0] + top_dist[1]) * 100
        print(f"Percentage of top performers: {pos_ratio:.2f}%")
    else:
        print("No top performers found after filtering.")

    # Daily Top Performers Statistics
    top_by_day = stock_data.groupby('Date')['Top_Performer'].sum()
    print("Some stats about daily top performers:")
    print(f"Mean top performers per day: {top_by_day.mean():.2f}")
    print(f"Median top performers per day: {top_by_day.median():.2f}")
    print("Example days with their top performer counts:")
    print(top_by_day.head(10))

    # Step 4: Prepare Dataset for Model
    print("Preparing dataset for model...")
    X, y, features = prepare_dataset(stock_data, tickers, tickers, window_size=window_size, output_dir=prepared_data_dir)
    print("Dataset prepared.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Feature list:", features)
    print("Sample y values:", y[:10])

    # Save Prepared Data
    print("Saving prepared data...")
    save_prepared_data(X, y, features, output_dir=prepared_data_dir)
    print(f"Prepared data saved to '{prepared_data_dir}' directory.")

    # Save Metadata
    metadata = {
        'tickers': tickers,
        'window_size': window_size,
        'features': features
    }
    metadata_path = os.path.join(prepared_data_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to '{metadata_path}'.")

    # Label Distribution
    y_counts = np.bincount(y)
    print("Distribution of labels after sequencing:")
    for i, count in enumerate(y_counts):
        perc = count / len(y) * 100
        print(f"Label {i}: {count} samples ({perc:.2f}%)")

    # Compute Class Weights for Imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights))
    print("Class weights computed:", class_weights)

    '''
    # Step 5: Model Training & Evaluation (Commented Out)
    print("=== Training & Evaluation ===")
    results, trained_model = train_and_evaluate_model(
        X, y, 
        create_classification_model, 
        class_weight=class_weights,  # Corrected parameter name
        n_splits=5, 
        features=features, 
        output_dir='artifacts'
    )
    print("Cross-validation completed.")
    for fold_idx, (acc, prec, rec, f1) in enumerate(results, start=1):
        print(f"Fold {fold_idx}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    print("Saving metrics...")
    save_metrics(results, 'artifacts')
    print("Metrics saved to 'artifacts' directory.")
    '''

    # Step 6: Run Backtest for All Folds
    print("=== Running Backtest for All Folds ===")
    all_metrics = []
    for fold in range(1, num_folds):
        print(f"\n--- Backtesting Fold {fold} ---")
        metrics = run_backtest(
            processed_data_dir=prepared_data_dir,
            artifacts_dir=artifacts_dir,
            backtest_output_dir=backtest_output_dir,
            top_k=20,
            raw_data_dir=raw_data_dir,
            graph_start_date='2023-10-01',  # Starting date for equity curve plot
            test_start_date='2023-10-01',   # Start date for backtest
            test_end_date='2024-10-01',     # End date for backtest
            fold=fold                        # Current fold number
        )
        metrics['fold'] = fold
        all_metrics.append(metrics)
        print(f"Backtest for Fold {fold} completed with Cumulative Return: {metrics['cumulative_return']*100:.2f}% and Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

    # Aggregate Metrics
    print("\n=== Aggregating Backtest Metrics Across All Folds ===")
    metrics_df = pd.DataFrame(all_metrics)
    print("Individual Fold Metrics:")
    print(metrics_df)

    # Calculate Average Cumulative Return and Sharpe Ratio
    avg_cumulative_return = metrics_df['cumulative_return'].mean()
    avg_sharpe_ratio = metrics_df['sharpe_ratio'].mean()
    print(f"\nAverage Cumulative Return across all folds: {avg_cumulative_return*100:.2f}%")
    print(f"Average Sharpe Ratio across all folds: {avg_sharpe_ratio:.2f}")

    # Calculate Total Money Gained/Lost
    # Assuming starting capital of 1.0 for each fold, the total money is the sum of final portfolio values
    final_portfolio_values = []
    for fold in range(1, num_folds):
        portfolio_csv = os.path.join(backtest_output_dir, f'fold_{fold}', 'portfolio.csv')
        if os.path.exists(portfolio_csv):
            portfolio_df = pd.read_csv(portfolio_csv, parse_dates=['Date'], index_col='Date')
            final_value = portfolio_df['Portfolio_Value'].iloc[-1]
            final_portfolio_values.append(final_value)
            print(f"Fold {fold}: Final Portfolio Value = {final_value:.2f}")
        else:
            print(f"Fold {fold}: Portfolio CSV not found at '{portfolio_csv}'.")
    
    total_money = sum(final_portfolio_values) - num_folds  # Subtract initial capital for each fold
    print(f"\nTotal Money Gained/Lost from the Model across all folds: ${total_money:.2f}")

    print("=== All Backtests Completed ===")

if __name__ == "__main__":
    main()
