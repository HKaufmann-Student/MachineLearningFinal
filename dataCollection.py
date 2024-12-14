import yfinance as yf
import pandas as pd
import os

def download_data(ticker, start, end):
    """
    Downloads historical stock data for a given ticker symbol and date range.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'GOOG').
        start (str): The start date in the format 'YYYY-MM-DD'.
        end (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: A DataFrame containing the stock's historical data with the ticker added as a column.
    """
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df['Ticker'] = ticker
    return df

def download_stock_data(tickers, start_date, end_date):
    """
    Downloads historical data for multiple stock tickers.

    Args:
        tickers (list): A list of stock ticker symbols.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: A combined DataFrame containing data for all specified tickers.
    """
    stock_data_list = []  # Initialize an empty list to hold individual DataFrames
    for ticker in tickers:
        # Download data for each ticker and append it to the list
        data = download_data(ticker, start_date, end_date)
        stock_data_list.append(data)
    # Concatenate all DataFrames into a single DataFrame
    all_data = pd.concat(stock_data_list, axis=0, ignore_index=True)
    return all_data

def download_macro_data(start_date, end_date):
    """
    Downloads macroeconomic data: S&P 500 (SPY) and VIX (Volatility Index).

    Args:
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: A DataFrame containing the SPY and VIX data, merged by date.
    """
    # Download SPY (S&P 500 proxy) data and keep the 'Date' and 'Close' columns
    spy = download_data('SPY', start_date, end_date)[['Date', 'Close']].copy()
    spy.rename(columns={'Close': 'SPY_Close'}, inplace=True)
    
    # Download VIX data and keep the 'Date' and 'Close' columns
    vix = download_data('^VIX', start_date, end_date)[['Date', 'Close']].copy()
    vix.rename(columns={'Close': 'VIX_Close'}, inplace=True)
    
    # Merge SPY and VIX data on the 'Date' column, performing an outer join to include all dates
    macro_df = pd.merge(spy, vix, on='Date', how='outer')
    # Sort the data by date for consistency
    macro_df.sort_values('Date', inplace=True)
    
    # Fill any missing values by forward-filling and backward-filling
    macro_df.ffill(axis=0, inplace=True)
    macro_df.bfill(axis=0, inplace=True)
    
    return macro_df

def save_raw_data(stock_df, macro_df, output_dir='data'):
    """
    Saves the stock and macroeconomic data to CSV files in the specified directory.

    Args:
        stock_df (pd.DataFrame): DataFrame containing stock data.
        macro_df (pd.DataFrame): DataFrame containing macroeconomic data.
        output_dir (str): Directory where the files will be saved. Default is 'data'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stock_df.to_csv(os.path.join(output_dir, 'raw_stock_data.csv'), index=False)
    macro_df.to_csv(os.path.join(output_dir, 'raw_macro_data.csv'), index=False)
