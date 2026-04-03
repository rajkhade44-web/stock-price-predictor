import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker="AAPL", start="2015-01-01", end="2024-01-01"):
    print(f"Fetching data for {ticker}...")

    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        print("No data fetched!")
        return

    # Reset index to make Date a column
    df.reset_index(inplace=True)

    # Create directory if not exists
    os.makedirs("data/raw", exist_ok=True)

    file_path = f"data/raw/{ticker}.csv"
    df.to_csv(file_path, index=False)

    print(f"Data saved to {file_path}")
    print(df.head())


if __name__ == "__main__":
    fetch_stock_data("AAPL")