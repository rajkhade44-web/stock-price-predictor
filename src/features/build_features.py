import pandas as pd
import os

def build_features(file_path):
    print("Loading cleaned data...")

    df = pd.read_csv(file_path)

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort again (safety)
    df = df.sort_values(by='Date')

    # 🔥 Feature 1: Daily Return
    df['return'] = df['Close'].pct_change()

    # 🔥 Feature 2: Moving Averages
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()

    # 🔥 Feature 3: Volatility
    df['volatility'] = df['Close'].rolling(window=5).std()

    # 🔥 Feature 4: Volume Change
    df['volume_change'] = df['Volume'].pct_change()

    # 🎯 Target Variable (MOST IMPORTANT)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop rows with NaN (due to rolling)
    df = df.dropna()

    # Save features
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/featured_data.csv"
    df.to_csv(output_path, index=False)

    print("Feature engineering complete!")
    print(df.head())

    return df


if __name__ == "__main__":
    build_features("data/processed/cleaned_data.csv")