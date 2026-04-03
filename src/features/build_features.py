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

    # Price difference
    df['price_diff'] = df['Close'] - df['Open']

    # High-Low range
    df['hl_range'] = df['High'] - df['Low']

    # Lag features (VERY IMPORTANT)
    df['lag_1'] = df['Close'].shift(1)
    df['lag_2'] = df['Close'].shift(2)

    # 🎯 Target Variable (MOST IMPORTANT)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Momentum
    df['momentum'] = df['Close'] - df['Close'].shift(5)

    # Rolling max/min
    df['rolling_max'] = df['Close'].rolling(5).max()
    df['rolling_min'] = df['Close'].rolling(5).min()

    # Price position
    df['price_position'] = (df['Close'] - df['rolling_min']) / (df['rolling_max'] - df['rolling_min'])

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