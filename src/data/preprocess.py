import pandas as pd
import os

def preprocess_data(file_path):
    print("Loading data...")

    # Load Excel file
    df = pd.read_excel(file_path)

    print("Original Data:")
    print(df.head())

    # Standardize column names (IMPORTANT)
    df.columns = [col.strip().capitalize() for col in df.columns]

    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date (VERY IMPORTANT ⚠️)
    df = df.sort_values(by='Date')

    # Handle missing values
    df = df.dropna()

    # Reset index
    df = df.reset_index(drop=True)

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/cleaned_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Cleaned data saved to {output_path}")
    print(df.head())

    print(df.info())
    print(df.describe())

    return df


if __name__ == "__main__":
    preprocess_data("data/raw/apple_stock_10years_daywise.xlsx")

