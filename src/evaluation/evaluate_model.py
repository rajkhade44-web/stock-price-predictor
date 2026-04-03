import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate():
    print("Loading model and data...")

    model = joblib.load("models/model.pkl")
    df = pd.read_csv("data/processed/featured_data.csv")

    X = df[['return', 'ma_5', 'ma_10', 'volatility', 'volume_change',
        'price_diff', 'hl_range', 'lag_1', 'lag_2',
        'momentum', 'price_position']]

    y = df['target']

    train_size = int(len(df) * 0.8)

    X_test = X[train_size:]
    y_test = y[train_size:]

    y_pred = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate()