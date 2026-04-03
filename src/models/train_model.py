import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model(file_path):
    print("Loading feature data...")

    df = pd.read_csv(file_path)

    # Features and Target
    X = df[['return', 'ma_5', 'ma_10', 'volatility', 'volume_change']]
    y = df['target']

    # 🧠 Time-based split (VERY IMPORTANT ⚠️)
    train_size = int(len(df) * 0.8)

    X_train = X[:train_size]
    X_test = X[train_size:]

    y_train = y[:train_size]
    y_test = y[train_size:]

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("Model saved to models/model.pkl")

    return model


if __name__ == "__main__":
    train_model("data/processed/featured_data.csv")