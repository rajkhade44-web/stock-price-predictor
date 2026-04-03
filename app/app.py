import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/model.pkl")

st.title("📈 Stock Price Direction Predictor")

st.write("Predict whether stock will go UP or DOWN next day")

# Input fields
return_val = st.number_input("Return", value=0.0)
ma_5 = st.number_input("MA 5", value=0.0)
ma_10 = st.number_input("MA 10", value=0.0)
volatility = st.number_input("Volatility", value=0.0)
volume_change = st.number_input("Volume Change", value=0.0)
price_diff = st.number_input("Price Difference", value=0.0)
hl_range = st.number_input("High-Low Range", value=0.0)
lag_1 = st.number_input("Lag 1", value=0.0)
lag_2 = st.number_input("Lag 2", value=0.0)
momentum = st.number_input("Momentum", value=0.0)
price_position = st.number_input("Price Position", value=0.0)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[
        return_val, ma_5, ma_10, volatility, volume_change,
        price_diff, hl_range, lag_1, lag_2,
        momentum, price_position
    ]], columns=[
        'return', 'ma_5', 'ma_10', 'volatility', 'volume_change',
        'price_diff', 'hl_range', 'lag_1', 'lag_2',
        'momentum', 'price_position'
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("📈 Stock will go UP")
    else:
        st.error("📉 Stock will go DOWN")