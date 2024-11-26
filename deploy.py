import streamlit as st
import pandas as pd
import pickle
from datetime import timedelta

# Load the model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Apple Stock Price Prediction")

# Specify the file path of your CSV data directly here
file_path = 'C:/Users/vedan/P452/apple_stock_data.csv'  # Change this to your actual file path

# Read the CSV file
data = pd.read_csv(file_path)

# Display the data and its columns for debugging
st.write("Uploaded Data:")
st.write(data)


# Check if 'ds' column exists before proceeding
if 'ds' in data.columns:
    # Convert 'ds' column to datetime
    data['Date'] = pd.to_datetime(data['ds'])
    
    # Proceed with feature extraction
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day

    # Define feature set based on the order the model expects
    # Ensure the features are in the same order as used in model training
    features = data[['Open', 'High', 'Low', 'Year', 'Month', 'Day']]

    # Predict the next 10 days
    future_dates = pd.date_range(start=data['Date'].max() + timedelta(days=1), periods=10, freq='D')
    future_data = pd.DataFrame({
        'Year': future_dates.year,
        'Month': future_dates.month,
        'Day': future_dates.day,
        'Open': data['Open'].iloc[-1],  # Use the last known value for Open
        'High': data['High'].iloc[-1],   # Use the last known value for High
        'Low': data['Low'].iloc[-1]       # Use the last known value for Low
    })

    # Ensure the feature order matches the training set
    future_data = future_data[['Open', 'High', 'Low', 'Year', 'Month', 'Day']]

    # Predict
    predictions = model.predict(future_data)

    # Display predictions
    predicted_prices = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predictions
    })

    st.write("Predicted Closing Prices for the Next 10 Days:")
    st.write(predicted_prices)
else:
    st.error("The uploaded file does not contain a 'ds' column.")
