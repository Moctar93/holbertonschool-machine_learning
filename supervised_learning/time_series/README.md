Bitcoin Price Forecast with RNN (LSTM)
This project uses a Recurrent Neural Network (RNN) with LSTM layers to forecast the next-minute price of Bitcoin (BTC) based on the previous 24 hours of historical data.

Objective
Predict the next minute’s Volume Weighted Average Price (VWAP) of Bitcoin using a sequence of the previous 1440 minutes (24 hours) of price data from Coinbase and Bitstamp.

Project Structure
bash
Copier
Modifier
.
├── coinbase.csv               # Coinbase historical data
├── bitstamp.csv               # Bitstamp historical data
├── preprocess_data.py         # Data preprocessing script
├── forecast_btc.py            # RNN model training and evaluation
├── btc_dataset.npz            # Preprocessed dataset (X, y)
├── btc_forecast_model.h5      # Trained model (HDF5 format)
└── README.md                  # This file
Requirements
Install required packages:

bash
Copier
Modifier
pip install pandas numpy scikit-learn tensorflow
How to Use
1. Preprocess the Data
This script:

Merges Coinbase and Bitstamp data

Cleans and normalizes the VWAP values

Generates 24-hour input sequences (1440 minutes)

Saves the training dataset as a .npz file

bash
Copier
Modifier
python preprocess_data.py
2. Train the Forecasting Model
This script:

Loads the preprocessed sequences

Builds and trains an LSTM-based model

Evaluates performance using MSE

Saves the trained model to disk

bash
Copier
Modifier
python forecast_btc.py
Model Overview
Input shape: (1440 time steps, 1 feature)

Output: 1 value (next-minute VWAP)

Model architecture: LSTM (64 units) + Dense(1)

Loss function: Mean Squared Error (MSE)

Data scaling: MinMaxScaler (values normalized between 0 and 1)

Framework: TensorFlow / Keras

Example Prediction Task
text
Copier
Modifier
Input: [price_1, price_2, ..., price_1440] → Output: price_1441
Possible Improvements
Include more features (e.g., volume, volatility)

Predict multiple future steps (e.g., next 5 or 10 minutes)

Add visualization of predictions

Deploy as an API or real-time forecast tool

Author
This project is part of an educational exercise in applying time series forecasting using deep learning with TensorFlow/Keras.


