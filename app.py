import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

# Load the LSTM model
model = load_model('D:\\Stock-Predictions-Model-main\\Stock Predictions Model.keras')

# Define Streamlit app
st.header('Stock Market Analyzer')
st.markdown(
    """
    <style>
    body {
        background-image: url("file:///D:/Stock-Predictions-Model-main/bg1.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# User input for stock symbol
available_stocks = ['TATAMOTORS.NS', 'AAPL', 'GOOG', 'MSFT', 'AMZN']  # Add more stocks as needed
selected_stock = st.selectbox('Select Stock Symbol', available_stocks)

# User input for start and end dates
start_date = st.date_input('Start Date', pd.to_datetime('2021-04-27'), format='YYYY/MM/DD')
end_date = st.date_input('End Date', pd.to_datetime('today'), format='YYYY/MM/DD')

# Fetch data using yfinance
data = yf.download(selected_stock, start=start_date, end=end_date)

# Preprocess data
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Make predictions using the LSTM model
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1 / scaler.scale_

predict = predict * scale
y = y * scale

# Create subplots
fig = make_subplots(rows=3, cols=2, 
                    subplot_titles=('Stock Prices', 'Volume', 'Moving Averages', 'RSI', 'MACD', 'Bollinger Bands'))

# Plot actual stock prices as candlestick chart
fig.add_trace(go.Candlestick(x=data.index,
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'],
                             name='Candlestick'), row=1, col=1)

# Plot volume
fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='blue'), row=1, col=2)

# Calculate moving averages
ma_50_days = data['Close'].rolling(50).mean()
ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()

# Plot Price vs Moving Averages
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'), row=2, col=1)
fig.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50'), row=2, col=1)
fig.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100'), row=2, col=1)
fig.add_trace(go.Scatter(x=data.index, y=ma_200_days, mode='lines', name='MA200'), row=2, col=1)

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi = calculate_rsi(data['Close'])

# Plot RSI
fig.add_trace(go.Scatter(x=data.index, y=rsi, mode='lines', name='RSI'), row=2, col=2)

# Calculate MACD
def calculate_macd(data, short_window=12, long_window=26):
    short_ema = data['Close'].ewm(span=short_window, min_periods=1).mean()
    long_ema = data['Close'].ewm(span=long_window, min_periods=1).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, min_periods=1).mean()
    histogram = macd - signal
    return macd, signal, histogram

macd, signal, histogram = calculate_macd(data)

# Plot MACD
fig.add_trace(go.Scatter(x=data.index, y=macd, mode='lines', name='MACD'), row=3, col=1)
fig.add_trace(go.Scatter(x=data.index, y=signal, mode='lines', name='Signal'), row=3, col=1)
fig.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram', marker_color='#6EB13F'), row=3, col=1)

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

upper_band, lower_band = calculate_bollinger_bands(data)

# Plot Bollinger Bands
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'), row=3, col=2)
fig.add_trace(go.Scatter(x=data.index, y=upper_band, mode='lines', name='Upper Bollinger Band'), row=3, col=2)
fig.add_trace(go.Scatter(x=data.index, y=lower_band, mode='lines', name='Lower Bollinger Band'), row=3, col=2)

# Update layout
fig.update_layout(height=1500, width=1200, showlegend=False, title_text=f'Stock Analysis for {selected_stock}',
                  margin=dict(l=40, r=40, t=100, b=40),
                  paper_bgcolor="LightSteelBlue",
                  plot_bgcolor="#352323")
st.plotly_chart(fig)
