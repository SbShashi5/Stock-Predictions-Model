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
st.header('Stock Market Predictor')

# User input for stock symbol
available_stocks = ['TATAMOTORS.NS', 'AAPL', 'GOOG', 'MSFT', 'AMZN']  # Add more stocks as needed
selected_stock = st.selectbox('Select Stock Symbol', available_stocks)

# User input for start and end dates
start_date = st.date_input('Start Date', pd.to_datetime('today'), format='%d-%m-%Y')
end_date = st.date_input('End Date', pd.to_datetime('today'), format='%d-%m-%Y')

# Fetch data using yfinance
data = yf.download(selected_stock, start=start_date, end=end_date)

# Display stock data
st.subheader('Stock Data')
st.write(data)

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

# Interactive Visualization
# Create dropdown menus for selecting different time periods
available_time_periods = ['1Y', '2Y', '5Y']  # Add more time periods as needed
selected_time_period = st.selectbox('Select Time Period', available_time_periods)

# Fetch data for the selected stock and time period
selected_data = yf.download(selected_stock, start=start_date, end=end_date)

# Plot actual stock prices against predicted prices
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=('Actual vs Predicted Prices', 'Volume'))

fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=predict.flatten(), mode='lines', name='Predicted Price'), row=1, col=1)
fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)

fig.update_layout(title_text=f'Stock Prices for {selected_stock} ({selected_time_period})',
                  xaxis_title='Date', legend=dict(x=0, y=1), height=800)

st.plotly_chart(fig)

# Plot moving averages
ma_50_days = data['Close'].rolling(50).mean()
ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()

fig_moving_avg = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                              subplot_titles=('Price vs MA50', 'Price vs MA50 vs MA100', 'Price vs MA100 vs MA200'))

fig_moving_avg.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'), row=1, col=1)
fig_moving_avg.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50'), row=1, col=1)

fig_moving_avg.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'), row=2, col=1)
fig_moving_avg.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50'), row=2, col=1)
fig_moving_avg.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100'), row=2, col=1)

fig_moving_avg.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'), row=3, col=1)


fig_moving_avg.update_layout(title_text=f'Moving Averages for {selected_stock} ({selected_time_period})',
                             xaxis_title='Date', height=800)

st.plotly_chart(fig_moving_avg)

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
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data.index, y=rsi, mode='lines', name='RSI'))
fig_rsi.update_layout(title_text=f'Relative Strength Index (RSI) for {selected_stock} ({selected_time_period})',
                      xaxis_title='Date', yaxis_title='RSI', legend=dict(x=0, y=1))
st.plotly_chart(fig_rsi)

# Calculate MACD
def calculate_macd(data, short_window=12, long_window=26):
    short_ema = data['Close'].ewm(span=short_window, min_periods=1).mean()
    long_ema = data['Close'].ewm(span=long_window, min_periods=1).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, min_periods=1).mean()
    return macd, signal

macd, signal = calculate_macd(data)

# Plot MACD
fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                         subplot_titles=('MACD Line', 'Signal Line'))

fig_macd.add_trace(go.Scatter(x=data.index, y=macd, mode='lines', name='MACD'), row=1, col=1)
fig_macd.add_trace(go.Scatter(x=data.index, y=signal, mode='lines', name='Signal'), row=2, col=1)

fig_macd.update_layout(title_text=f'MACD for {selected_stock} ({selected_time_period})',
                       xaxis_title='Date', height=800)

st.plotly_chart(fig_macd)

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

upper_band, lower_band = calculate_bollinger_bands(data)

# Plot Bollinger Bands
fig_bb = go.Figure()
fig_bb.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))
fig_bb.add_trace(go.Scatter(x=data.index, y=upper_band, mode='lines', name='Upper Bollinger Band'))
fig_bb.add_trace(go.Scatter(x=data.index, y=lower_band, mode='lines', name='Lower Bollinger Band'))

fig_bb.update_layout(title_text=f'Bollinger Bands for {selected_stock} ({selected_time_period})',
                     xaxis_title='Date', yaxis_title='Price', legend=dict(x=0, y=1))
st.plotly_chart(fig_bb)
