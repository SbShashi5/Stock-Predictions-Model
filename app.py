import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

side_bg = 'D:\\Stock-Predictions-Model-main\\bg1.jpg'
sidebar_bg(side_bg)

def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
   
# Load the LSTM model
model = load_model('D:\\Stock-Predictions-Model-main\\Stock Predictions Model.keras')

# Define Streamlit app
st.header('Stock Market Analyzer')

# User input for stock symbol
available_stocks = ['TATAMOTORS.NS', 'AAPL', 'GOOG', 'MSFT', 'AMZN']  # Add more stocks as needed
selected_stock = st.selectbox('Select Stock Symbol', available_stocks)

# User input for start and end dates
start_date = st.date_input('Start Date', pd.to_datetime('2021-04-27'), format='YYYY/MM/DD')
end_date = st.date_input('End Date', pd.to_datetime('today'), format='YYYY/MM/DD')


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

# Plot actual stock prices as candlestick chart
fig_candlestick = go.Figure()
fig_candlestick.add_trace(go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         name='Candlestick'))
fig_candlestick.update_layout(title_text=f'Candlestick Chart for {selected_stock} ({start_date.strftime("%Y/%m/%d")} to {end_date.strftime("%Y/%m/%d")})',
                              xaxis_title='Date', yaxis_title='Price', legend=dict(x=0, y=1))
st.plotly_chart(fig_candlestick)

# Plot volume
fig_volume = go.Figure()
fig_volume.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='blue'))
fig_volume.update_layout(title_text=f'Volume for {selected_stock} ({start_date.strftime("%Y/%m/%d")} to {end_date.strftime("%Y/%m/%d")})',
                         xaxis_title='Date', yaxis_title='Volume', legend=dict(x=0, y=1))
st.plotly_chart(fig_volume)





# Calculate moving averages
ma_50_days = data['Close'].rolling(50).mean()
ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()

# Plot Price vs Moving Averages
fig_ma = go.Figure()

fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))
fig_ma.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50'))
fig_ma.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100'))
fig_ma.add_trace(go.Scatter(x=data.index, y=ma_200_days, mode='lines', name='MA200'))

fig_ma.update_layout(title_text=f'Price vs Moving Averages for {selected_stock} ({start_date.strftime("%Y/%m/%d")} to {end_date.strftime("%Y/%m/%d")})',
                     xaxis_title='Date', yaxis_title='Price', legend=dict(x=0, y=1), height=600)

st.plotly_chart(fig_ma)

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
fig_rsi.update_layout(title_text=f'Relative Strength Index (RSI) for {selected_stock} ({start_date.strftime("YYYY/MM/DD")} to {end_date.strftime("YYYY/MM/DD")})',
                      xaxis_title='Date', yaxis_title='RSI', legend=dict(x=0, y=1))
st.plotly_chart(fig_rsi)

# Plot MACD
def calculate_macd(data, short_window=12, long_window=26):
    short_ema = data['Close'].ewm(span=short_window, min_periods=1).mean()
    long_ema = data['Close'].ewm(span=long_window, min_periods=1).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, min_periods=1).mean()
    histogram = macd - signal
    return macd, signal, histogram

macd, signal, histogram = calculate_macd(data)

# Plot MACD
fig_macd = go.Figure()

fig_macd.add_trace(go.Scatter(x=data.index, y=macd, mode='lines', name='MACD'))
fig_macd.add_trace(go.Scatter(x=data.index, y=signal, mode='lines', name='Signal'))
fig_macd.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram', marker_color='rgba(0, 128, 0, 0.5)'))

fig_macd.update_layout(title_text=f'MACD for {selected_stock} ({start_date.strftime("%Y/%m/%d")} to {end_date.strftime("%Y/%m/%d")})',
                       xaxis_title='Date', yaxis_title='MACD', legend=dict(x=0, y=1), height=600)

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

fig_bb.update_layout(title_text=f'Bollinger Bands for {selected_stock} ({start_date.strftime("YYYY/MM/DD")} to {end_date.strftime("YYYY/MM/DD")})',
                     xaxis_title='Date', yaxis_title='Price', legend=dict(x=0, y=1))
st.plotly_chart(fig_bb)
