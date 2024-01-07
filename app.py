import yfinance as yf
import pandas as pd
from prophet import Prophet
import streamlit as st
import plotly.graph_objects as go

# Set the ticker symbol and the start and end dates
ticker = str(st.text_input('Stock Ticker'))
start_date = "2017-01-01"
end_date = "2023-12-31"

# Fetch the data
try:
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data available for the specified ticker symbol.")
        st.stop()
except Exception as e:
    st.error("Error occurred while fetching the data: " + str(e))
    st.stop()

# Extract the closing prices
df = data[["Close"]].reset_index()
df.columns = ["ds", "y"]

# Create the Prophet model
model = Prophet()

# Fit the model to the data
model.fit(df)

# Generate future dates
future_dates = model.make_future_dataframe(periods=365)

# Make predictions
forecast = model.predict(future_dates)

# Create a Streamlit app
st.title(f"{ticker} Stock Price Forecast")

# Plot the forecast using Plotly
fig = go.Figure()

# Add the forecasted values
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Forecast',
    line=dict(color='blue')
))

# Add the actual values
fig.add_trace(go.Scatter(
    x=df['ds'],
    y=df['y'],
    mode='markers',
    name='Actual',
    marker=dict(color='red')
))

# Configure the layout
fig.update_layout(
    xaxis=dict(title="Date"),
    yaxis=dict(title="Stock Price"),
    hovermode="x",
    showlegend=True
)

# Display the plot using Streamlit
st.plotly_chart(fig)

# Show additional information
st.title(f"{ticker} Additional Information")

# Display available columns in the data
st.write("Available columns in the data:", data.columns)
