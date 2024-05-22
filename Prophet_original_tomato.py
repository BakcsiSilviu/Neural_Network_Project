import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import webbrowser

# Load your dataset from the CSV file
df = pd.read_csv('Tomato_average.csv')

# Prophet requires the columns to be named 'ds' and 'y'
df_prophet = df.rename(columns={'Date': 'ds', 'Average': 'y'})

# Convert the 'ds' column to datetime format
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

# Split the data into training and testing sets
train_size = int(len(df_prophet) * 2 / 3)
train_data = df_prophet[:train_size]
test_data = df_prophet[train_size:]

# Initialize and train the Prophet model
model = Prophet()
model.fit(train_data)

# Make future dataframe for the length of the test set
future = model.make_future_dataframe(periods=len(test_data))

# Predict
forecast = model.predict(future)

# Calculate RMSE for each point in the test set
test_forecast = forecast[-len(test_data):]
rmse_values = np.sqrt((test_data['y'].values - test_forecast['yhat'].values) ** 2)

# Plotly traces
trace1 = go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Original Data')
trace2 = go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Data',
                    line=dict(color='orange'))
trace3 = go.Scatter(x=train_data['ds'], y=train_data['y'], mode='lines', name='Training Data', line=dict(color='green'))
trace4 = go.Scatter(x=test_data['ds'], y=test_data['y'], mode='lines', name='Test Data', line=dict(color='red'))

# RMSE Trace
rmse_trace = go.Scatter(x=test_data['ds'], y=rmse_values, mode='lines', name='RMSE', line=dict(color='blue'))

# Creating a subplot for RMSE graph
fig = make_subplots(rows=2, cols=1, subplot_titles=('Original Data and Forecast', 'RMSE Over Time'))

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=1)
fig.add_trace(trace3, row=1, col=1)
fig.add_trace(trace4, row=1, col=1)
fig.add_trace(rmse_trace, row=2, col=1)

# Update layout
layout = {
    'title': 'Tomato Price Prediction',
    'xaxis_title': 'Date',
    'yaxis_title': 'Average Price',
    'height': 800
}
fig.update_layout(layout)

# Save graph as HTML file
fig.write_html("forecast_plot_prophet.html")

# Automatically open the HTML file
webbrowser.open("forecast_plot_prophet.html", new=2)
