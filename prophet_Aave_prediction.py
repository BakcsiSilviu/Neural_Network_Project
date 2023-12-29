import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error
from plotly.subplots import make_subplots
import webbrowser

# Load the original DataFrame
df = pd.read_csv("coin_Aave.csv")
new_df = df[['Date', 'Open']]

# Rename 'Open' column to 'y' for Prophet
new_df = new_df.rename(columns={'Open': 'y'})

# Ensure 'ds' is in datetime format
new_df['ds'] = pd.to_datetime(new_df['Date'])

# Create and fit the model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    changepoint_prior_scale=0.01,
    seasonality_prior_scale=10,
    seasonality_mode='additive',
    interval_width=0.6,
)
model.fit(new_df)

# Create a dataframe with future dates for prediction
future = model.make_future_dataframe(periods=300, freq='D')

# Make predictions
forecast = model.predict(future)

# Calculate RMSE
rmse = mean_squared_error(new_df['y'], forecast['yhat'][:len(new_df)], squared=False)

# Plot the original data and the forecast using Plotly
trace1 = go.Scatter(x=new_df['ds'], y=new_df['y'], mode='lines', name='Original Data')
trace2 = go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast')
trace3 = go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='Forecast Lower Bound', line=dict(color='rgba(0,100,80,0.2)'))
trace4 = go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='Forecast Upper Bound', line=dict(color='rgba(0,100,80,0.2)'))

data = [trace1, trace2, trace3, trace4]

layout = dict(title='Aave Price Forecast',
              xaxis=dict(title='Date'),
              yaxis=dict(title='Aave Price'),
              )

# Add RMSE subplot
fig = make_subplots(rows=2, cols=1, subplot_titles=('Aave Price Forecast', 'RMSE Over Time'))

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=1)
fig.add_trace(trace3, row=1, col=1)
fig.add_trace(trace4, row=1, col=1)

fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'] - new_df['y'], mode='lines', name='RMSE Over Time', line=dict(color='blue')), row=2, col=1)

fig.update_layout(layout)

# Save the plot as an HTML file
fig.write_html("forecast_plot.html")

# Open the HTML file in the web browser
webbrowser.open("forecast_plot.html", new=2)
