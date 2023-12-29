import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
import webbrowser

df = pd.read_csv("Tomato.csv")

# Ensure a datetime format for the ds column
df['ds'] = pd.to_datetime(df['ds'])

# Adăugați coloane pentru anotimpuri în funcție de luna
df['Winter'] = (df['ds'].dt.month.isin([12, 1, 2])).astype(int)
df['Spring'] = (df['ds'].dt.month.isin([3, 4, 5])).astype(int)
df['Summer'] = (df['ds'].dt.month.isin([6, 7, 8])).astype(int)
df['Fall'] = (df['ds'].dt.month.isin([9, 10, 11])).astype(int)

# We create the train and test data frames
train_size = int(len(df) * 0.99)
train, test = df[:train_size], df[train_size:]

# Definition of Prophet model with the Hyperparameters
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    changepoint_prior_scale=2,
    seasonality_prior_scale=40,
    seasonality_mode='multiplicative',
    interval_width=0.1,
    changepoint_range=0.75,
)

# Creating a new condition for the model
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    model.add_seasonality(name=season, period=365, fourier_order=5, condition_name=season)

# Train the model using the training data
model.fit(train)

# Creating a new dataframe for testing and the next 365 days of prediction
future = model.make_future_dataframe(periods=len(test) + 365, freq='D')

# We add the seasons column to the future datafrane
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    future[season] = (future['ds'].dt.month.isin([12, 1, 2])).astype(int)

# Generating the forecast
forecast = model.predict(future)

# Calculating the RMSE
rmse = mean_squared_error(test['y'], forecast['yhat'][-len(test):], squared=False)
print(f"RMSE: {rmse:.4f}")

# Plot the graphs
trace1 = go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Original Data', line=dict(color='blue'))
trace2 = go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='red'))
trace3 = go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='Forecast Lower Bound', line=dict(color='rgba(0,100,80,0.2)'))
trace4 = go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='Forecast Upper Bound', line=dict(color='rgba(0,100,80,0.2)'))
rmse_trace = go.Scatter(x=forecast['ds'], y=forecast['yhat'] - df['y'], mode='lines', name='RMSE Over Time', line=dict(color='blue'))

data = [trace1, trace2, trace3, trace4, rmse_trace]

layout = go.Layout(title='Original Data and Forecast (yhat)',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Value'))

# Creating a subplot for RMSE graph
fig = make_subplots(rows=2, cols=1, subplot_titles=('Original Data and Forecast', 'RMSE Over Time'))

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=1)
fig.add_trace(trace3, row=1, col=1)
fig.add_trace(trace4, row=1, col=1)
fig.add_trace(rmse_trace, row=2, col=1)

fig.update_layout(layout)

# Save graph as HTML file
fig.write_html("forecast_plot_prophet.html")

# Automatically open the HTML file
webbrowser.open("forecast_plot_prophet.html", new=2)
