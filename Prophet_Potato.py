# Importing necessary libraries
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Sample time series data (replace this with your own dataset)
data = pd.read_csv('Daily_Retail_Price_of_Potato_prophet.csv')
# Ensure your dataset has columns named 'ds' for dates and 'y' for values

# Creating a Prophet model
model = Prophet()

# Fit the model to the data
model.fit(data)

# Making predictions for future dates
future_dates = model.make_future_dataframe(periods=360)  # Adjust periods as needed
forecast = model.predict(future_dates)

# Plotting the forecast
fig = model.plot(forecast, uncertainty=True)  # Use uncertainty=True to show uncertainty interval
plt.title('Forecast')
plt.xlabel('Date')
plt.ylabel('Value')

# Adding legend
plt.legend(['Observed', 'Trend', 'Uncertainty Interval'], loc='upper left')

plt.show()
