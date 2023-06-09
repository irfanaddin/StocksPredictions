import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Download historical data
symbol = 'AAPL'  # Replace with the desired stock symbol
start_date = '2016-01-01'
end_date = '2023-06-01'

data = yf.download(symbol, start_date, end_date)

# Prepare the data for prediction
data = data[['Close']]
forecast_out = 30
data['Prediction'] = data['Close'].shift(-forecast_out)

# Create the independent and dependent data sets
X = data.drop('Prediction', axis=1).iloc[:-forecast_out]
y = data['Prediction'].iloc[:-forecast_out]

# Split the data into training and testing sets
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Create and train the Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Test the model using R-squared score
lr_confidence = lr.score(X_test, y_test)
print("Linear Regression Confidence (R-squared):", lr_confidence)

# Predict the future values
x_forecast = data.drop('Prediction', axis=1).iloc[-forecast_out:]
forecast = lr.predict(x_forecast)
print("Forecast for the next {} days:".format(forecast_out))
print(forecast)

# Get predictions for specific days
prediction_days = {
    'Yesterday': forecast[0],
    'Today': forecast[1],
    'Tomorrow': forecast[2],
    'Next 3 Days': forecast[3:6],
    'Next 4 Days': forecast[3:7]
}

# Print the predictions
for day, prediction in prediction_days.items():
    print(day, "Prediction:", prediction)
