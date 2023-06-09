import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Download historical data
data = yf.download('AAPL','2016-01-01','2023-06-01')

# Use only Close price for prediction
data = data[['Close']]

# Predict for the next 'n' days
forecast_out = 30

# Create another column (the target) shifted 'n' units up
data['Prediction'] = data[['Close']].shift(-forecast_out)

# Create the independent and dependent data sets
X = data.drop('Prediction', axis=1)[:-forecast_out]
y = data['Prediction'][:-forecast_out]

# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the Linear Regression Model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Test the model using score
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

# Set x_forecast equal to the last 'n' rows of the original data set from the Close column
x_forecast = data.drop('Prediction', axis=1)[-forecast_out:]

# Predict the future values
forecast = lr.predict(x_forecast)
print("Forecast for the next {} days:".format(forecast_out))
print(forecast)


import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Download historical data
data = yf.download('AAPL', '2016-01-01', '2023-06-01')

# Use only Close price for prediction
data = data[['Close']]

# Predict for the next 'n' days
forecast_out = 30

# Create another column (the target) shifted 'n' units up
data['Prediction'] = data[['Close']].shift(-forecast_out)

# Create the independent and dependent data sets
X = data.drop('Prediction', axis=1)[:-forecast_out]
y = data['Prediction'][:-forecast_out]

# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the Linear Regression Model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Test the model using score
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

# Set x_forecast equal to the last 'n' rows of the original data set from the Close column
x_forecast = data.drop('Prediction', axis=1)[-forecast_out:]

# Predict the future values
forecast = lr.predict(x_forecast)
print("Forecast for the next {} days:".format(forecast_out))
print(forecast)

# Get today's, tomorrow's, next 3 days', and next 4 days' predictions
today_prediction = forecast[0]
tomorrow_prediction = forecast[1]
next_3_days_predictions = forecast[2:5]  # Next 3 days' predictions (index 2 to 4)
next_4_days_predictions = forecast[2:6]  # Next 4 days' predictions (index 2 to 5)

print("Today's Prediction:", today_prediction)
print("Tomorrow's Prediction:", tomorrow_prediction)
print("Next 3 Days' Predictions:")
print(next_3_days_predictions)
print("Next 4 Days' Predictions:")
print(next_4_days_predictions)