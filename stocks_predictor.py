import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Download historical data
symbol = 'SNAP'  
start_date = '2016-01-01'
end_date = '2023-06-01'

data = yf.download(symbol, start_date, end_date)

# Define how many days ahead you want to predict
forecast_out = 30

# Prepare the data for prediction
data['Prediction'] = data['Close'].shift(-forecast_out)
data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Prediction']]

# TODO: add technical indicators here

# Create the independent and dependent data sets
X = data.drop('Prediction', axis=1).iloc[:-forecast_out]
y = data['Prediction'].iloc[:-forecast_out]

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Create the Random Forest Regressor
rf = RandomForestRegressor()

# Define hyperparameters grid for GridSearch
# Define hyperparameters grid for GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': [1.0, 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8]
}

# Apply GridSearchCV for hyperparameter tuning
CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
CV_rf.fit(X, y)

print("Best Parameters: ", CV_rf.best_params_)

# Predict the future values
x_forecast = scaler.transform(data.drop('Prediction', axis=1).iloc[-forecast_out:])
forecast = CV_rf.predict(x_forecast)
print("Forecast for the next {} days:".format(forecast_out))
print(forecast)

# Print out performance metrics
y_pred = CV_rf.predict(X)
print('Mean Squared Error:', mean_squared_error(y, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y, y_pred)))
print('Mean Absolute Error:', mean_absolute_error(y, y_pred))







# Get predictions for specific days
prediction_days = {
    'Yesterday': forecast[0],
    'Today': forecast[1],
    'Tomorrow': forecast[2],
    'Next 3 Days': forecast[2:4],
    'Next 4 Days': forecast[2:5]
}

# Print the predictions
for day, prediction in prediction_days.items():
    print(day, "Prediction:", prediction)
