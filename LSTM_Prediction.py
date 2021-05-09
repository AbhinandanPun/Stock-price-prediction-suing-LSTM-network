import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import investpy
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Downloading the stock History
df = investpy.get_index_historical_data(index="Nifty 50",
                                        country="India",
                                        from_date='01/01/2010',
                                        to_date='12/01/2020')
df = df.sort_values('Date')     
cdf = df.reset_index()                                     # convert date index into column
cdf = cdf[['Date','Close']]                                # extracting the data


# Visualization of Data
plt.figure(figsize = (50,7))
plt.plot(cdf.Date, cdf.Close, color='blue')
ax = plt.gca()                                                  # get current axes
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))     # set monthly locator of X - labels
plt.gcf().autofmt_xdate()                                       # Rotation of X-labels
plt.xlabel("Date",fontsize=15)                                  
plt.ylabel("Close",fontsize=15)
plt.title('Stock Price Prediction - REAL STOCK PRICE')
plt.show()                    


# Train / Test Splitting
dataset_train = cdf.iloc[:1500, 1:2]                            
dataset_test = cdf.iloc[1500:, 1:2]


# Normalize the Training Set
sc = MinMaxScaler(feature_range = (0, 1))                       
training_set_scaled = sc.fit_transform(dataset_train)


# Preprocessing the Training Set
X_train = []                                                    
y_train = []
for i in range(60, 1500):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# LSTM model for prediction
model = Sequential()                                            
model.add(LSTM(units = 50,                                      # Adding the first LSTM layer and some Dropout regularisation
               return_sequences = True,
               input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))            # Adding a second LSTM layer
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))            # Adding a third LSTM layer
model.add(Dropout(0.2))
model.add(LSTM(units = 50))                                     # Adding a third LSTM layer
model.add(Dropout(0.2))
model.add(Dense(units = 1))                                     # Adding the output layer
model.compile(optimizer = 'adam', loss = 'mean_squared_error')  # Compiling the Recurrent Neural Network(RNN)
model.fit(X_train, y_train, epochs = 100, batch_size = 32)      # Fitting the RNN to the Training set


# Test Data Preprocessing
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)   
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 1045):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Stock Price Prediction
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualisation of results
plt.figure(figsize = (50,7))
plt.plot(cdf.loc[1500:, 'Date'],dataset_test.values, color = 'red', label = 'Real Stock Price')
plt.plot(cdf.loc[1500:, 'Date'],predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
ax = plt.gca()                                               # get current axes
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # set monthly locator of X-labels
plt.gcf().autofmt_xdate()                                    # Rotation of X-labels
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()