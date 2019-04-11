# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:21:39 2019

@author: William Wei
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import joblib

pd.options.display.float_format = '{:.2f}'.format
sns.set(rc={'figure.figsize':(30, 20)})

aapl = pd.read_csv('AAPL.csv')

# data understanding
aapl.head()
aapl.info()
aapl.describe()
aapl.isnull().sum() # only 1 is null
aapl[aapl.Open.isnull()] # the null value is on 1981-08-10

# clean data, fill the null value with last date price
aapl.fillna(method='ffill', inplace=True)
aapl.set_index('Date', inplace=True)
aapl.index = pd.to_datetime(aapl.index)
aapl.Close.plot(title='Apple Inc. Historical Stock Price')

# the data is from 1980 to 2019
split_yr = 2007
#split_yr = 2011
#split_yr = 2015
training = aapl[:str(split_yr-1)][['Close']]
testing = aapl[str(split_yr):][['Close']]

scl = MinMaxScaler()
training_scl = scl.fit_transform(training)
testing_scl = scl.transform(testing)

# create time series data, use previous t-30 d to predict t time stock price
ts = 30
X_train = []
y_train = []
for i in range(ts, training_scl.shape[0]):
    X_train.append(training_scl[i - ts:i, 0])
    y_train.append(training_scl[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_test = []
for i in range(ts, testing_scl.shape[0]):
    X_test.append(testing_scl[i - ts:i, 0])
X_test = np.array(X_test)
y_test = testing[ts:] # still a dataframe, containing original data

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# create a single hidden layer LSTM and stacked LSTM separately
def create_simple_model():
    model = Sequential()
    model.add(LSTM(units = 50, input_shape = (X_train.shape[1], 1)))
    model.add(Dense(units = 1))
    return model

def create_stacked_model():
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True,
                   input_shape = (X_train.shape[1], 1)))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(LSTM(units = 50))
    model.add(Dense(units = 1))
    return model

def compile_and_run(model, epochs=50, batch_size=32):
    model.compile(metrics=['accuracy'], optimizer='adam',
                  loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, verbose=3)
    return history

def plot_metrics(history):
    metrics_df = pd.DataFrame(data={"loss": history.history['loss']})
    metrics_df.plot()

def make_predictions(X_test, layer='Simple'):
    if layer == 'Simple':
        model = simple_model
    elif layer == 'Stacked':
        model = stacked_model

    y_pred = scl.inverse_transform(model.predict(X_test))
    prediction = np.ndarray.flatten(y_pred)
    actual = np.ndarray.flatten(y_test.values)
    mse = mean_squared_error(actual, prediction)
    r2 = r2_score(actual, prediction)

    plt.plot(actual, color = 'red', label = 'Actual Stock Price')
    plt.plot(prediction, color = 'blue', label = 'Predicted Stock Price')
    plt.ylabel('Price')
    plt.figtext(0.15, 0.8, 'mse: %.2f \nr_square: %.2f'%(mse, r2))
    plt.title('%s LSTM model'%layer)
    plt.legend()
    plt.savefig('./images/%s_LSTM_model_%s.png'%(layer, split_yr))

# prediction of single layer model
simple_model = create_simple_model()
history = compile_and_run(simple_model)
plot_metrics(history)
make_predictions(X_test, 'Simple')

# prediction of multi-layers model
stacked_model = create_stacked_model()
history = compile_and_run(stacked_model)
plot_metrics(history)
make_predictions(X_test, 'Stacked')

joblib.dump(simple_model, 'simple_model_%s.joblib'%split_yr)
joblib.dump(stacked_model, 'stacked_model_%s.joblib'%split_yr)
