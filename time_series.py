import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.layers import LSTM, Dense
from keras.models import Sequential

file_path = 'data/international-airline-passengers.csv'
test_ratio = 0.1

dataset = pd.read_csv(file_path, usecols=[1])

dataset = dataset.values
dataset = dataset.astype('float32')

# normalise dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

def create_dataset(dataset):
    x, y = [], []
    for index, value in enumerate(dataset):
        try:
            prediction = dataset[index+1]
        except IndexError:
            break
        x.append(value)
        y.append(prediction)
    return np.array(x), np.array(y)

train, test = train_test_split(dataset, test_size=test_ratio, shuffle=False)

x_train, y_train = create_dataset(train)
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

x_test, y_test = create_dataset(test)
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=x_train[0].shape))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=30, batch_size=1, validation_split=0.05)

# make predictions
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

plt.plot(train_predict, label='train_predict')
plt.plot(scaler.inverse_transform(x_train.reshape(x_train.shape[0],
    x_train.shape[1])), label='x_train')
plt.plot(test_predict, label='test_predict')
plt.plot(scaler.inverse_transform(x_test.reshape(x_test.shape[0],
    x_test.shape[1])), label='x_test')
plt.legend()
plt.show()
