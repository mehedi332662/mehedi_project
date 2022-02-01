# Quandl will be used for importing historical oil prices
import quandl
# from sklearn.linear_model import LSTM
# numpy and pandas will be used for data manipulation
# matplotlib will be used for visually representing our data
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt

# To import data set using API
quandl.ApiConfig.api_key = "yyKLQLy8-L34fCsi8pz3"
data = quandl.get("OPEC/ORB", start_date="2018-01-01", end_date="2022-01-01")
data.head()

# Setting the size of our graph
plt.ylabel("Petrol  Prices Europe")
data.Value.plot(figsize=(10, 5))

#moving averages for the past three and nine days.
data['MA3'] = data['Value'].shift(1).rolling(window=3).mean()
data['MA9'] = data['Value'].shift(1).rolling(window=9).mean()

# Dropping the NaN values
data = data.dropna()
# Initialising X and assigning the two feature variables
X = data[['MA3', 'MA9']]
X.head()

X = np.array(X)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# Getting the head of the data

y = data['Value']
y.head()

# y = np.array(y)
# n_features = 1
# y = y.reshape((y.shape[0], y.shape[1], n_features))
# Getting the head of the data

# Setting the training set to 80% of the data
training = 0.8
t = int(training * len(data))

# Training dataset
X_train = X[:t]
y_train = y[:t]

# Testing dataset
X_test = X[t:]
y_test = y[t:]

# Generate the coefficient and constant for the regression
model = Sequential()
n_steps = 2
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=200, verbose=0)

predicted_price = model.predict(X_test)
predicted_price = pd.DataFrame(predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 5))
y_test.plot()
plt.legend(['Predicted Price with LSTM', 'Actual Price '])
plt.ylabel("Petrol  Prices Europe ")
plt.show()


#the model has a 97% accuracy
R_squared_score = model.score(X[t:], y[t:]) * 100
accuracy = ("{0:.2f}".format(R_squared_score))
print("The model has a " + accuracy + "% accuracy.")

