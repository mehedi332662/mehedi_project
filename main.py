import quandl
from sklearn.linear_model import LinearRegression
# numpy and pandas will be used for data manipulation
import numpy as np
from numpy import array
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
# matplotlib will be used for visually representing our data
import matplotlib.pyplot as plt

quandl.ApiConfig.api_key = "o9_TqjxrTyz-dUvC3zBx"
data = quandl.get("OPEC/ORB", start_date="1980-01-01", end_date="2020-01-01")

data.head()
#plt.ylabel("Crude Oil Prices: Brent - Europe")
# Setting the size of our graph
#data.Value.plot(figsize=(10, 5))

data['MA3'] = data['Value'].shift(1).rolling(window=3).mean()
data['MA9'] = data['Value'].shift(1).rolling(window=9).mean()

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

#y = np.array(y)
#n_features = 1
#y = y.reshape((y.shape[0], y.shape[1], n_features))
# Getting the head of the data

# Setting the training set to 80% of the data
training = 0.8
t = int(training*len(data))

# Training dataset
X_train = X[:t]
y_train = y[:t]

# Testing dataset
X_test = X[t:]
y_test = y[t:]

# Generate the coefficient and constant for the regression
model = Sequential()
n_steps = 3
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=200, verbose=0)

predicted_price = model.predict(X_test)
predicted_price = pd.DataFrame(predicted_price, index=y_test.index, columns = ['price'])
predicted_price.plot(figsize=(10, 5))
y_test.plot()
plt.legend(['Predicted Price', 'Actual Price'])
plt.ylabel("Crude Oil Prices: Brent - Europe")
plt.show()

# Computing the accuracy of our model
#R_squared_score = linear.score(X[t:],y[t:])*100
#accuracy = ("{0:.2f}".format(R_squared_score))
#print("The model has a " + accuracy + "% accuracy.")
R_squared_score = linear.score(X[t:], y[t:])*100
accuracy = ("{0:.2f}".format(R_squared_score))
print("The model has a " + accuracy + "% accuracy.")
