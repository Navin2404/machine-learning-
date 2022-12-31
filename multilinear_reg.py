#multiple linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('50_startups.csv')
x=dataset.iloc[:,0:3].values
y=dataset.iloc[:,-1].values

#splitting the dataset in to the training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

#Training the multiple linear Regression model on the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test set results

#y=26777.3913412+(9360.26128619*5.3)
y_pred=regressor.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

####mean squared error(MSE)####
###mean sqaured error represents the average of the squared difference between
#the original and predicted values in the dataset.
#It measures the varience of the residuals.
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)

####Root mean squared Error (RMSE)####
#Root mean squared error is the sqaure root of mean squared error.
#It measures the standard deviation of residuals.
np.sqrt(mean_squared_error(y_test,y_pred))

####Mean Absolute Error (MAE)####
#The mean absolute error represents the average of the absolute difference between
#the actual value and predicted value in the dataset.
#It measures the average of the residuals in the dataset.
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_pred)

#calculation of Explained varience
from sklearn.model_selection import cross_val_score
print(cross_val_score(regressor,x,y,cv=10,scoring="explained_variance").mean())

#calculation of Bias and varience
from mlxtend.evaluate import bias_varience_decomp
mse, bias, varience =bias_varience_decomp (regressor,x_train,y_train, x_test, y_test,
                                           loss='mse',nu_rounds=200, random_seed=123)




