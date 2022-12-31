# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:01:10 2022

@author: HAI
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset=pd.read_csv('D:\dataset\height_weight.csv')

dataset

x=dataset.iloc[:,[0]].values
y=dataset.iloc[:,[-1]].values

#split the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=0)

#Training the simple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

regressor.intercept_

regressor.coef_

y=26777.3913412+(9360.26128619*5.3)

#predicting the test results
y_pred=regressor.predict(x_test)
regressor.score(x_train,y_train)

# visualizing the training set results
plt.figure(dpi=300)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

# visualizing the training set results
plt.figure(dpi=300)
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regressor.predict(x_test),color='blue')
plt.title('salary vs experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#new observation data
regressor.predict([[1.5]])

dat={"yearsexperience:":[1.2,1.4,1.8]}

d=pd.DataFrame(dat)

regressor.predict(d)

#Linear regression performance calculation

from sklearn.metrics import r2score

r2_score(y_test,y_pred)









