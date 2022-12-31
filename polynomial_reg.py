#polynomial regresiion 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('position_salaries.csv')
x=dataset.iloc[:,[1]].values
y=dataset.iloc[:,-1].values

plt.scatter(x,y)

#Training the dataset in liear regression model
#from sklearn.linear_model import LinearRegression
#LR=LinearRegression()
#LR.fit(x,y)

#Linear regression prediction 
#LR.predict([[6.5]])

#let check the plot for liear regression
#plt.scatter(x,y, color='red')
#plt.plot(x, LR.predict(x), color='blue')
#plt.show()

#Training the whole dataset polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=5)
x_poly=poly_reg.fit_transform(x)

from sklearn.linear_model import LinearRegression
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)
 
#lin_reg_2.predict([[6.5]])  #it shows error

#polynomial regression prediction
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

#let visualize the polynomial regression
plt.scatter(x,y, color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.show()
