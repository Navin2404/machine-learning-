#Logistic regression

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('social_Network_Ads.csv')
x= dataset.iloc[:,[2, 3]].values
y= dataset.iloc[:, [-1]].values

#Feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
y = sc.transform(y)

#splitting the dataset in to Train set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

#Training the Logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(x_test)

classifier.score(x_train,y_train) # training score

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

acc=(sum(np.diag(cm))/len(y_test))
acc

#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#another method to find the accuracy
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)

#visualizing the training set results
from matplotlib.colors import ListedColormap
#it helps to separate the colours use of map
x_set, y_set, = x_train, y_train

x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() -1,
                               stop= x_set[:, 0].max() +1,
                               step=0.001),
                     np.arange(start=x_set[:, 1].min() -1,
                               stop= x_set[:, 1].max() +1,
                               step=0.001))

var =classifier.predict(np.array([x1.ravel().]))



