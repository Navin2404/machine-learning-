#Naive bayes

#Importing the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset=pd.read_csv('iris_new.csv')

x=dataset.iloc[:,0:4].values
y=dataset.iloc[:, -1].values

#splitting the dataset in to train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25,random_state=0)

#Training the Naive bayes model on the training set
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

classifier = GaussianNB()

classifier.fit(x_train, y_train)

#predicting the Test set results
y_pred = classifier.predict(x_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

acc = (sum(np.diag(cm))/len(y_test))
acc

#Method to findind accuracy
from sklearn import metrics

acc1=metrics.accuracy_score(y_test,y_pred)

from sklearn.preprocessing import LabelEncoder

labelen = LabelEncoder()
yy = labelen.fit_transform(y)



