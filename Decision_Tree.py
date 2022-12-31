# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:37:30 2022

@author: HAI
"""

#Decision Tree

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the dataset
dataset = pd.read_csv('iris_new.csv')
x=dataset.iloc[:,0:4].values
y=dataset.iloc[:, -1].values

#splitting the dataset in to train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25,random_state=0)

#Feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Training the decision Tree classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

#Predicting the Test set results
y_pred = classifier.predict(x_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

acc = (sum(np.diag(cm))/len(y_test))
acc

feature_col=dataset
feature_col=feature_col.drop('spectype',axis=1)

#Lets visualize the decision tree

from sklearn.tree import plot_tree
plt.figure(dpi=300)
dec_tree = plot_tree(decision_tree=classifier, max_depth=20, feature_names=feature_col.columns,
                     class_names=["setosa","vercicolor","verginica"],
                     filled=True, precision=1, rounded= True, fontsize=6)

plt.show()







