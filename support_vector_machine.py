#Support vector machine

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the dataset
dataset = pd.read_csv('social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:, -1].values

#splitting the dataset in to train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.20,random_state=0)

#Feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Training the svm model in the training set
from sklearn.svm import SVC
classifier = SVC(C=1000,kernel='rbf',random_state=0,gamma= 0.5)
classifier.fit(x_train ,y_train)

#Predictig the test set result
y_pred = classifier.predict(x_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

acc = (sum(np.diag(cm))/len(y_test))
acc





