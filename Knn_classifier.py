# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 05:51:38 2017

@author: Mahedy
"""
import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("heart disease knn.csv", sep =',', header=None)
#print(df)
print("\t")

X = np.array(df.iloc[:,0:13])
y = np.array(df[13])

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.40,random_state=50)

print("Data For Testing\t")
print(X_test,y_test)

clf = KNeighborsClassifier(n_neighbors=5)

clf.fit(X_train,y_train)
predicted = clf.predict(X_test)
print("\nAccuracy :\t", accuracy_score(y_test,predicted)*100)



matrix = confusion_matrix(y_test, predicted)
print("\nConfusion Matrix For Test Data :\t")
print(matrix)

report = classification_report(y_test, predicted)
print("\nClassification Report For Test Data :\t")
print(report)


kfold = model_selection.KFold(n_splits=10, random_state=None)
model =KNeighborsClassifier()
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print(results)
print("\nArea Under ROC Curve :\t",results.mean())


