#! /usr/bin/python

#default lib imports
import sys
import numpy as np
from sys import argv
#sklearn imports
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

#Store arguments in sample array
sample = [sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]]
sample = np.array(sample).reshape(1,-1)

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)

#Classifier
clf = joblib.load('mejor_modelo.pkl')
#Fitting
clf.fit(X_train, y_train)

#Show prediction
print np.mean(clf.predict(sample))
