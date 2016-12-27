# -*- coding: utf-8 -*-
"""
Created on 2016/12/21 16:00

Support Vector Machine
Use grid_search and cross_validation
to find best paramaters and kernel of SVC

@author: lguduy
"""
from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt

"""Load data"""
iris = datasets.load_iris()

"""train_test_split():将输入数据随机分为训练数据和测试数据

Parameters
----------
test_size: float, int, or None (default is None)
    如果是小数，是test_data占data的百分比
    如果是int,是test_data的个数
    如果是None,是除去train_data的剩余数据，如果train_data也是None,test_size=0.25

random_state:int or RandomState
    Pseudo-random number generator state used for random sampling.
"""
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size=0.3,
                                                    random_state=0)

"""Parameters_grid"""
parameters_grid = [{'kernel':['rbf'],
                    'gamma':[float(i)/1000 for i in xrange(10,1000,5)],
                    'C':xrange(1, 50, 5)}]

"""Grid_search and Cross_validation"""
clf = GridSearchCV(SVC(), param_grid=parameters_grid, scoring='accuracy', cv=5, n_jobs=-1)

clf.fit(X_train, y_train)

"""Predict based on best paramaters"""
y_true, y_pred =y_test, clf.predict(X_test)

score = clf.score(X_test, y_true)

print(classification_report(y_true, y_pred))
