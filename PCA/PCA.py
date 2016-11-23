# -*- coding: utf-8 -*-
"""
Created on 2016/11/21 15:21

@author: lguduy
"""

from sklearn import datasets, decomposition
import matplotlib.pyplot as plt

#Data
digits = datasets.load_digits()
x = digits.data
y = digits.target

#Model
pca = decomposition.PCA()

pca.fit(x)

plt.figure()
plt.plot(pca.explained_variance_, 'k')
plt.xlabel('n_components', fontsize=16)
plt.ylabel('explained_variance_', fontsize=16)
plt.show()
