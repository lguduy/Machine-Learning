# -*- coding: utf-8 -*-
"""
Created on 2016/12/19 20:00

支持向量机
Plot different SVM classifiers on the iris dataset
注意：
1. 原来一直以为分类边界是用分类器的coef_和intercept_参数画出来的
   实际上是生成采样点，用各个分类器去预测每个点的分类，间接生成分类边界

@author: lguduy
"""

from sklearn import svm, datasets
import numpy as np
import matplotlib.pyplot as plt
"""
To list all available styles, use print(plt.style.available)
"""

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

# 建模
Linearsvc = svm.LinearSVC(dual=False).fit(X, Y)
Linear_svc = svm.SVC(kernel='linear').fit(X, Y)           # Linear kernel
Poly_svc = svm.SVC(kernel='poly').fit(X, Y)               # Ploy kernel
Rbf_svc = svm.SVC(C=2.0).fit(X, Y)                        # Rbf kernel

# 网格
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))       # 生成采样点

# Title
titles = ['LinearSVC (linear kernel)',
          'SVC with linear kernel',
          'SVC with polynomial kernel',
          'SVC with RBF kernel']

for i, clf in enumerate((Linearsvc, Linear_svc, Poly_svc, Rbf_svc)):
    plt.subplot(2, 2, i+1)
    plt.subplots_adjust(wspace=0.3, hspace=0.6)           # 控制图之间的间隙

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.coolwarm, alpha=0.8)
    """scatter: 散点图
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, alpha=0.8)
    c: color
    s: scalar 散点大小
    alpha: 透明度
    """
    plt.xlabel('Sepal length', fontsize=12)
    plt.ylabel('Sepal width', fontsize=12)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.yticks(xrange(int(yy.min()), int(yy.max())+1, 1))  # 坐标点标注
    plt.title(titles[i], fontsize=14)

fig_file = r"/home/liangyu/Github/Machine-Learning/SVM/fig-01"
plt.savefig(fig_file, dpi=300)
plt.show()
plt.close()
