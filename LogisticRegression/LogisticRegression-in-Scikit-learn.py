# -*- coding: utf-8 -*-
"""
Created on 2016/12/03 15:00

基于Scikit-learn实现LogisticRegression
利用交叉验证和auc最优准则选取参数C
画出ROC曲线评价分类器

@author: lguduy
"""

from sklearn import datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

"""肿瘤数据集，二元分类，用于识别恶性良性"""
cancer = datasets.load_breast_cancer()
X = cancer.data
target = cancer.target
Y = target.reshape((target.shape[0], 1))

train_x = X[:400, :]
train_y = Y[:400, :]

test_x = X[400:, :]
test_y = Y[400:, :]

C = xrange(10, 101, 2)                           # 惩罚系数的倒数， 用来控制模型复杂度

"""用roc_auc为评价标准选择参数"""
clf = LogisticRegressionCV(Cs=C, cv=4, class_weight="balanced",
                           scoring="roc_auc")
clf.fit(train_x, train_y)

"""Predict"""
predict_y = clf.predict(test_x)
Proba_y = clf.predict_proba(test_x)
score = clf.score(test_x, test_y)                # mean accuracy on test data

"""模型相关参数"""
coef = clf.coef_                                 # 权重
intercept = clf.intercept_                       # 偏置
C = clf.C_                                       # 惩罚系数倒数

"""Plot roc_curve"""
fpr, tpr, thresholds = roc_curve(test_y, Proba_y[:, 1], pos_label=1)
auc = roc_auc_score(test_y, Proba_y[:, 1])

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
