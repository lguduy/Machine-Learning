# -*- coding: utf-8 -*-
"""
Created on 2016/12/06 10:00

DecisionTree

@author: lguduy
"""

from sklearn import tree, datasets
import numpy as np
import pydotplus
from IPython.display import Image

"""载入鸢尾花数据"""
iris = datasets.load_iris()
data = iris.data
target = iris.target

"""训练测试输入输出"""
index = np.random.permutation(len(data))        # 产生随机序列
iris_x_train = data[index[:120]]
iris_y_train = target[index[:120]]
iris_x_test = data[index[120:]]
iris_y_test = target[index[120:]]

"""训练及预测"""
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris_x_train, iris_y_train)

iris_y_predict = clf.predict(iris_x_test)

"""可视化决策树"""
pdf_filename = r"~/Github/Machine-Learning/DecisionTree/iris_tree.pdf"
dot_data = tree.export_graphviz(clf, out_file=pdf_filename,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names)


graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(pdf_filename)
