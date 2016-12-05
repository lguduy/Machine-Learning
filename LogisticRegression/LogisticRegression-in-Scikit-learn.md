## 逻辑回归

> 翻译自Scikit-learn官方文档

逻辑回归不同于它的名字，它是一个用于**分类的线性模型**. 在这个模型中应用**逻辑函数**反映事件发生的可能性.

可用于二元和多元分类，并能用L1和L2范数限制模型复杂度，避免过拟合。

***

在函数 *LogisticRegression* 中，用于求解模型参数的优化算法有四种，分别为：

* “liblinear”: A Library for Large Linear Classification
* “newton-cg”: 一种改进的牛顿法
* “lbfgs”: 一种大规模优化算法，具备牛顿法收敛速度快的特点，但不需要牛顿法那样存储Hesse矩阵，因此节省了大量的空间以及计算资源
* “sag”： 一种SGD类算法，SAG是一种线性收敛算法

“liblinear”方法基于坐标下降算法( coordinate descent algorithm)，这是一种无导数的优化算法,为了找到目标函数的局部最小值，在每次迭代中的当前点上的一个坐标方向上搜索。使用封装在Scikit-learn中的用C++编写的 [LIBLINEAR library](http://www.csie.ntu.edu.tw/~cjlin/liblinear/). 但是基于liblinear库实现的CD算法不能学习一个”真“的多元分类模型. 作为替代，优化问题被分解为”一对多”的方式，为所有类都训练一个二元分类器, 最终表现为一个多元分类器. 使用L1正则化时，该算法能得到一个**稀疏解**.

“lbfgs”, “sag” and “newton-cg”算法只支持**L2正则化**, 对于高维数据优化速度快. 设置参数 **multi_class为 “multinomial”**，用上述优化算法，能够得到"真"的多元回归模型，理论上模型效果优于默认参数为 **“one-vs-rest”** 的情形.

“lbfgs”, “sag” 和 “newton-cg”优化算法不能优化L1正则项模型，因此 “multinomial” 选项不能得到模型的稀疏解.

"sag" 是一种SGD类算法，对于维度和数量都很大的数据集在速度上要优于其他算法。

| 案例                   |优化算法                       |
|----------------       |-------------                |
|小数据集或L1正则化        |“liblinear”
|“multinomial”或大数据集  |“lbfgs”, “sag” or “newton-cg” |
|超大数据集               |“sag”                         |

***

*LogisticRegressionCV* 用交叉验证的方法选择惩罚参数 *C*,“newton-cg”, “sag” and “lbfgs” 优化算法由于 **warm-starting**, 对于高维密集型速度更快.

对于多元分类， 如果参数 multi_class 设置为 **"ovr"**(“one-vs-rest”), 每个类得到一个最优的 C , 如果设置为 **"multinomial"**, 通过最小化交叉熵损失得到一个最优的 C.

***

sklearn.linear_model.LogisticRegression参数介绍

sklearn.linear_model.LogisticRegression(*penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1*)

‘newton-cg’, ‘sag’, and ‘lbfgs’ 算法只支持L2正则化的原始解法(primal formulation)，‘liblinear’ 支持L1和L2正则化， 支持L2正则化的**对偶解法**(dual formulation).

**Parameters:**

* penalty : ‘l1’ or ‘l2’, default: ‘l2’
* dual : bool, default: False
* C : float, default: 1.0. 正则项惩罚系数的倒数.
* fit_intercept : bool, default: True ?
* intercept_scaling : float, default 1 ?
当使用‘liblinear’算法，且 self.fit_intercept 为"True"时起作用
* class_weight : dict or ‘balanced’, default: None
**“balanced”** 模式根据y值自动确定权值，n_samples / (n_classes * np.bincount(y)); np.bincount(y)): 用于统计y中各类的个数. 这个权值会和 **sample_weight** 指定的权值通过合适的方法进行叠加.
* max_iter : int, default: 100. 只对newton-cg, sag 和 lbfgs solvers优化算法有效.
* random_state : int seed, RandomState instance, default: None. 只对‘sag’ 和 ‘liblinear’ 有效.
* solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’}, default: ‘liblinear’. Note that ‘sag’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing
* tol : float, default: 1e-4
* multi_class : str, {‘ovr’, ‘multinomial’}, default: ‘ovr’
* warm_start : bool, default: False. 当设置为真时，调用先前的解决方案作为其初始化， 否则清除先前的解决方案. 只对‘newton-cg’, ‘sag’ 和 ‘lbfgs’ 有效.
* n_jobs : int, default: 1. 在交叉验证循环式调用的CPU核心数. 如果设置为-1全部核心会被调用.

**Attributes:**

* coef_ : array, shape (n_classes, n_features). 模型参数.
* intercept_ : array, shape (n_classes,). 偏置. Intercept (a.k.a. bias) added to the decision function. **If fit_intercept is set to False, the intercept is set to zero**.

***

sklearn.linear_model.LogisticRegressionCV *(Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', **scoring=None**, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=1, verbose=0, refit=True, intercept_scaling=1.0, multi_class='ovr', random_state=None)*

**Parameters:**

* scoring : callabale
Scoring function to use as cross-validation criteria. For a list of scoring functions that can be used, look at **sklearn.metrics**. The default scoring option used is **accuracy_score**.

```python
clf = LogisticRegressionCV(Cs=Cs, cv=4, class_weight="balanced",
                            scoring="roc_auc")
clf.fit(train_x, train_y)
```

常用的评价标准：

* ‘accuracy’
* ‘f1’
* ‘precision’
* ‘recall’
* ‘roc_auc’
