## 支持向量机

> 翻译自Scikit-learn官方文档

### 优缺点

优点：

* 在高维空间依然有效
* 数据维度大于数据量是依然有效
* 可以用不同的核函数解决不同的问题，还可以自定义核函数

缺点：

* 特征数量远大于样本数时，表现不好
* 不能直接给出概率估计，而是用一种代价很大的五折交叉验证方式计算概率估计

***

### 多元分类

**SVC, NuSVC and LinearSVC** 都能进行 **多元分类**

* SVC: 支持向量分类，支持各种核函数，基于libsvm实现
* NuSVC: 与SVC类似，但使用一个参数控制支持向量的个数,基于libsvm实现
* LinearSVC: 与线性核的SVC类似，基于liblinear，对于惩罚项和损失函数的选择很 灵活，对于大型数据集效果更好。

SVC和NuSVC对于多元分类采用 **"one-against-one"** 方法
LinearSVC采用 **"one-vs-the-rest"**

### sklearn.svm.LinearSVC

class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

**Parameters:**

* loss : 损失函数, ‘hinge’ or ‘squared_hinge’ (default=’squared_hinge’), ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’ is the square of the hinge loss.

* class_weight : {dict, ‘balanced’}, optional
例: clf = svm.SVC(class_weight={1: 10}) 
