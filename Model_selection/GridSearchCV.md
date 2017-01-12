## sklearn.model_selection.GridSearchCV

GridSearchCV(estimator, **param_grid**, **scoring=None**, fit_params=None, n_jobs=1, iid=True, refit=True, **cv=None**, verbose=0, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)

对给定的参数**穷举搜索**.

***

**Paramaters:**

* **estimator** : estimator object
    Either estimator needs to provide a score function, or **scoring** must be passed.

* **param_grid**: 参数网格
    dict or list of dictionaries

* **scoring**: 评价标准
    string, callable or None, default=None
    **常用的评价标准**：

    * ‘accuracy’
    * ‘f1’
    * ‘precision’
    * ‘recall’
    * ‘roc_auc’
    **sklearn.metrics.make_scorer**: Make a **scorer** from a performance metric or loss function.

* cv: K折交叉验证
    int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy
    if None, default 3-fold cross validation
    if int, 指定K值，一般为5
    可以是交叉验证生成器对象
    An iterable yielding train, test splits.

* verbose: integer
    Controls the verbosity: the higher, the more message.

* return_train_score : boolean, default=True
    如果为"False",属性"cv_results_"将不包括训练分数

***

**Attributes:**

* cv_results_ : dict of numpy (masked) ndarrays

* best_estimator_ : estimator

* best_score_ : float
    Score of best_estimator on the left out data.

* best_params_ : dict
    Parameter setting that gave the best results on the hold out data.

* scorer_ : function
    Scorer function used on the held out data to choose the best parameters for the mod

* n_splits_ : int
    The number of cross-validation splits (folds/iterations).
