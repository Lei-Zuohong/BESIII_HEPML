# -*- coding: UTF-8 -*-
# Public package
import copy
import time
import numpy
import pandas
import sklearn
import sklearn.model_selection
# Private package
# Internal package


class CVSearcher:
    def __init__(self, argv={}):
        self.argv = argv

    def set(self, name, values):
        self.argv[name] = values

    def set_data(self, df_r, weight=None):
        self.df_r = df_r
        if(not weight is None):
            self.weight_r = weight

    def set_mc(self, df_m, weight=None):
        self.df_m = df_m
        if(not weight is None):
            self.weight_m = weight

    def gridsearchcv(self, model, n_jobs=-1, cv=None):
        # 生成输入数据
        X = pandas.concat([self.df_m, self.df_r]).reset_index(drop=True)
        y = numpy.append(numpy.ones(self.df_m.shape[0]), numpy.zeros(self.df_r.shape[0]))
        if(hasattr(self, 'weight_m')):
            weight_m = self.weight_m
        else:
            weight_m = numpy.ones(self.df_m.shape[0])
        if(hasattr(self, 'weight_r')):
            weight_r = self.weight_r
        else:
            weight_r = numpy.ones(self.df_r.shape[0])
        weight = numpy.append(weight_m, weight_r)
        series = numpy.arange(0, X.shape[0], 1)
        numpy.random.shuffle(series)
        self.X = X.iloc[series]
        self.y = y[series]
        self.weight = weight[series]
        self.yw = numpy.array([self.y, self.weight]).T
        # 优化模型
        self.reweighter = copy.deepcopy(model)
        searcher = sklearn.model_selection.GridSearchCV(estimator=self.reweighter,
                                                        param_grid=self.argv,
                                                        n_jobs=n_jobs,
                                                        cv=cv)
        searcher.fit(self.X, self.yw)
        self.searcher = searcher
        return searcher

    def randomserachcv(self, model, iter=10):
        # 生成输入数据
        X = pandas.concat([self.df_m, self.df_r]).reset_index(drop=True)
        y = numpy.append(numpy.ones(self.df_m.shape[0]), numpy.zeros(self.df_r.shape[0]))
        if(hasattr(self, 'weight_m')):
            weight_m = self.weight_m
        else:
            weight_m = numpy.ones(self.df_m.shape[0])
        if(hasattr(self, 'weight_r')):
            weight_r = self.weight_r
        else:
            weight_r = numpy.ones(self.df_r.shape[0])
        weight = numpy.append(weight_m, weight_r)
        series = numpy.arange(0, X.shape[0], 1)
        numpy.random.shuffle(series)
        self.X = X.iloc[series]
        self.y = y[series]
        self.weight = weight[series]
        self.yw = numpy.array([self.y, self.weight]).T
        # 优化模型
        self.reweighter = copy.deepcopy(model)
        searcher = sklearn.model_selection.RandomizedSearchCV(estimator=self.reweighter, param_grid=self.argv, n_jobs=-1,
                                                              n_iter=iter)
        searcher.fit(self.X, self.yw)
        self.searcher = searcher
        return searcher

    def get_reweighter(self):
        self.reweighter.set_params(**self.searcher.best_params_)
        self.reweighter.fit(self.X, self.yw)
        return self.reweighter
