# -*- coding: UTF-8 -*-
"""
**hep_ml.reweight** contains reweighting algorithms.

Reweighting is procedure of finding such weights for original distribution,
that make distribution of one or several variables identical in original distribution and target distribution.

Typical application of this technique in HEP is reweighting of Monte-Carlo simulation results to minimize
disagreement between simulated data and real data.
Frequently the reweighting rule is trained on one part of data (normalization channel)
and applied to different (signal channel).

Remark: if each variable has identical distribution in two samples,
this doesn't imply that multidimensional distributions are equal (almost surely they aren't).
Aim of reweighters is to get identical multidimensional distributions.

Algorithms are implemented as estimators, fitting and reweighting stages are split.
Fitted reweighter can be applied many times to different data, pickled and so on.


Folding over reweighter is also availabel. This provides an easy way to run k-Folding cross-validation.
Also it is a nice way to combine weights predictions of trained reweighters.

Examples
________

The most common use case is reweighting of Monte-Carlo simulations results to sPlotted real data.
(original weights are all equal to 1 and could be skipped, but left here for example)

>>> from hep_ml.reweight import BinsReweighter, GBReweighter
>>> original_weights = numpy.ones(len(MC_data))
>>> reweighter = BinsReweighter(n_bins=100, n_neighs=3)
>>> reweighter.fit(original=MC_data, target=RealData,
>>>                original_weight=original_weights, target_weight=sWeights)
>>> MC_weights = reweighter.predict_weights(MC_data, original_weight=original_weights)

The same example for `GBReweighter`:

>>> reweighter = GBReweighter(max_depth=2, gb_args={'subsample': 0.5})
>>> reweighter.fit(original=MC_data, target=RealData, target_weight=sWeights)
>>> MC_weights = reweighter.predict_weights(MC_data)

Folding over reweighter:

>>> reweighter_base = GBReweighter(max_depth=2, gb_args={'subsample': 0.5})
>>> reweighter = FoldingReweighter(reweighter_base, n_folds=3)
>>> reweighter.fit(original=MC_data, target=RealData, target_weight=sWeights)

If the same data used in the training process are predicted by folding reweighter
weights predictions will be unbiased: each reweighter predicts only those part of data which is not used during its training

>>> MC_weights = reweighter.predict_weights(MC_data)
"""
# Public package
from __future__ import division, print_function, absolute_import
import copy
import numpy
import xgboost
from scipy.ndimage import gaussian_filter
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
# Private package
# Internal package
from .commonutils import check_sample_weight, weighted_quantile
from . import gradientboosting as gb
from . import binboost as bb
from . import losses

__author__ = 'Alex Rogozhnikov, Tatiana Likhomanenko'
__all__ = ['BinsReweighter', 'GBReweighter', 'FoldingReweighter']


def _bincount_nd(x, weights, shape):
    """
    Does the same thing as numpy.bincount, but allows binning in several integer variables.
    :param x: numpy.array of shape [n_samples, n_features] with non-negative integers
    :param weights: weights of samples, array of shape [n_samples]
    :param shape: shape of result, should be greater, then maximal value
    :return: weighted number of event in each bin, of shape=shape
    """
    assert len(weights) == len(x), 'length of weight is different: {} {}'.format(len(x), len(weights))
    assert x.shape[1] == len(shape), 'wrong length of shape: {} {}'.format(x.shape[1], len(shape))
    maximals = numpy.max(x, axis=0)
    assert numpy.all(maximals < shape), 'small shape passed: {} {}'.format(maximals, shape)

    result = numpy.zeros(shape, dtype=float)
    numpy.add.at(result, tuple(x.T), weights)
    return result


class ReweighterMixin(object):
    """Supplementary class which shows the interface of reweighter.
     Reweighters should be derived from this class."""
    n_features_ = None

    def _normalize_input(self, data, weights, normalize=True):
        """ Normalize input of reweighter
        :param data: array like of shape [n_samples] or [n_samples, n_features]
        :param weights: array-like of shape [n_samples] or None
        :return: tuple with
            data - numpy.array of shape [n_samples, n_features]
            weights - numpy.array of shape [n_samples] with mean = 1.
        """
        weights = check_sample_weight(data, sample_weight=weights, normalize=normalize)
        data = numpy.array(data)
        if len(data.shape) == 1:
            data = data[:, numpy.newaxis]
        if self.n_features_ is None:
            self.n_features_ = data.shape[1]
        assert self.n_features_ == data.shape[1], \
            'number of features is wrong: {} {}'.format(self.n_features_, data.shape[1])
        return data, weights

    def fit(self, original, target, original_weight, target_weight):
        raise NotImplementedError('To be overriden in descendants')

    def predict_weights(self, original, original_weight=None):
        raise NotImplementedError('To be overriden in descendants')


class BinsReweighter(BaseEstimator, ReweighterMixin):
    def __init__(self, n_bins=200, n_neighs=3.):
        """
        Use bins for reweighting. Bins' edges are computed using quantiles along each axis
        (which is better than bins of even size).

        This method works fine for 1d/2d histograms,
        while being unstable or inaccurate for higher dimensions.

        To make computed rule more smooth and stable, after computing weights in bins,
        gaussian filter is applied (so reweighting coefficient also includes information from neighbouring bins).

        :param int n_bins: how many bins to use for each input variable.
        :param float n_neighs: size of gaussian filter (in bins).
            This parameter is responsible for tradeoff between stability of rule and accuracy of predictions.
            With increase of n_neighs the reweighting rule becomes more stable.

        """
        self.n_bins = n_bins
        self.n_neighs = n_neighs
        # if number of events in bins is less than this value, number of events is clipped.
        self.min_in_the_bin = 1.

    def compute_bin_indices(self, data):
        """
        Compute id of bin along each axis.

        :param data: data, array-like of shape [n_samples, n_features]
            with the same order of features as in training
        :return: numpy.array of shape [n_samples, n_features] with integers, each from [0, n_bins - 1]
        """
        bin_indices = []
        for axis, axis_edges in enumerate(self.edges):
            bin_indices.append(numpy.searchsorted(axis_edges, data[:, axis]))
        return numpy.array(bin_indices).T

    def fit(self, original, target, original_weight=None, target_weight=None):
        """
        Prepare reweighting formula by computing histograms.

        :param original: values from original distribution, array-like of shape [n_samples, n_features]
        :param target: values from target distribution, array-like of shape [n_samples, n_features]
        :param original_weight: weights for samples of original distributions
        :param target_weight: weights for samples of original distributions
        :return: self
        """
        self.n_features_ = None
        original, original_weight = self._normalize_input(original, original_weight)
        target, target_weight = self._normalize_input(target, target_weight)
        target_perc = numpy.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = []
        for axis in range(self.n_features_):
            self.edges.append(weighted_quantile(target[:, axis], quantiles=target_perc, sample_weight=target_weight))

        bins_weights = []
        for data, weights in [(original, original_weight), (target, target_weight)]:
            bin_indices = self.compute_bin_indices(data)
            bin_w = _bincount_nd(bin_indices, weights=weights, shape=[self.n_bins] * self.n_features_)
            smeared_weights = gaussian_filter(bin_w, sigma=self.n_neighs, truncate=2.5)
            bins_weights.append(smeared_weights.clip(self.min_in_the_bin))
        bin_orig_weights, bin_targ_weights = bins_weights
        self.transition = bin_targ_weights / bin_orig_weights
        return self

    def predict_weights(self, original, original_weight=None):
        """
        Returns corrected weights. Result is computed as original_weight * reweighter_multipliers.

        :param original: values from original distribution of shape [n_samples, n_features]
        :param original_weight: weights of samples before reweighting.
        :return: numpy.array of shape [n_samples] with new weights.
        """
        original, original_weight = self._normalize_input(original, original_weight, normalize=False)
        bin_indices = self.compute_bin_indices(original)
        results = self.transition[tuple(bin_indices.T)] * original_weight
        return results


class EBinsReweighter(BaseEstimator, ReweighterMixin):
    def __init__(self, **argv):
        # 完备超参数集
        if('branchs_train' in argv):
            self.branchs_train = argv['branchs_train']
        else:
            self.branchs_train = []
        if('branchs_test' in argv):
            self.branchs_test = argv['branchs_test']
        else:
            self.branchs_test = []
        if('bins' in argv):
            self.bins = argv['bins']
        else:
            self.bins = []

    def get_params(self, deep=True):
        output = {'branchs_train': self.branchs_train,
                  'branchs_test': self.branchs_test,
                  'bins': self.bins}
        return output

    def set_params(self, **argv):
        for key in argv:
            if(not hasattr(self, key)):
                raise ValueError('Invalid parameter %s' % (key))
            exec('self.%s = argv["%s"]' % (key, key))
        return self

    def fit(self, X, yw):
        y = yw[:, 0]
        weight = yw[:, 1]
        index_m = numpy.where(y > 0.5)
        index_r = numpy.where(y < 0.5)
        X_m = X.iloc[index_m][self.branchs_train]
        X_r = X.iloc[index_r][self.branchs_train]
        y_m = y[index_m]
        y_r = y[index_r]
        weight_m = weight[index_m]
        weight_r = weight[index_r]
        self.bb = bb.REWEIGHTER()
        self.bb.set_bins(self.bins)
        self.bb.set_r(X_r, weight=weight_r)
        self.bb.set_m(X_m, weight=weight_m)
        self.bb.fit()
        return self

    def score(self, X, yw):
        # 切割参数
        y = yw[:, 0]
        weight = yw[:, 1]
        index_m = numpy.where(y > 0.5)
        index_r = numpy.where(y < 0.5)
        feature_m = X.iloc[index_m]
        feature_r = X.iloc[index_r]
        weight_m = weight[index_m]
        weight_r = weight[index_r]
        # 得到新权
        result, resulte = self.bb.get_weight(feature_m[self.branchs_train])
        weight_m = weight_m * result
        weighte_m = weight_m * resulte
        # 计算得分
        chisq = bb.scorend(feature_m[self.branchs_test],
                           weight_m,
                           weighte_m,
                           feature_r[self.branchs_test],
                           weight_r,
                           bins=int(numpy.mean(self.bins)))
        output = 1 / chisq
        print('Score: %f' % (output))
        return output

    def predict_weights(self, X, weight=None):
        output = self.bb.get_weight(X)
        if(not weight is None):
            output *= weight
        return output


class XGBReweighter(BaseEstimator, ReweighterMixin):
    def __init__(self, **gb_argv):
        # 完备超参数集
        default_argv = {'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 4,
                        'max_leaves': 10}
        if(gb_argv is None):
            self.gb_argv = {}
        else:
            self.gb_argv = gb_argv
        for key in default_argv:
            if(key not in self.gb_argv):
                self.gb_argv[key] = default_argv[key]

    def get_params(self, deep=True):
        output = copy.deepcopy(self.gb_argv)
        return output

    def set_params(self, **argv):
        for key in argv:
            if(key not in self.gb_argv):
                raise ValueError('Invalid parameter %s' % (key))
            self.gb_argv[key] = argv[key]
        return self

    def fit(self, X, yw):
        # 训练模型
        y = yw[:, 0]
        weight = yw[:, 1]
        self.xgb = xgboost.XGBClassifier(**self.gb_argv)
        self.xgb.fit(X, y, sample_weight=weight)
        return self

    def score(self, X, yw):
        # 切割参数
        y = yw[:, 0]
        weight = yw[:, 1]
        index_m = numpy.where(y > 0.5)
        index_r = numpy.where(y < 0.5)
        feature_m = X.iloc[index_m]
        feature_r = X.iloc[index_r]
        weight_m = weight[index_m]
        weight_r = weight[index_r]
        # 得到新权
        result = self.xgb.predict_proba(feature_m)
        weight_m = weight_m * result[:, 0] / result[:, 1]
        # 计算得分
        label_pred = self.xgb.predict_proba(X)[:, 1]
        label_pred_m = label_pred[index_m]
        label_pred_r = label_pred[index_r]
        p_m = numpy.tile(label_pred_m, (feature_r.shape[0], 1)).T
        p_r = numpy.tile(label_pred_r, (feature_m.shape[0], 1))
        index = numpy.where(p_m < p_r)
        weight_n = numpy.outer(weight_m, weight_r)
        output = numpy.sum(weight_n[index]) / numpy.sum(weight_n)
        output = 1 - abs(output - 0.5)
        print('Score: %f' % (output))
        return output

    def predict_weights(self, X, weight=None):
        output = self.xgb.predict_proba(X)
        output = output[:, 0] / output[:, 1]
        if(not weight is None):
            output *= weight
        return output


class GBReweighter(BaseEstimator, ReweighterMixin):
    def __init__(self, loss_regularization=5, **gb_argv):
        # 完备超参数集
        default_argv = {'n_estimators': 100,
                        'learning_rate': 0.1,
                        'subsample': 1.0,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': None,
                        'max_leaf_nodes': None,
                        'max_depth': 3,
                        'splitter': 'best',
                        'update_tree': True,
                        'train_features': None,
                        'random_state': None}
        if(gb_argv is None):
            self.gb_argv = {}
        else:
            self.gb_argv = gb_argv
        for key in default_argv:
            if(key not in self.gb_argv):
                self.gb_argv[key] = default_argv[key]
        # 其它参数
        self.loss_regularization = loss_regularization

    def get_params(self, deep=True):
        output = copy.deepcopy(self.gb_argv)
        return output

    def set_params(self, **argv):
        for key in argv:
            if(key not in self.gb_argv):
                raise ValueError('Invalid parameter %s' % (key))
            self.gb_argv[key] = argv[key]
        return self

    def fit(self, X, yw):
        # 训练模型
        loss = losses.ReweightLossFunction(regularization=self.loss_regularization)
        self.gb = gb.UGradientBoostingClassifier(loss=loss, **self.gb_argv)
        y = yw[:, 0]
        weight = yw[:, 1]
        self.gb.fit(X, y, sample_weight=weight)
        return self

    def score(self, X, yw):
        # 切割参数
        y = yw[:, 0]
        weight = yw[:, 1]
        index_m = numpy.where(y > 0.5)
        index_r = numpy.where(y < 0.5)
        feature_m = X.iloc[index_m]
        feature_r = X.iloc[index_r]
        weight_m = weight[index_m]
        weight_r = weight[index_r]
        # 得到新权
        result = numpy.exp(self.gb.decision_function(feature_m))
        weight_m = weight_m * result
        # 计算得分
        label_pred = self.gb.predict_proba(X)[:, 1]
        label_pred_m = label_pred[index_m]
        label_pred_r = label_pred[index_r]
        p_m = numpy.tile(label_pred_m, (feature_r.shape[0], 1)).T
        p_r = numpy.tile(label_pred_r, (feature_m.shape[0], 1))
        index = numpy.where(p_m < p_r)
        weight_n = numpy.outer(weight_m, weight_r)
        output = numpy.sum(weight_n[index]) / numpy.sum(weight_n)
        output = 1 - abs(output - 0.5)
        print('Score: %f' % (output))
        return output

    def predict_weights(self, X, weight=None):
        X, weight = self._normalize_input(X, weight, normalize=False)
        multipliers = numpy.exp(self.gb.decision_function(X))
        return multipliers * weight


class FoldingReweighter(BaseEstimator, ReweighterMixin):
    def __init__(self, base_reweighter, n_folds=2, random_state=None, verbose=True):
        """
        This meta-regressor implements folding algorithm over reweighter:

        * training data is splitted into n equal parts;

        * we train n reweighters, each one is trained using n-1 folds

        To build unbiased predictions for data, pass the **same** dataset (with same order of events)
        as in training to `predict_weights`, in which case
        a reweighter will be used to predict each event that the reweighter didn't use it during training.
        To use information from not one, but several reweighters during predictions,
        provide appropriate voting function. Examples of voting function:
        >>> voting = lambda x: numpy.mean(x, axis=0)
        >>> voting = lambda x: numpy.median(x, axis=0)

        :param base_reweighter: base reweighter object
        :type base_reweighter: ReweighterMixin
        :param n_folds: number of folds
        :param random_state: random state for reproducibility
        :type random_state: None or int or RandomState
        :param bool verbose:
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose
        self.base_reweighter = base_reweighter
        self.reweighters = []
        self._random_number = None
        self.train_length = None

    def _get_folds_column(self, length):
        """
        Return special column with indices of folds for all events.
        """
        if self._random_number is None:
            self._random_number = check_random_state(self.random_state).randint(0, 100000)
        folds_column = numpy.arange(length) % self.n_folds
        folds_column = numpy.random.RandomState(self._random_number).permutation(folds_column)
        return folds_column

    def fit(self, original, target, original_weight=None, target_weight=None):
        """
        Prepare reweighting formula by training a sequence of trees.

        :param original: values from original distribution, array-like of shape [n_samples, n_features]
        :param target: values from target distribution, array-like of shape [n_samples, n_features]
        :param original_weight: weights for samples of original distributions
        :param target_weight: weights for samples of original distributions
        :return: self
        """
        original, original_weight = self._normalize_input(original, original_weight, normalize=False)
        target, target_weight = self._normalize_input(target, target_weight, normalize=False)

        folds_original = self._get_folds_column(len(original))
        folds_target = self._get_folds_column(len(target))
        for _ in range(self.n_folds):
            self.reweighters.append(clone(self.base_reweighter))

        original = numpy.array(original)
        target = numpy.array(target)

        for i in range(self.n_folds):
            self.reweighters[i].fit(original[folds_original != i, :], target[folds_target != i, :],
                                    original_weight=original_weight[folds_original != i],
                                    target_weight=target_weight[folds_target != i])
        self.train_length = len(original)
        return self

    def predict_weights(self, original, original_weight=None, vote_function=None):
        """
        Returns corrected weights. Result is computed as original_weight * reweighter_multipliers.

        :param original: values from original distribution of shape [n_samples, n_features]
        :param original_weight: weights of samples before reweighting.
        :return: numpy.array of shape [n_samples] with new weights.
        :param vote_function: if using averaging over predictions of folds, this function shall be passed.
            For instance: lambda x: numpy.mean(x, axis=0), which means averaging result over all folds.
            Another useful option is lambda x: numpy.median(x, axis=0)
        """
        original, original_weight = self._normalize_input(original, original_weight, normalize=False)
        if vote_function is not None:
            if self.verbose:
                print('KFold prediction with voting function')
            results = []
            for reweighter in self.reweighters:
                results.append(reweighter.predict_weights(original, original_weight=original_weight))
            # results: [n_classifiers, n_samples], reduction is expected over 0th axis
            results = numpy.array(results)
            return vote_function(results)
        else:
            if self.verbose:
                if len(original) != self.train_length:
                    print('KFold prediction using random reweighter '
                          '(length of data passed not equal to length of train)')
                else:
                    print('KFold prediction using folds column')
            folds_original = self._get_folds_column(len(original))
            new_original_weight = numpy.zeros(len(original))
            original = numpy.asarray(original)
            for i in range(self.n_folds):
                new_original_weight[folds_original == i] = self.reweighters[i].predict_weights(
                    original[folds_original == i, :], original_weight=original_weight[folds_original == i])
            return new_original_weight

    def get_score_auc(self):
        aucs = [reweighter.get_score_aux() for reweighter in self.reweighters]
        output = numpy.mean(aucs)
        return output
