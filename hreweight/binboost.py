# -*- coding: UTF-8 -*-
# Public package
import copy
import numpy
import pandas
# Private package
import headpy.hmath.hstatis as hstatis
# Internal package


class REWEIGHTER:
    def __init__(self):
        self.bins = []

    def set_bins(self, bins):  # 设定多维分bin
        self.bins = bins
        self.dimension = len(bins)

    def set_r(self, X, weight=None):  # 设定Data数据
        self.data_r = X
        if(weight is None):
            self.weight_r = numpy.ones(X.shape[0])
        else:
            self.weight_r = weight
        self.branchs = self.data_r.columns

    def set_m(self, X, weight=None):  # 设定MC数据
        self.data_m = X
        if(weight is None):
            self.weight_m = numpy.ones(X.shape[0])
        else:
            self.weight_m = weight

    def set_axis(self):  # 设定hist坐标轴
        self.axis = []
        for count, branch in enumerate(self.branchs):
            min = numpy.min(numpy.append(self.data_m[branch], self.data_r[branch]))
            max = numpy.max(numpy.append(self.data_m[branch], self.data_r[branch]))
            self.axis.append(hstatis.BINS(left=min - 0.05 * (max - min),
                                          right=max + 0.05 * (max - min),
                                          inter=self.bins[count]))

    def set_matrix(self, data, weight):  # 转换数据为hist
        output = numpy.zeros(self.bins)
        for i in range(data.shape[0]):
            output[tuple([self.axis[j].get_bin_index(data[self.branchs[j]].iloc[i]) for j in range(self.dimension)])] += weight[i]
        return output

    def fit(self):
        # 准备数据
        self.check()
        self.set_axis()
        # 统计hist
        matrix_r = self.set_matrix(self.data_r, self.weight_r)
        matrix_m = self.set_matrix(self.data_m, self.weight_m)
        # 计算误差并归一化
        matrix_r *= self.data_r.shape[0] / numpy.sum(matrix_r)
        matrix_er = numpy.sqrt(matrix_r)
        matrix_r /= self.data_r.shape[0]
        matrix_er /= self.data_r.shape[0]
        matrix_m /= numpy.sum(matrix_m)
        # 计算权重及误差
        index = numpy.where(matrix_m <= 0)
        matrix_r[index] = 0
        matrix_er[index] = 0
        matrix_m[index] = 1
        self.matrix_w = matrix_r / matrix_m
        self.matrix_ew = matrix_er / matrix_m

    def get_weight(self, X):  # 获得权重
        output = numpy.array([self.matrix_w[tuple([self.axis[j].get_bin_index(X[self.branchs[j]].iloc[i]) for j in range(self.dimension)])] for i in range(X.shape[0])])
        outpute = numpy.array([self.matrix_ew[tuple([self.axis[j].get_bin_index(X[self.branchs[j]].iloc[i]) for j in range(self.dimension)])] for i in range(X.shape[0])])
        return output, outpute

    def check(self):  # 检查数据合法性
        assert(self.dimension == self.data_m.shape[1])
        assert(self.dimension == self.data_r.shape[1])
        assert(self.dimension == len(self.bins))
        assert(self.data_m.shape[0] == self.weight_m.shape[0])
        assert(self.data_r.shape[0] == self.weight_r.shape[0])
        for i in range(self.dimension):
            assert(self.data_m.columns[i] == self.data_r.columns[i])


def scorend(data1, weight1, weighte, data2, weight2, bins=50):
    dimension = data1.shape[1]
    columns = data1.columns
    assert(dimension == data2.shape[1])
    assert(data1.shape[0] == weight1.shape[0])
    assert(data1.shape[0] == weighte.shape[0])
    assert(data2.shape[0] == weight2.shape[0])
    for i in range(dimension):
        assert(data1.columns[i] == data2.columns[i])
    chisq = numpy.array([score1d(data1[column],
                                 weight1,
                                 weighte,
                                 data2[column],
                                 weight2,
                                 bins=bins) for column in columns])
    chisq = numpy.sqrt(numpy.sum(chisq**2) / chisq.shape[0])
    return chisq


def score1d(data1, weight1, weighte, data2, weight2, bins=50):
    min = numpy.min(numpy.append(data1, data2))
    max = numpy.max(numpy.append(data1, data2))

    axis = hstatis.BINS(left=min - 0.05 * (max - min),
                        right=max + 0.05 * (max - min),
                        inter=bins)
    # 统计MC
    hist1 = numpy.zeros(bins)
    hist1e = numpy.zeros(bins)
    for count, i in enumerate(data1):
        index = axis.get_bin_index(i)
        hist1[index] += weight1[count]
        hist1e[index] = numpy.sqrt(hist1e[index]**2 + weighte[count]**2)
    hist2 = numpy.zeros(bins)
    for count, i in enumerate(data2):
        index = axis.get_bin_index(i)
        hist2[index] += weight2[count]
    hist2e = numpy.sqrt(hist2)
    scale1 = numpy.sum(hist1)
    scale2 = numpy.sum(hist2)
    hist1 /= scale1
    hist1e /= scale1
    hist2 /= scale2
    hist2e /= scale2
    histe = numpy.sqrt(hist1e**2 + hist2e**2)
    index = numpy.where(histe > 0)
    chisq = abs(hist1[index] - hist2[index]) / histe[index]
    chisq = numpy.sqrt(numpy.sum(chisq**2) / chisq.shape[0])
    return chisq
