# -*- coding: UTF-8 -*-
# Public package
import numpy
import matplotlib.pyplot as plt
# Private package
# Internal package


def draw_distributions(origin, target,
                       origin_weights=None,
                       target_weights=None,
                       figures=[], show=False):
    hist_config_mc = {'bins': 50, 'density': True, 'alpha': 0.5}
    hist_config_data = {'bins': 50, 'density': True, 'alpha': 0.0}
    if(origin_weights is None):
        origin_weights = numpy.ones(origin.shape[0])
    if(target_weights is None):
        target_weights = numpy.ones(target.shape[0])
    plt.figure(figsize=[15, 7])
    columns = origin.columns
    for id, column in enumerate(columns, 1):
        xlim = numpy.percentile(numpy.hstack([target[column]]), [0.01, 99.99])
        plt.subplot(2, 3, id)
        plt.hist(origin[column], weights=origin_weights, range=xlim, **hist_config_mc, label='MC')
        hn, hb, hp = plt.hist(target[column], weights=target_weights, range=xlim, **hist_config_data, label='Data')
        hb = numpy.append([0], hb) + numpy.append(hb, [0])
        hb = hb[1:-1] / 2
        en = numpy.sqrt(hn) * numpy.sqrt(numpy.sum(hn)) / numpy.sqrt(len(target_weights))
        plt.errorbar(x=hb, y=hn, yerr=en, fmt='ko')
        plt.title(column)
        plt.legend(loc='best')
    for figure in figures:
        plt.savefig(figure)
    if(show):
        plt.show()
    