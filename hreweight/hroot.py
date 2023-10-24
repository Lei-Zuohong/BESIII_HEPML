# -*- coding: UTF-8 -*-
import pandas
import uproot


def read_tree(file_input='',  # 将一个root:tree转换为pandas格式
              tree='',
              branchs=[]):
    with uproot.open(file_input) as tfile:
        ttree = tfile[tree]
        tdict = ttree.arrays(library='np')
        if(len(branchs) > 0):
            ndict = {}
            for branch in branchs:
                ndict[branch] = tdict[branch]
            output = pandas.DataFrame(ndict)
        else:
            output = pandas.DataFrame(tdict)
    return output
