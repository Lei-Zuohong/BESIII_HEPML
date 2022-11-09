# -*- coding: UTF-8 -*-
import hreweight as hreweight
import hreweight.cv as cv
import hreweight.hroot as hroot
import hreweight.reweight as reweight

# 读取数据
branchs = ['pip_heli', 'pim_heli', 'piz_heli',
           'pipm_m', 'pipz_m', 'pimz_m']
df_data = hroot.read_tree(file_input='test_file/data.root',
                          tree='data',
                          branchs=branchs)
df_mc = hroot.read_tree(file_input='test_file/mc.root',
                        tree='data',
                        branchs=branchs)
print(df_data)
print(df_mc)
# 设定超参数
argv = {'max_depth': [0, 3, 5, 7],
        'max_leaves': [0, 5, 10]}
searcher = cv.CVSearcher(argv)
# 输入数据
searcher.set_data(df_data)
searcher.set_mc(df_mc)
# 选择模型拟合
model = reweight.XGBReweighter(n_jobs=1)
searcher.gridsearchcv(model, n_jobs=1)
model = searcher.get_reweighter()
# 使用结果
weight = model.predict_weights(df_mc)