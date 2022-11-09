# -*- coding: UTF-8 -*-
import hreweight as hreweight
import hreweight.hroot as hroot

#
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
#
argv = {'max_depth': [0, 3, 5, 7],
        'max_leaves': [0, 5, 10]}
