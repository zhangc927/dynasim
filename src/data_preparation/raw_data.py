import os
import numpy as np
import pandas as pd
from numba import njit
import importlib

data_path = "/Users/christianhilscher/Desktop/dynsim/src/data_preparation/"
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"


def _get_multiindex(dataf):
    dataf = dataf.copy()
    index_list = _get_tupleindices(dataf)

    mindex = pd.MultiIndex.from_tuples(index_list, names=['syear' ,
                                                          'pid'])
    dataf_out = dataf.set_index(mindex)
    dataf_out = dataf_out.sort_index(level=1)

    return dataf_out

def _get_tupleindices(dataf):
    years = dataf['syear'].tolist()
    hids = dataf['pid'].tolist()
    return list(zip(years, hids))

df = pd.read_stata(input_path + 'SOEP_prepared_costs_2019-11-27.dta')
df_addition = df[['pid', 'syear', 'pglfs', 'migback', 'whours_actual', 'whours_usual']]

df_basic = pd.read_stata(input_path + 'SOEP_prepared_costs_2019-11-27_restricted.dta')


def _bring_together(dataf1, dataf2):
    dataf1 = dataf1.copy()
    dataf2 = dataf2.copy()

    dataf1 = _get_multiindex(dataf1)
    dataf2 = _get_multiindex(dataf2)

    dataf_out = pd.concat([dataf1, dataf2], axis = 1, join='inner')
    dataf_out.rename(columns={dataf_out.columns[18] : 'drop1',
                              dataf_out.columns[19] : 'drop2'},
                     inplace=True)

    dataf_out.reset_index(inplace=True)

    dataf_out.drop(['drop1', 'drop2'], axis=1, inplace=True)

    return dataf_out

final_df = _bring_together(df_basic, df_addition)
final_df

final_df.to_pickle(input_path + "full")
