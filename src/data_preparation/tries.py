import os
import numpy as np
import pandas as pd
import pickle

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import lightgbm as lgb

input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
model_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/models/04/"

def getdf(dataf):
    dataf = dataf.copy()

    condition = dataf.groupby('pid')['year'].count()>2
    dataf = dataf.set_index('pid')[condition]
    year_list = dataf['year'].unique()

    dataf['hours_t1'] = np.NaN
    dataf['gross_earnings_t1'] = np.NaN

    dataf_out = pd.DataFrame()
    for i in np.sort(year_list)[2:]:
        df_now = dataf[dataf['year'] == i].copy()
        df_yesterday = dataf[dataf['year'] == (i-1)].copy()
        df_twoyesterdays = dataf[dataf['year'] == (i-2)].copy()

        df_now['lfs_t1'] = df_yesterday['lfs']
        df_now['working_t1'] = df_yesterday['working']
        df_now['fulltime_t1'] = df_yesterday['fulltime']
        df_now['hours_t1'] = df_yesterday['hours']
        df_now['hours_t2'] = df_twoyesterdays['hours']
        df_now['gross_earnings_t1'] = df_yesterday['gross_earnings']
        df_now['gross_earnings_t2'] = df_twoyesterdays['gross_earnings']

        dataf_out = pd.concat([dataf_out, df_now])

    dataf_out.reset_index(inplace=True)
    dataf_out.dropna(inplace=True)
    return dataf_out


df = pd.read_pickle(input_path + 'joined.pkl').dropna()
#df1 = getdf(df)


dataf = df.copy()

condition = dataf.groupby('pid')['year'].count()>2
dataf = dataf.set_index('pid')[condition]
year_list = dataf['year'].unique()

dataf['hours_t1'] = np.NaN
dataf['gross_earnings_t1'] = np.NaN

dataf_out = pd.DataFrame()
for i in np.sort(year_list)[2:13]:
    df_now = dataf[dataf['year'] == i].copy()
    df_yesterday = dataf[dataf['year'] == (i-1)].copy()
    df_twoyesterdays = dataf[dataf['year'] == (i-2)].copy()

    df_now['lfs_t1'] = df_yesterday['lfs']
    df_now['working_t1'] = df_yesterday['working']
    df_now['fulltime_t1'] = df_yesterday['fulltime']
    df_now['hours_t1'] = df_yesterday['hours']
    df_now['hours_t2'] = df_twoyesterdays['hours']
    df_now['gross_earnings_t1'] = df_yesterday['gross_earnings']
    df_now['gross_earnings_t2'] = df_twoyesterdays['gross_earnings']

    dataf_out = pd.concat([dataf_out, df_now])


sum(df_now.index.duplicated())
sum(df_yesterday.index.duplicated())

df_now.sort_index()
