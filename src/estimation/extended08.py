import os
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression, LinearRegression
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
    for i in year_list[2:]:
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

def get_dependent_var(dataf, dep_var):
    dataf = dataf.copy()

    dataf.rename(columns={dep_var: 'dep_var'}, inplace=True)
    return dataf

def _prepare_classifier(dataf):
    dataf = dataf.copy()

    y = dataf['dep_var']
    X = dataf.drop('dep_var', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

    X_train_scaled = StandardScaler().fit_transform(np.asarray(X_train))
    X_test_scaled = StandardScaler().fit_transform(np.asarray(X_test))

    feature_names = X.columns.tolist()

    # For ML part:
    lgb_train = lgb.Dataset(X_train_scaled, y_train, free_raw_data=False)
    lgb_test = lgb.Dataset(X_test_scaled, y_test, free_raw_data=False)

    out_dici = {'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'lgb_train': lgb_train,
                'lgb_test': lgb_test,
                'features': feature_names}
    return out_dici

def _prepare_regressor(dataf):
        dataf = dataf.copy()

        y = dataf['dep_var']
        X = dataf.drop('dep_var', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

        X_train_scaled = StandardScaler().fit_transform(np.asarray(X_train))
        X_test_scaled = StandardScaler().fit_transform(np.asarray(X_test))
        y_train_scaled = StandardScaler().fit_transform(np.asarray(y_train).reshape(-1,1))

        # Saving the scaler of the test data to convert the predicted values again
        y_test_scaler = StandardScaler().fit(np.asarray(y_test).reshape(-1,1))
        y_test_scaled = y_test_scaler.transform(np.asarray(y_test).reshape(-1,1))

        feature_names = X.columns.tolist()
        y_test_scaled = np.ravel(y_test_scaled)
        y_train_scaled = np.ravel(y_train_scaled)

        # For ML part:
        lgb_train = lgb.Dataset(X_train_scaled,
                                y_train_scaled, free_raw_data=False)
        lgb_test = lgb.Dataset(X_test_scaled,
                               y_test_scaled, free_raw_data=False)


        out_dici = {'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'scaler': y_test_scaler,
                    'lgb_train': lgb_train,
                    'lgb_test': lgb_test,
                    'features': feature_names}
        return out_dici

#############################################################################

def data_general(dataf, dep_var, estimate=1):
    dataf = dataf.copy()


    if estimate == 1:
        dataf = get_dependent_var(dataf, dep_var)
    else:
        dataf = get_dependent_var(dataf, dep_var)
        dataf.drop('dep_var', axis=1, inplace=True)

    vars_drop = ['pid',
                 'hid',
                 'orighid',
                 'age_max']
    dataf.drop(vars_drop, axis=1, inplace=True)
    return dataf

def _estimate(dataf, dep_var, type):
    dataf = dataf.copy()

    dataf = data_general(dataf, dep_var)
    dataf.dropna(inplace=True)

    if type == 'regression':
        dict = _prepare_regressor(dataf)
        params = {'boosting_type' : 'gbdt',
                  'n_estimators': 350,
                  'objective' : 'l2',
                  'metric' : 'l2',
                  'num_leaves' : 31,
                  'learning_rate' : 0.15,
                  'feature_fraction': [0.9],
                  'bagging_fraction': [0.8],
                  'bagging_freq': [5],
                  'verbose' : 5}
    elif type == 'classifier':
        dict = _prepare_classifier(dataf)
        params = {'task' : 'train',
                  'boosting_type' : 'gbdt',
                  'n_estimators': 350,
                  'objective': 'multiclass',
                  'num_class': len(dict['y_train'].unique()),
                  'eval_metric': 'multi_logloss',
                  'learning_rate': 0.05,
                  'feature_fraction': [0.9],
                  'num_leaves': 31,
                  'verbose': 0}
    else:
        dict = _prepare_classifier(dataf)
        params = {'task' : 'train',
                  'boosting_type' : 'gbdt',
                  'n_estimators': 350,
                  'objective': 'binary',
                  'eval_metric': 'logloss',
                  'learning_rate': 0.05,
                  'feature_fraction': [0.9],
                  'num_leaves': 31,
                  'verbose': 0}

    modl = lgb.train(params,
                     train_set = dict['lgb_train'],
                     valid_sets = dict['lgb_test'],
                     feature_name = dict['features'],
                     early_stopping_rounds = 5)

    if dep_var == 'gross_earnings':
        modl.save_model(model_path + "earnings_extended.txt")
    else:
        modl.save_model(model_path + dep_var + "_extended.txt")


# df = pd.read_pickle(input_path + 'imputed08').dropna()
# df1 = getdf(df)
# df1['predicted'] = 1
#
# _estimate(df1, 'birth', 'binary')
# _estimate(df1, 'lfs', 'binary')
# _estimate(df1, 'working', 'binary')
# _estimate(df1, 'fulltime', 'binary')
# _estimate(df1, 'hours', 'regression')
# _estimate(df1, 'gross_earnings', 'regression')
