import os
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import lightgbm as lgb

input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
model_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/models/"

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

    dataf.sort_values(by=["pid", "year"])
    dups = dataf["pid"].duplicated()

    dataf["dep_var"] = dataf[dep_var].shift(-1)

    dups = dups.shift(-1)[:-1]
    dups.reset_index(drop=True, inplace=True)

    dataf = dataf[:-1]
    dataf.reset_index(drop=True, inplace=True)
    dataf = dataf[dups]

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
##############################################################################

def data_lfs(dataf, estimate=1):
    dataf = dataf.copy()

    if estimate == 1:
        dataf= get_dependent_var(dataf, 'lfs')
        vars_retain = ['dep_var',
                       'lfs_t1',
                       'working_t1',
                       'n_children',
                       'hh_youngest_age',
                       'hh_income',
                       'hh_frac_working']
    elif estimate == 0:
        vars_retain = ['lfs_t1',
                       'working_t1',
                       'n_children',
                       'hh_youngest_age',
                       'hh_income',
                       'hh_frac_working']
    else:
        raise ValueError("0 is for simulation, 1 for estimation")

    dataf_out = dataf[vars_retain]
    return dataf_out

def estimate_lfs(dataf):
    dataf = dataf.copy()

    dataf = data_lfs(dataf)
    dataf.dropna(inplace=True)
    dict = _prepare_classifier(dataf)

    params_m = {'task' : 'train',
        'boosting_type' : 'gbdt',
        'n_estimators': 350,
        'objective': 'binary',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'feature_fraction': [0.9],
        'num_leaves': 31,
        'verbose': 0}

    logit = LogisticRegression().fit(dict['X_train'],
                                         dict['y_train'])

    ml = lgb.train(params_m,
                   train_set = dict['lgb_train'],
                   valid_sets = dict['lgb_test'],
                   feature_name = dict['features'],
                   early_stopping_rounds = 5)

    pickle.dump(logit,
                open(model_path + "lfs_logit", 'wb'))
    ml.save_model(model_path + "lfs_ml.txt")

def data_working(dataf, estimate=1):
    dataf = dataf.copy()
    dataf = dataf[dataf['lfs']==1]


    if estimate == 1:
        dataf= get_dependent_var(dataf, 'working')
        vars_retain = ['dep_var',
                       'fulltime_t1',
                       'working_t1',
                       'n_children',
                       'hh_youngest_age',
                       'hh_income',
                       'hh_frac_working']
    elif estimate == 0:
        vars_retain = ['fulltime_t1',
                       'working_t1',
                       'n_children',
                       'hh_youngest_age',
                       'hh_income',
                       'hh_frac_working']
    else:
        raise ValueError("0 is for simulation, 1 for estimation")

    dataf_out = dataf[vars_retain]
    return dataf_out

def estimate_working(dataf):
    dataf = dataf.copy()

    dataf = data_working(dataf)
    dataf.dropna(inplace=True)
    dict = _prepare_classifier(dataf)

    params_m = {'task' : 'train',
        'boosting_type' : 'gbdt',
        'n_estimators': 350,
        'objective': 'binary',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'feature_fraction': [0.9],
        'num_leaves': 31,
        'verbose': 0}

    logit = LogisticRegression().fit(dict['X_train'],
                                         dict['y_train'])

    ml = lgb.train(params_m,
                   train_set = dict['lgb_train'],
                   valid_sets = dict['lgb_test'],
                   feature_name = dict['features'],
                   early_stopping_rounds = 5)

    pickle.dump(logit,
                open(model_path + "working_logit", 'wb'))
    ml.save_model(model_path + "working_ml.txt")

def data_fulltime(dataf, estimate=1):
    dataf = dataf.copy()
    dataf = dataf[dataf['working']==1]


    if estimate == 1:
        dataf= get_dependent_var(dataf, 'fulltime')
        vars_retain = ['dep_var',
                       'fulltime_t1',
                       'working_t1',
                       'n_children',
                       'hh_youngest_age',
                       'hh_income',
                       'hh_frac_working']
    elif estimate == 0:
        vars_retain = ['fulltime_t1',
                       'working_t1',
                       'n_children',
                       'hh_youngest_age',
                       'hh_income',
                       'hh_frac_working']
    else:
        raise ValueError("0 is for simulation, 1 for estimation")

    dataf_out = dataf[vars_retain]
    return dataf_out

def estimate_fulltime(dataf):
    dataf = dataf.copy()

    dataf = data_fulltime(dataf)
    dataf.dropna(inplace=True)
    dict = _prepare_classifier(dataf)

    params_m = {'task' : 'train',
        'boosting_type' : 'gbdt',
        'n_estimators': 350,
        'objective': 'binary',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'feature_fraction': [0.9],
        'num_leaves': 31,
        'verbose': 0}

    logit = LogisticRegression().fit(dict['X_train'],
                                         dict['y_train'])

    ml = lgb.train(params_m,
                   train_set = dict['lgb_train'],
                   valid_sets = dict['lgb_test'],
                   feature_name = dict['features'],
                   early_stopping_rounds = 5)

    pickle.dump(logit,
                open(model_path + "fulltime_logit", 'wb'))
    ml.save_model(model_path + "fulltime_ml.txt")

def data_hours(dataf, estimate=1):
    dataf = dataf.copy()
    dataf = dataf[dataf['working']==1]


    if estimate == 1:
        dataf= get_dependent_var(dataf, 'hours')
        vars_retain = ['dep_var',
                       'hours_t1',
                       'hours_t2',
                       'fulltime',
                       'fulltime_t1',
                       'gross_earnings_t1',
                       'n_children',
                       'hh_youngest_age',
                       'hh_income',
                       'hh_frac_working']
    elif estimate == 0:
        vars_retain = ['hours_t1',
                       'hours_t2',
                       'fulltime',
                       'fulltime_t1',
                       'gross_earnings_t1',
                       'n_children',
                       'hh_youngest_age',
                       'hh_income',
                       'hh_frac_working']
    else:
        raise ValueError("0 is for simulation, 1 for estimation")

    dataf_out = dataf[vars_retain]
    return dataf_out

def estimate_hours(dataf):
    dataf = dataf.copy()

    dataf = data_hours(dataf)
    dataf.dropna(inplace=True)
    dict = _prepare_regressor(dataf)

    params_r = {'boosting_type' : 'gbdt',
              'n_estimators': 350,
              'objective' : 'l2',
              'metric' : 'l2',
              'num_leaves' : 31,
              'learning_rate' : 0.15,
              'feature_fraction': [0.9],
              'bagging_fraction': [0.8],
              'bagging_freq': [5],
              'verbose' : 5}

    ols = LinearRegression().fit(dict['X_train'],
                                         dict['y_train'])

    ml = lgb.train(params_r,
                   train_set = dict['lgb_train'],
                   valid_sets = dict['lgb_test'],
                   feature_name = dict['features'],
                   early_stopping_rounds = 5)

    pickle.dump(ols,
                open(model_path + "hours_ols", 'wb'))
    ml.save_model(model_path + "hours_ml.txt")

def data_earnings(dataf, estimate=1):
    dataf = dataf.copy()
    dataf = dataf[dataf['working']==1]


    if estimate == 1:
        dataf= get_dependent_var(dataf, 'gross_earnings')
        vars_retain = ['dep_var',
                       'gross_earnings_t1',
                       'gross_earnings_t2',
                       'fulltime',
                       'hours',
                       'education',
                       'n_children',
                       'hh_youngest_age',
                       'hh_income',
                       'hh_frac_working']
    elif estimate == 0:
        vars_retain = ['gross_earnings_t1',
                       'gross_earnings_t2',
                       'fulltime',
                       'hours',
                       'education',
                       'n_children',
                       'hh_youngest_age',
                       'hh_income',
                       'hh_frac_working']
    else:
        raise ValueError("0 is for simulation, 1 for estimation")

    dataf_out = dataf[vars_retain]
    return dataf_out

def estimate_earnings(dataf):
    dataf = dataf.copy()

    dataf = data_earnings(dataf)
    dataf.dropna(inplace=True)
    dict = _prepare_regressor(dataf)

    params_r = {'boosting_type' : 'gbdt',
              'n_estimators': 350,
              'objective' : 'l2',
              'metric' : 'l2',
              'num_leaves' : 31,
              'learning_rate' : 0.15,
              'feature_fraction': [0.9],
              'bagging_fraction': [0.8],
              'bagging_freq': [5],
              'verbose' : 5}

    ols = LinearRegression().fit(dict['X_train'],
                                         dict['y_train'])

    ml = lgb.train(params_r,
                   train_set = dict['lgb_train'],
                   valid_sets = dict['lgb_test'],
                   feature_name = dict['features'],
                   early_stopping_rounds = 5)

    pickle.dump(ols,
                open(model_path + "earnings_ols", 'wb'))
    ml.save_model(model_path + "earnings_ml.txt")

#
# df = pd.read_pickle(input_path + 'imputed').dropna()
# df1 = getdf(df)
#
# estimate_lfs(df1)
# estimate_working(df1)
# estimate_fulltime(df1)
# estimate_hours(df1)
# estimate_earnings(df1)
