import os
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import lightgbm as lgb

import matplotlib.pyplot as plt

data_path = "/Users/christianhilscher/Desktop/dynsim/src/data_preparation/"
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
model_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/models/"

# General functions
def _shift_variables(dataf, vars, delta, replace=1):
    dataf = dataf.copy()

    dataf.sort_values(by=["pid", "year"])
    dups = dataf["pid"].duplicated()

    col_names = [i+"_t-"+str(delta) for i in vars]
    dataf[col_names] = dataf[vars].shift(-(delta))

    dups = dups.shift(-(delta))[:-(delta)]
    dups.reset_index(drop=True, inplace=True)

    dataf = dataf[:-(delta)]
    dataf.reset_index(drop=True, inplace=True)
    dataf = dataf[dups]

    return dataf

def _get_dependent_var(dataf, dep_var):
    dataf = dataf.copy()

    dataf.sort_values(by=["pid", "year"])
    dups = dataf["pid"].duplicated()

    dataf["dep_var"] = dataf[dep_var].shift(-1)

    dups = dups.shift(-1)[:-1]
    dups.reset_index(drop=True)

    dataf = dataf[:-1]
    dataf.reset_index(drop=True, inplace=True)
    dataf = dataf[dups]

    return dataf

def _prepare_classifier(dataf):
    dataf = dataf.copy()

    y = dataf['dep_var']
    X = dataf.drop('dep_var', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

    X_train = StandardScaler().fit_transform(np.asarray(X_train))
    X_test = StandardScaler().fit_transform(np.asarray(X_test))

    feature_names = X.columns.tolist()

    # For ML part:
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_test = lgb.Dataset(X_test, y_test, free_raw_data=False)

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

        X_train = StandardScaler().fit_transform(np.asarray(X_train))
        X_test = StandardScaler().fit_transform(np.asarray(X_test))
        y_train = StandardScaler().fit_transform(np.asarray(y_train).reshape(-1,1))

        # Saving the scaler of the test data to convert the predicted values again
        y_test_scaler = StandardScaler().fit(np.asarray(y_test).reshape(-1,1))
        y_test = y_test_scaler.transform(np.asarray(y_test).reshape(-1,1))

        feature_names = X.columns.tolist()
        y_test = np.ravel(y_test)
        y_train = np.ravel(y_train)

        # For ML part:
        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        lgb_test = lgb.Dataset(X_test, y_test, free_raw_data=False)

        out_dici = {'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'scaler': y_test_scaler,
                    'lgb_train': lgb_train,
                    'lgb_test': lgb_test,
                    'features': feature_names}
        return out_dici

# Functions for the individual regressions
def _data_lfs(dataf):
    dataf = dataf.copy()

    dataf = _get_dependent_var(dataf, 'lfs')

    vars_shift = ['lfs', 'working']
    dataf = _shift_variables(dataf, vars_shift, 1, replace=0)

    vars_retain = ['dep_var',
                   'lfs_t-1',
                   'working_t-1',
                   'n_children',
                   'hh_youngest_age',
                   'hh_income',
                   'hh_frac_working']

    dataf_out = dataf[vars_retain]
    return dataf_out

def estimate_lfs(dataf):
    dataf = dataf.copy()

    dataf = _data_lfs(dataf)
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

def _data_working(dataf):
    dataf = dataf.copy()

    # Taking only those who are in the labor force in this period
    dataf = dataf[dataf['lfs'] == 1].reset_index(drop=True)
    dataf = _get_dependent_var(dataf, 'working')

    vars_shift = ['working', 'fulltime']
    dataf = _shift_variables(dataf, vars_shift, 1, replace=0)

    vars_retain = ['dep_var',
                   'working_t-1',
                   'fulltime_t-1',
                   'n_children',
                   'hh_youngest_age',
                   'hh_income',
                   'hh_frac_working']

    dataf_out = dataf[vars_retain]
    return dataf_out

def estimate_working(dataf):
    dataf = dataf.copy()

    dataf = _data_working(dataf)
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

def _data_fulltime(dataf):
    dataf = dataf.copy()

    # Taking only those who are working in this period
    dataf = dataf[dataf['working'] == 1].reset_index(drop=True)
    dataf = _get_dependent_var(dataf, 'fulltime')

    vars_shift = ['working', 'fulltime']
    dataf = _shift_variables(dataf, vars_shift, 1, replace=0)

    vars_retain = ['dep_var',
                   'working_t-1',
                   'fulltime_t-1',
                   'n_children',
                   'hh_youngest_age',
                   'hh_income',
                   'hh_frac_working']

    dataf_out = dataf[vars_retain]
    return dataf_out

def estimate_fulltime(dataf):
    dataf = dataf.copy()

    dataf = _data_fulltime(dataf)
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

def _data_hours(dataf):
    dataf = dataf.copy()

    # Taking only those who are working in this period
    dataf = dataf[dataf['working'] == 1].reset_index(drop=True)
    dataf = _get_dependent_var(dataf, 'whours_actual')

    vars_shift = ['whours_actual']
    dataf = _shift_variables(dataf, vars_shift, 2, replace=0)

    vars_shift2 = ['whours_actual', 'fulltime', 'gross_earnings']
    dataf = _shift_variables(dataf, vars_shift2, 1, replace=0)

    vars_retain = ['dep_var',
                   'whours_actual_t-1',
                   'whours_actual_t-2',
                   'fulltime_t-1',
                   'gross_earnings_t-1',
                   'n_children',
                   'hh_youngest_age',
                   'hh_income',
                   'hh_frac_working']

    dataf_out = dataf[vars_retain]
    return dataf_out

def estimate_hours(dataf):
    dataf = dataf.copy()

    dataf = _data_hours(dataf)
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

def _data_earnings(dataf):
    dataf = dataf.copy()

    # Taking only those who are working in this period
    dataf = dataf[dataf['working'] == 1].reset_index(drop=True)
    dataf = _get_dependent_var(dataf, 'gross_earnings')

    vars_shift = ['gross_earnings']
    dataf = _shift_variables(dataf, vars_shift, 2, replace=0)

    vars_shift2 = ['gross_earnings']
    dataf = _shift_variables(dataf, vars_shift2, 1, replace=0)

    vars_retain = ['dep_var',
                   'gross_earnings_t-1',
                   'gross_earnings_t-2',
                   'fulltime',
                   'education',
                   'married',
                   'whours_actual',
                   'hh_youngest_age']

    dataf_out = dataf[vars_retain]
    return dataf_out

def estimate_earnings(dataf):
    dataf = dataf.copy()

    dataf = _data_earnings(dataf)
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



df = pd.read_pickle(input_path + "imputed")

#
# estimate_lfs(df)
# estimate_working(df)
# estimate_fulltime(df)
# estimate_hours(df)
# estimate_earnings(df)
