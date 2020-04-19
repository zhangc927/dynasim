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

# Functions for estimating
def _data_general(dataf, dep_var):
    dataf = dataf.copy()

    dataf = _get_dependent_var(dataf, dep_var)

    vars_shift = ['gross_earnings',
                  'whours_actual',
                  'employment_status',
                  'hh_income']
    dataf = _shift_variables(dataf, vars_shift, 1, replace=0)

    vars_drop = ['pid',
                 'hid',
                 'orighid',
                 'age_max']

    dataf.drop(vars_drop, axis=1, inplace=True)

    dataf = _boenke_style_vars(dataf, dep_var)

    return dataf

def _boenke_style_vars(dataf, dep_var):
    dataf = dataf.copy()

    if dep_var == 'lfs':
        dataf.drop(['working',
                    'fulltime',
                    'whours_actual',
                    'gross_earnings'],
                   axis=1,
                   inplace=True)
    elif dep_var == 'working':
        dataf.drop(['fulltime',
                    'whours_actual',
                    'gross_earnings'],
                   axis=1,
                   inplace=True)
    elif dep_var == 'fulltime':
        dataf.drop(['whours_actual',
                    'gross_earnings'],
                   axis=1,
                   inplace=True)
    elif dep_var == 'whours_actual':
        dataf.drop(['gross_earnings'],
                   axis=1,
                   inplace=True)
    else:
        pass
    dataf.drop(dep_var, axis=1, inplace=True)
    return dataf

def _estimate(dataf, dep_var, type):
    dataf = dataf.copy()

    dataf = _data_general(dataf, dep_var)
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

    modl.save_model(model_path + dep_var + "_extended.txt")


df = pd.read_pickle(input_path + "imputed")
df.drop('whours_usual', axis=1,inplace=True)
# For now follwoing exactly the approach by Bönke
_estimate(df, 'lfs', 'binary')
_estimate(df, 'working', 'binary')
_estimate(df, 'fulltime', 'binary')
_estimate(df, 'whours_actual', 'regression')
_estimate(df, 'gross_earnings', 'regression')
