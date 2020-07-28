import numpy as np
import pandas as pd
import pickle
import os

import lightgbm as lgb
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
##############################################################################
##############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
model_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/models/"
estimation_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/"
sim_path = "/Users/christianhilscher/desktop/dynsim/src/sim/"

os.chdir(estimation_path)
from standard import getdf, data_lfs, data_working, data_fulltime, data_hours, data_earnings
from multic import data_general
os.chdir(sim_path)
##############################################################################
##############################################################################

# Functions used for predicting values
def scale_data(dataf, dep_var=None):
    dataf = dataf.copy()
    X = StandardScaler().fit_transform(np.asarray(dataf))
    return X

def _logit(X, variable):
    X= X.copy()

    X_scaled = scale_data(X, variable)
    #X_scaled['const'] = 1
    estimator  = pd.read_pickle(model_path + variable + "_logit")
    pred = estimator.predict(X_scaled)

    pred_scaled = np.zeros(len(pred))
    if np.any(pred)>0.5:
        pred_scaled[pred>0.5] = 1
    else:
        raise ValueError('Only Zeros')

    return pred_scaled

def _ols(X, variable):
    X= X.copy()

    X_scaled = scale_data(X, variable)
    #X['const'] = 1
    estimator  = pd.read_pickle(model_path + variable + "_ols")
    pred = estimator.predict(X_scaled)

    scaler = pd.read_pickle(model_path + variable + "_scaler")
    pred_scaled = scaler.inverse_transform(pred)

    return pred_scaled

def _ml(X, variable):
    X = X.copy()

    X_scaled = scale_data(X, variable)
    estimator = lgb.Booster(model_file = model_path + variable + '_ml.txt')
    pred = estimator.predict(X_scaled)

    if variable in ['hours', 'earnings']:
        pred_scaled = pred
        # Inverse transform regression results
        # scaler = pd.read_pickle(model_path + variable + "_scaler")
        # pred_scaled = scaler.inverse_transform(pred)
    else:
        # Make binary prediction to straight 0 and 1
        pred_scaled = np.zeros(len(pred))
        pred_scaled[pred>0.5] = 1

    pred_scaled[pred_scaled<0] = 0
    return pred_scaled

def _ext(X, variable):
    X = X.copy()

    X_scaled = scale_data(X, variable)
    estimator = lgb.Booster(model_file = model_path + \
                            variable + '_extended.txt')
    pred = estimator.predict(X_scaled)

    if variable == "hours":
        # Inverse transform regression results
        predictions = []

        for x in pred:
            predictions.append(np.argmax(x))
        # scaler = pd.read_pickle(model_path + variable + "_scaler")
        # pred_scaled = scaler.inverse_transform(pred)
    else:
        # Make binary prediction to straight 0 and 1
        pred_scaled = pred

    pred_scaled[pred_scaled<0] = 0
    return pred_scaled
###############################################################################
# Making household wide variables
def make_hh_vars(dataf):
    dataf = dataf.copy()
    dataf.set_index(['hid'],inplace=True)

    dataf = _hh_income(dataf)
    dataf = _hh_age_youngest(dataf)
    dataf = _hh_children(dataf)
    dataf = _hh_size(dataf)
    #dataf = _hh_fraction_working(dataf)
    dataf.reset_index(inplace=True)
    return dataf

def _hh_income(dataf):
    dataf = dataf.copy()
    earnings = dataf.groupby('hid')['gross_earnings'].sum()
    dataf['hh_income'] = earnings
    return dataf

def _hh_size(dataf):
    dataf = dataf.copy()
    size = dataf.groupby('hid')['gross_earnings'].size()
    dataf['n_people'] = size
    return dataf

def _hh_children(dataf):
    dataf = dataf.copy()
    children = dataf.groupby('hid')['child'].sum()
    dataf['n_children'] = children
    return dataf

def _hh_fraction_working(dataf):
    dataf = dataf.copy()

    dataf = _hh_size(dataf)
    dataf = _hh_children(dataf)

    total = dataf.groupby('hid')['working'].sum()
    dataf['total_working'] = total

    dataf['n_adults'] = dataf['n_people'] - dataf['n_children']
    if dataf['n_adults'].any() == 0:
        raise ValueError('No adult in HH')
    else:
        dataf['hh_frac_working'] = dataf['total_working']/dataf['n_adults']

    dataf.drop(['total_working', 'n_adults'], axis=1, inplace=True)
    return dataf

def _hh_age_youngest(dataf):
    dataf = dataf.copy()

    smallest_age = dataf.groupby('hid')['age'].min()
    dataf['hh_youngest_age'] = smallest_age
    return dataf


##############################################################################

def sim_lfs(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_lfs(dataf, estimate=0)
        predictions = _logit(X, 'lfs')
    elif type == 'ml':
        X = data_lfs(dataf, estimate=0)
        predictions = _ml(X, 'lfs')
    elif type == "ext":
        predictions = np.zeros(len(dataf))
    else:
        raise ValueError("Unkown Type")

    return predictions

def sim_working(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_working(dataf, estimate=0)
        predictions = _logit(X, 'working')
    elif type == 'ml':
        X = data_working(dataf, estimate=0)
        predictions = _ml(X, 'working')
    elif type == "ext":
        predictions = np.zeros(len(dataf))
    else:
        raise ValueError("Unkown Type")

    return predictions

def sim_fulltime(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_fulltime(dataf, estimate=0)
        predictions = _logit(X, 'fulltime')
    elif type == 'ml':
        X = data_fulltime(dataf, estimate=0)
        predictions = _ml(X, 'fulltime')
    elif type == "ext":
        predictions = np.zeros(len(dataf))
    else:
        raise ValueError("Unkown Type")

    return predictions

def sim_hours(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_hours(dataf, estimate=0)
        predictions = _ols(X, 'hours')
    elif type == 'ml':
        X = data_hours(dataf, estimate=0)
        predictions = _ml(X, 'hours')
    elif type == "ext":
        X = data_general(dataf, "hours", estimate=0)
        predictions = _ext(X, "hours")
    else:
        raise ValueError("Unkown Type")

    return predictions

def sim_earnings(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_earnings(dataf, estimate=0)
        predictions = _ols(X, 'earnings')
    elif type == 'ml':
        X = data_earnings(dataf, estimate=0)
        predictions = _ml(X, 'earnings')
    elif type == "ext":
        X = data_general(dataf, "gross_earnings", estimate=0)
        predictions = _ext(X, "gross_earnings")
    else:
        raise ValueError("Unkown Type")

    return predictions

def sim_multi_employment(dataf):
    dataf = dataf.copy()

    X = data_general(dataf, "employment_status", estimate=0)
    predictions = _ext(X, "employment_status")

    return predictions
