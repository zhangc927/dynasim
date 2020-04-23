import numpy as np
import pandas as pd
import pickle
import os

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
##############################################################################
##############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
model_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/models/"
estimation_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/"
sim_path = "/Users/christianhilscher/desktop/dynsim/src/sim/"

os.chdir(estimation_path)
from standard import getdf, data_lfs, data_working, data_fulltime, data_hours, data_earnings
from extended import data_general
os.chdir(sim_path)
##############################################################################
##############################################################################

def scale_data(dataf, dep_var=None):
    dataf = dataf.copy()


    if dep_var in ['hours_t1', 'gross_earnings_t1']:
        toscale = dataf[dep_var]
        X = dataf
        scaler = StandardScaler().fit(np.asarray(toscale).reshape(-1, 1))
        X = StandardScaler().fit_transform(np.asarray(X))
    else:
        X = StandardScaler().fit_transform(np.asarray(dataf))
        scaler = 0
    return X, scaler
##############################################################################

def _logit(X, variable):
    X= X.copy()

    estimator  = pd.read_pickle(model_path + variable + "_logit")
    pred = estimator.predict(X)

    return pred

def _ols(X, variable):
    X= X.copy()

    estimator  = pd.read_pickle(model_path + variable + "_ols")
    pred = estimator.predict(X)

    return pred

def _ml(X, variable):
    X = X.copy()

    shifted_var = variable+"_t1"
    X_scaled, scaler = scale_data(X, shifted_var)
    estimator = lgb.Booster(model_file = model_path + variable + '_ml.txt')
    pred = estimator.predict(X_scaled)

    if shifted_var in ['hours_t1', 'gross_earnings_t1']:
        # Inverse transform regression results
        pred_scaled = scaler.inverse_transform(pred)
    else:
        # Make binary prediction to straight 0 and 1
        pred_scaled = np.zeros(len(pred))
        pred_scaled[pred>0.5] = 1

    return pred_scaled

def _ext(X, variable):
    X = X.copy()

    shifted_var = variable+"_t1"
    X_scaled, scaler = scale_data(X, shifted_var)
    estimator = lgb.Booster(model_file = model_path + variable + '_extended.txt')
    pred = estimator.predict(X_scaled)

    if shifted_var in ['hours_t1', 'gross_earnings_t1']:
        # Inverse transform regression results
        pred_scaled = scaler.inverse_transform(pred)
    else:
        # Make binary prediction to straight 0 and 1
        pred_scaled = np.zeros(len(pred))
        pred_scaled[pred>0.5] = 1

    return pred_scaled

##############################################################################

def sim_lfs(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_lfs(dataf, estimate=0)
        predictions = _logit(X, 'lfs')
    elif type == 'ml':
        X = data_lfs(dataf, estimate=0)
        predictions = _ml(X, 'lfs')
    elif type == 'ext':
        X = data_general(dataf, estimate=0)
        predictions = _ext(X, 'lfs')

    return predictions

def sim_working(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_working(dataf, estimate=0)
        predictions = _logit(X, 'working')
    elif type == 'ml':
        X = data_working(dataf, estimate=0)
        predictions = _ml(X, 'working')
    elif type == 'ext':
        X = data_general(dataf, estimate=0)
        predictions = _ext(X, 'working')

    return predictions

def sim_fulltime(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_fulltime(dataf, estimate=0)
        predictions = _logit(X, 'fulltime')
    elif type == 'ml':
        X = data_fulltime(dataf, estimate=0)
        predictions = _ml(X, 'fulltime')
    elif type == 'ext':
        X = data_general(dataf, estimate=0)
        predictions = _ext(X, 'fulltime')

    return predictions

def sim_hours(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_hours(dataf, estimate=0)
        predictions = _ols(X, 'hours')
    elif type == 'ml':
        X = data_hours(dataf, estimate=0)
        predictions = _ml(X, 'hours')
    elif type == 'ext':
        X = data_general(dataf, estimate=0)
        predictions = _ext(X, 'hours')

    return predictions

def sim_earnings(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_earnings(dataf, estimate=0)
        predictions = _ols(X, 'earnings')
    elif type == 'ml':
        X = data_earnings(dataf, estimate=0)
        predictions = _ml(X, 'earnings')
    elif type == 'ext':
        X = data_general(dataf, estimate=0)
        predictions = _ext(X, 'earnings')

    return predictions
