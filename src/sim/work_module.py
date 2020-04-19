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
os.chdir(sim_path)
##############################################################################
##############################################################################

def scale_data(dataf, dep_var=None):
    dataf = dataf.copy()

    if dep_var is not None:
        toscale = dataf[dep_var]
        X = dataf
        scaler = StandardScaler().fit(np.asarray(toscale).reshape(-1, 1))
        X = StandardScaler().fit_transform(np.asarray(X))
    else:
        X = StandardScaler().fit_transform(np.asarray(dataf))
        scaler = 0
    return X, scaler
##############################################################################
def sim_lfs(dataf):
    dataf = dataf.copy()

    X = data_lfs(dataf, estimate=0)
    X_scaled, scaler = scale_data(X)

    logit_estimator = pd.read_pickle(model_path + 'lfs_logit')
    ml_estimator = lgb.Booster(model_file = model_path + 'lfs_ml.txt')

    logit_pred = logit_estimator.predict(X)
    ml_pred = ml_estimator.predict(X_scaled)


    for i in [logit_pred, ml_pred]:
        i[i<0.1] = 0
        i[i>0.1] = 1

    out_dict = {'standard': logit_pred,
                'ml': ml_pred}
    return out_dict

def sim_working(dataf):
    dataf = dataf.copy()

    X = data_working(dataf, estimate=0)
    X_scaled, scaler = scale_data(X)

    logit_estimator = pd.read_pickle(model_path + 'working_logit')
    ml_estimator = lgb.Booster(model_file = model_path + 'working_ml.txt')

    logit_pred = logit_estimator.predict(X)
    ml_pred = ml_estimator.predict(X_scaled)


    for i in [logit_pred, ml_pred]:
        i[i<0.1] = 0
        i[i>0.1] = 1

    out_dict = {'standard': logit_pred,
                'ml': ml_pred}
    return out_dict

def sim_fulltime(dataf):
    dataf = dataf.copy()

    X = data_fulltime(dataf, estimate=0)
    X_scaled, scaler = scale_data(X)

    logit_estimator = pd.read_pickle(model_path + 'fulltime_logit')
    ml_estimator = lgb.Booster(model_file = model_path + 'fulltime_ml.txt')

    logit_pred = logit_estimator.predict(X)
    ml_pred = ml_estimator.predict(X_scaled)


    for i in [logit_pred, ml_pred]:
        i[i<0.1] = 0
        i[i>0.1] = 1

    out_dict = {'standard': logit_pred,
                'ml': ml_pred}
    return out_dict

def sim_hours(dataf):
    dataf = dataf.copy()

    X = data_hours(dataf, estimate=0)
    X_scaled, scaler = scale_data(X, 'hours_t1')

    ols_estimator = pd.read_pickle(model_path + 'hours_ols')
    ml_estimator = lgb.Booster(model_file = model_path + 'hours_ml.txt')

    ols_pred = ols_estimator.predict(X)
    ml_pred = ml_estimator.predict(X_scaled)
    ml_pred_scaled = scaler.inverse_transform(ml_pred)

    out_dict = {'standard': ols_pred,
                'ml': ml_pred_scaled}
    return out_dict

def sim_earnings(dataf):
    dataf = dataf.copy()

    X = data_hours(dataf, estimate=0)
    X_scaled, scaler = scale_data(X, 'gross_earnings_t1')

    ols_estimator = pd.read_pickle(model_path + 'earnings_ols')
    ml_estimator = lgb.Booster(model_file = model_path + 'earnings_ml.txt')

    ols_pred = ols_estimator.predict(X)
    ml_pred = ml_estimator.predict(X_scaled)
    ml_pred_scaled = scaler.inverse_transform(ml_pred)

    out_dict = {'standard': ols_pred,
                'ml': ml_pred_scaled}
    return out_dict
