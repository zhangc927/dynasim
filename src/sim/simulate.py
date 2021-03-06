import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

sim_path = "/Users/christianhilscher/Desktop/dynsim/src/sim/"
estimation_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/"
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"

cwd = os.getcwd()
##############################################################################
os.chdir(sim_path)
from family_module import separations, marriage, dating_market, birth, death
from work_module import sim_lfs, sim_working, sim_fulltime, sim_hours, sim_earnings, scale_data, make_hh_vars, sim_multi_employment

os.chdir(cwd)
##############################################################################
def make_cohort(dataf):
    dataf = dataf.copy()
    birthyear = dataf['year'] - dataf['age']

    condition = birthyear.isin([1954, 1955, 1956, 1957, 1958])
    dataf = dataf[condition]
    return dataf

def _moving(dataf):
    dataf = dataf.copy()

    hid_max = dataf['hid'].max()
    n_grownups = sum(dataf['age'] == 18)

    hids = np.arange((hid_max+1), (hid_max + n_grownups+1))
    dataf.loc[dataf['age'] == 18, 'hid'] = hids
    dataf.loc[dataf['age'] == 18, 'child'] = 0
    return dataf

def _return_hh_vars(dataf):
    dataf = dataf.copy()
    now = dataf['year'].max()

    dataf_new = dataf[dataf['year']==now]
    dataf_old = dataf[dataf['year']<now]

    hh_vars = ['hh_income',
               'hh_youngest_age',
               'n_people',
               'n_children']
    dataf_new.drop(hh_vars, axis=1, inplace=True)
    dataf_new = make_hh_vars(dataf_new)

    dataf_out = pd.concat([dataf_old, dataf_new], ignore_index=True)
    return dataf_out

def _shift_vars(dataf):
    dataf = dataf.copy()

    dataf['lfs_t1'] = dataf['lfs']
    dataf['working_t1'] = dataf['working']
    dataf['fulltime_t1'] = dataf['fulltime']
    dataf['hours_t2'] = dataf['hours_t1']
    dataf['gross_earnings_t2'] = dataf['gross_earnings_t1']
    dataf['employment_status_t2'] = dataf['employment_status_t1']

    dataf['hours_t1'] = dataf['hours']
    dataf['gross_earnings_t1'] = dataf['gross_earnings']
    dataf['employment_status_t1'] = dataf['employment_status']

    return dataf

def update(dataf):
    dataf = dataf.copy()

    dataf['year'] += 1
    dataf['age'] += 1
    dataf = _shift_vars(dataf)
    estimated_vars = ['birth',
                      'lfs',
                      'working',
                      'fulltime',
                      'hours',
                      'gross_earnings']

    dataf[estimated_vars] = 0
    return dataf

def run_family_module(dataf, type):
    dataf = dataf.copy()

    dataf, deaths_this_period = death(dataf)
    dataf, separations_this_period = separations(dataf)
    dataf, marriages_this_period = marriage(dataf)
    dataf, new_couples_this_period = dating_market(dataf)
    dataf, births_this_period = birth(dataf)

    out_dici={'dataf' : dataf}
    return out_dici

def run_work_module(dataf, type):
    dataf = dataf.copy()

    dataf = _return_hh_vars(dataf)

    lfs = sim_lfs(dataf,type)
    dataf['lfs'] = lfs

    # From now on always conditional on being in the labor force
    if np.sum(lfs)>0:
        working = sim_working(dataf[dataf['lfs'] == 1], type)
    else:
        working = 0
    dataf.loc[dataf['lfs'] == 1, 'working'] = working

    # From now on always conditional on being employed
    if np.sum(working)>0:
        fulltime = sim_fulltime(dataf[dataf['working'] == 1], type)
    else:
        fulltime = 0
    dataf.loc[dataf['working'] == 1, 'fulltime'] = fulltime

    if type == "ext":
        empl = sim_multi_employment(dataf)
        dataf["employment_status"] = empl
    else:
        pass

    if np.sum(working)>0:
        hours = sim_hours(dataf[dataf['working'] == 1], type)
    else:
        hours = 0
    dataf.loc[dataf['working'] == 1, 'hours'] = hours

    if np.sum(working)>0:
        earnings = sim_earnings(dataf[dataf['working'] == 1], type)
    else:
        earnings = 0
    dataf.loc[dataf['working'] == 1, 'gross_earnings'] = earnings

    return dataf
##############################################################################
##############################################################################
def predict(dataf, type):
    dataf = dataf.copy()

    dataf = update(dataf)
    dataf = run_family_module(dataf, type)['dataf']
    dataf = run_work_module(dataf, type)
    return dataf

def fill_dataf(dataf):
    dataf = dataf.copy()
    dataf['predicted'] = 0

    start = dataf['year'].min()
    end = dataf['year'].max()

    #dataf = make_cohort(dataf)

    df_base = dataf[dataf['year'] == start]
    history_dici = {'standard': df_base,
                    'ml': df_base,
                    'ext': df_base}
    base_dici = {'standard': df_base,
                 'ml': df_base,
                 'ext': df_base}
    for i in np.arange(start, end):
        df = dataf.copy()

        df_next_year = df[df['year'] == i+1]
        for type in ['standard', 'ml']:

            df_base = base_dici[type]

            have_data = df_base['pid'].isin(df_next_year['pid'])
            pids_next_year = df_base.loc[have_data, 'pid'].tolist()

            df_have_data = df_next_year[df_next_year['pid'].isin(pids_next_year)]
            df_topredict = df_base[~have_data]
            df_predicted = predict(df_topredict, type)
            df_predicted['predicted'] = 1

            df_complete = pd.concat([df_next_year,
                                     df_predicted])

            base_dici[type] = df_complete

            appended = pd.concat([history_dici[type],
                                  df_complete])
            history_dici[type] = appended

            print('Done with year', i, '. Approach: ', type)
    return history_dici
