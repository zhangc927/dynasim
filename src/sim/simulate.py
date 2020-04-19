import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

sim_path = "/Users/christianhilscher/Desktop/dynsim/src/sim/"
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
data_prep_path = "/Users/christianhilscher/Desktop/dynsim/src/data_preparation/"
estimation_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/"
model_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/models/"

os.chdir(estimation_path)
from standard import getdf, data_lfs, data_working, data_fulltime, data_hours, data_earnings

os.chdir(data_prep_path)
from data_prep import make_hh_vars

os.chdir(sim_path)
from family_module import separations, marriage, dating_market
from work_module import sim_lfs, sim_working, sim_fulltime, sim_hours, sim_earnings, scale_data
##############################################################################
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
               'n_children',
               'hh_frac_working']
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

    dataf['hours_t1'] = dataf['hours']
    dataf['gross_earnings_t1'] = dataf['gross_earnings']

    return dataf

def _update(dataf):
    dataf = dataf.copy()

    dataf = _shift_vars(dataf)
    estimated_vars = ['lfs',
                      'working',
                      'fulltime',
                      'hours',
                      'gross_earnings']

    dataf[estimated_vars] = 0
    dataf['year'] += 1
    dataf['age'] += 1

    dataf = _moving(dataf)
    dataf = _return_hh_vars(dataf)
    return dataf

def _run_family_module(dataf):
    dataf = dataf.copy()

    dataf, separations_this_period = separations(dataf)
    dataf, marriages_this_period = marriage(dataf)
    dataf, new_couples_this_period = dating_market(dataf)

    out_dici={'dataf' : dataf,
              'separations_this_period': separations_this_period,
              'marriages_this_period': marriages_this_period,
              'new_couples_this_period': new_couples_this_period}
    return out_dici

def _run_work_module(dataf, type):
    dataf = dataf.copy()

    lfs = sim_lfs(dataf)
    dataf['lfs'] = lfs[type]

    working = sim_working(dataf[dataf['lfs'] == 1])
    dataf.loc[dataf['lfs'] == 1, 'working'] = working[type]

    fulltime = sim_fulltime(dataf[dataf['working'] == 1])
    dataf.loc[dataf['working'] == 1, 'fulltime'] = fulltime[type]

    hours = sim_hours(dataf[dataf['working'] == 1])
    dataf.loc[dataf['working'] == 1, 'hours'] = hours[type]

    earnings = sim_earnings(dataf[dataf['working'] == 1])
    dataf.loc[dataf['working'] == 1, 'gross_earnings'] = earnings[type]

    return dataf

def predict(dataf, type):
    dataf = dataf.copy()

    dataf = _update(dataf)
    dataf = _run_family_module(dataf)['dataf']
    dataf = _run_work_module(dataf, type)

    return dataf

def fill_dataf(dataf):
    dataf = dataf.copy()

    start = dataf['year'].min()
    end = dataf['year'].max()

    df_base = dataf[dataf['year'] == start]
    history_dici = {'standard': df_base,
                    'ml': df_base}
    base_dici = {'standard': df_base,
                    'ml': df_base}

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

            df_complete = pd.concat([df_next_year,
                                     df_predicted])
            base_dici[type] = df_complete

            appended = pd.concat([history_dici[type],
                                  df_complete])
            history_dici[type] = appended

            print('Done with year', i, '. Approach: ', type)
    return history_dici
##############################################################################


df = pd.read_pickle(input_path + 'imputed').dropna()
df1 = getdf(df)

abc = fill_dataf(df1)


variable = 'gross_earnings'
vals = pd.DataFrame()
df_comp = df[df['year']>1996]

vals['year'] = df_comp['year'].unique()
vals['no_impute'] = df_comp.groupby('year')[variable].mean().tolist()


for type in ['standard', 'ml']:
    vals[type] = abc[type].groupby('year')[variable].mean().tolist()

vals
vals.set_index('year', inplace=True)

fig, ax = plt.subplots()
ax.plot(vals['no_impute'], label='original SOEP')
ax.plot(vals['standard'], label='standard')
ax.plot(vals['ml'], label='ml')
ax.legend()
plt.title('fraction '+variable)
plt.show()
