import numpy as np
import pandas as pd
import os

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

cwd = os.getcwd()
sim_path = "/Users/christianhilscher/Desktop/dynsim/src/sim/"
estimation_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/"
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"

current_week = 21
output_path = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"

os.chdir(sim_path)
from simulate08 import fill_dataf, predict

os.chdir(estimation_path)
from standard08 import getdf

os.chdir(cwd)

##############################################################################
def prepare_plots(orig_data, filled_dici, variable):


    df_comp = orig_data[orig_data['year']>1997]
    #df_comp = _make_cohort(df_comp)
    df_standard = filled_dici['standard']
    df_standard = df_standard[df_standard['year']>1997]
    df_ml = filled_dici['ml']
    df_ml = df_ml[df_ml['year']>1997]
    # if variable in ['fulltime', 'hours', 'gross_earnings']:
    #     df_comp = df_comp[df_comp['working']==1]
    #     df_standard = df_standard[df_standard['working']==1]
    #     df_ml = df_ml[df_ml['working']==1]
    # else:
    #     pass

    df_standard_cond = df_standard[df_standard['predicted']==1]
    df_ml_cond = df_ml[df_ml['predicted']==1]
    # df_ext = filled_dici['ext']
    # df_ext = df_ext[df_ext['year']>1997]
    # df_ext_cond = df_ext[df_ext['predicted']==1]

    vals = pd.DataFrame()
    vals['year'] = df_comp['year'].unique()
    vals.set_index('year', inplace=True)
    vals['no_impute'] = df_comp.groupby('year')[variable].mean().tolist()

    vals['standard'] = df_standard.groupby('year')[variable].mean().tolist()
    vals['ml'] = df_ml.groupby('year')[variable].mean().tolist()
    #vals['ext'] = df_ext.groupby('year')[variable].mean().tolist()

    vals['standard_cond'] = df_standard_cond.groupby('year')[variable].mean().tolist()
    vals['ml_cond'] = df_ml_cond.groupby('year')[variable].mean().tolist()
    #vals['ext_cond'] = df_ext_cond.groupby('year')[variable].mean().tolist()

    return vals

##############################################################################

df = pd.read_pickle(input_path + 'joined.pkl').dropna()
df1 = getdf(df)

abc = fill_dataf(df1)

var_list = ['lfs', 'working', 'fulltime', 'hours', 'gross_earnings', 'birth', 'married']
for variable in var_list:
    vals = prepare_plots(df1, abc, variable)

    fig, ax = plt.subplots(1,2, figsize=(15,5), sharey=True)
    ax[0].plot(vals['no_impute'], label='original SOEP')
    ax[0].plot(vals['standard'], label='standard')
    ax[0].plot(vals['ml'], label='ml')
    #ax[0].plot(vals['ext_cond'], label='ext')
    ax[0].title.set_text('all values: mean ' + variable)

    ax[1].plot(vals['no_impute'], label='original SOEP')
    ax[1].plot(vals['standard_cond'], label='standard')
    ax[1].plot(vals['ml_cond'], label='ml')
    #ax[1].plot(vals['ext_cond'], label='ext')
    ax[1].title.set_text('only predicted: mean ' + variable)


    for a in ax:
        a.grid(axis='x', visible=False)
        a.legend()
    plt.xticks([1995, 2000, 2005, 2010, 2015])
    fig.suptitle('Overall')
    #plt.show()
    plt.savefig(output_path + variable)


########################################################
########################################################
########################################################
