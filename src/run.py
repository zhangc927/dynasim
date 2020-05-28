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

current_week = 22
output_path = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"

os.chdir(sim_path)
from simulate import fill_dataf, predict

os.chdir(estimation_path)
from standard import getdf

os.chdir(cwd)

##############################################################################
def prepare_ownplots(orig_data, filled_dici, variable):


    df_comp = orig_data[orig_data['year']>1997]
    #df_comp = _make_cohort(df_comp)
    df_standard = filled_dici['standard']
    df_standard = df_standard[df_standard['year']>1997]
    df_ml = filled_dici['ml']
    df_ml = df_ml[df_ml['year']>1997]
    #df_ext = filled_dici['ext']
    #df_ext = df_ext[df_ext['year']>1997]
    # if variable in ['fulltime', 'hours', 'gross_earnings']:
    #     df_comp = df_comp[df_comp['working']==1]
    #     df_standard = df_standard[df_standard['working']==1]
    #     df_ml = df_ml[df_ml['working']==1]
    # else:
    #     pass

    df_standard_cond = df_standard[df_standard['predicted']==1]
    df_ml_cond = df_ml[df_ml['predicted']==1]
    #df_ext_cond = df_ext[df_ext['predicted']==1]

    vals = pd.DataFrame()
    vals['year'] = df_comp['year'].unique()
    vals.set_index('year', inplace=True)
    vals['no_impute'] = weighted_average(df_comp, variable, 'personweight', 'year')
    #vals['no_impute'] = df_comp.groupby('year')[variable].mean()

    vals['standard'] = df_standard.groupby('year')[variable].mean().tolist()
    vals['ml'] = df_ml.groupby('year')[variable].mean().tolist()
    #vals['ext'] = df_ext.groupby('year')[variable].mean().tolist()

    vals['standard_cond'] = df_standard_cond.groupby('year')[variable].mean().tolist()
    vals['ml_cond'] = df_ml_cond.groupby('year')[variable].mean().tolist()
    #vals['ext_cond'] = df_ext_cond.groupby('year')[variable].mean().tolist()

    return vals

def prepare_oecdplots():
    # Time from 1986-2018

    lfs_oecd = pd.read_csv(input_path + "OECD/lfs.csv")
    lfs_oecd = lfs_oecd[['Series', 'Time', 'Value']]
    lfs_oecd.rename(columns={'Time': 'year'}, inplace=True)
    lfs_oecd.set_index('year', inplace=True)

    lfs_oecd['Series'].unique()
    lfs_oecd['lfs'] = lfs_oecd.loc[lfs_oecd['Series']=='Labour Force', 'Value'] / lfs_oecd.loc[lfs_oecd['Series']=='Population', 'Value']

    lfs_oecd['working'] = lfs_oecd.loc[lfs_oecd['Series']=='Employment', 'Value'] / lfs_oecd.loc[lfs_oecd['Series']=='Population', 'Value']

    oecd_plot = lfs_oecd[0:33]
    oecd_plot.drop('Value', axis=1, inplace=True)
    oecd_plot.drop('Series', axis=1, inplace=True)

    hrs_oecd = pd.read_csv(input_path + "OECD/hrs.csv")
    hrs_oecd = hrs_oecd[['Time', 'Value']]
    hrs_oecd.rename(columns={'Time': 'year'}, inplace=True)
    hrs_oecd.set_index('year', inplace=True)
    hrs_oecd.rename(columns={'Value': 'hours'}, inplace=True)

    oecd_plot = pd.concat([oecd_plot, hrs_oecd], axis=1)
    #oecd_plot.reset_index(drop=True, inplace=True)

    return oecd_plot


def prepare_plots(orig_data, filled_dici, variable, oecd_data):

    if variable in oecd_data.columns.tolist():
        oecd_data = oecd_data[variable]

        own_data = prepare_ownplots(orig_data, filled_dici, variable)

        dataf = pd.concat([oecd_data, own_data], axis=1)
        dataf.rename(columns={variable: 'oecd'}, inplace=True)
    else:
        dataf = prepare_ownplots(orig_data, filled_dici, variable)
    dataf = dataf.dropna()
    return dataf

def weighted_average(df, data_col, weight_col, by_col):

    df['_data_times_weight'] = df[data_col] * df[weight_col]
    df['_weight_where_notnull'] = df[weight_col] * pd.notnull(df[data_col])
    g = df.groupby(by_col)
    result = g['_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
    del df['_data_times_weight'], df['_weight_where_notnull']
    return result
##############################################################################

df = pd.read_pickle(input_path + 'not_imputed08').dropna()
df1 = getdf(df)

abc = fill_dataf(df1)

var_list = ['lfs', 'working', 'fulltime', 'hours', 'gross_earnings', 'birth', 'married']
oecd_data = prepare_oecdplots()


for variable in var_list:
    vals = prepare_plots(df1, abc, variable, oecd_data)

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

    if variable in oecd_data:
        ax[0].plot(vals['oecd'], label='oecd')
        ax[1].plot(vals['oecd'], label='oecd')
    else:
        pass

    for a in ax:
        a.grid(axis='x', visible=False)
        a.legend()
    #plt.xticks([1995, 2000, 2005, 2010, 2015])
    fig.suptitle('Overall')
    #plt.show()
    plt.savefig(output_path + variable)


########################################################
########################################################
########################################################
