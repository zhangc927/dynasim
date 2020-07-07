import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from bokeh.plotting import figure, output_file, show, gridplot
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6


input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
###############################################################################

# Functions for plotting - have to outsource them
def prepare_ownplots(filled_dici, variable):

    df = filled_dici['standard']
    orig_data = df[df['predicted']==0]
    min_year = orig_data['year'].min() + 2

    df_comp = orig_data[orig_data['year']>min_year]
    #df_comp = _make_cohort(df_comp)
    df_standard = filled_dici['standard']
    df_standard = df_standard[df_standard['year']>min_year]
    df_ml = filled_dici['ml']
    df_ml = df_ml[df_ml['year']>min_year]
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


def prepare_plots(filled_dici, variable, oecd_data):

    df = filled_dici['standard']
    orig_data = df[df['predicted']==0]

    if variable in oecd_data.columns.tolist():
        oecd_data = oecd_data[variable]

        own_data = prepare_ownplots(filled_dici, variable)

        dataf = pd.concat([oecd_data, own_data], axis=1)
        dataf.rename(columns={variable: 'oecd'}, inplace=True)
    else:
        dataf = prepare_ownplots(filled_dici, variable)
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
# Cohort comparisons

def prep_countbplot(dataf):
    dataf = dataf.copy()

    counts = dataf.groupby('pid')['pid'].count()
    in_soep = np.empty(np.max(counts))

    for i in np.arange(np.max(counts)):
        in_soep[i] = sum(counts>i)

    df_counts = pd.DataFrame(in_soep, columns=['count'])
    df_counts['year'] = np.arange(1, np.max(counts)+1)
    df_counts['relative'] = df_counts['count'] / len(np.unique(dataf['pid']))

    return df_counts

def prep_sizebplot(dataf):
    dataf = dataf.copy()

    size = dataf.groupby("year")["pid"].count()
    size_df = pd.DataFrame(size)
    size_df.columns = ["size"]
    size_df.reset_index(inplace=True)

    return size_df

###############################################################################
# Actual function used
def variable_means(dataf, varlist, output_week):
    dataf = dataf.copy()
    oecd_data = prepare_oecdplots()

    # Plotting
    for variable in varlist:
        vals = prepare_plots(dataf, variable, oecd_data)

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
        plt.savefig(output_week + variable)

def countplot(dataf):
    dataf = dataf.copy()

    orig_df = dataf["standard"]
    orig_data = orig_df[orig_df['predicted']==0]

    plt_data = prep_countbplot(orig_data)
    plt = sns.barplot(x='year', y='relative',
                       data=plt_data, palette="Blues_d")
    fig = plt.get_figure()
    return fig

def sizeplot(dataf):
    dataf = dataf.copy()

    plt_data = dataf["ml"]
    plt_data = prep_sizebplot(plt_data)
    plt = sns.barplot(x='year', y='size',
                       data=plt_data, palette="Blues_d")
    plt.set_title('Number of households in simulation')
    fig = plt.get_figure()
    return fig



def data_age_employment(dataf, vals):
    dataf = dataf.copy()

    ret = pd.DataFrame()
    cut = dataf[dataf['age'].isin(np.arange(20, 60))]
    df = cut.loc[(cut['female']==vals[0]) & (cut['east']==vals[1])]

    ret['age'] = sorted(df["age"].unique())
    ret['fulltime'] = df.groupby("age")['fulltime'].mean().tolist()

    df['parttime'] = 0
    df.loc[(df['fulltime']==0)&(df['working']==1), 'parttime'] = 1
    ret['parttime'] = df.groupby("age")['parttime'].mean().tolist()

    df['unemployed'] = 0
    df.loc[(df['working']==0)&(df['lfs']==1), 'unemployed'] = 1
    ret['unemployed'] = df.groupby("age")['unemployed'].mean().tolist()

    ret['inactive'] = df.groupby("age")['lfs'].mean().tolist()
    ret['inactive'] = 1 - ret['inactive']

    return ret

def data_cohort_employment(dataf, vals):
    dataf = dataf.copy()

    dataf['birthyear'] = dataf['year'] - dataf['age']
    dataf['unemployed'] = 0
    dataf.loc[(dataf['working']==0)&(dataf['lfs']==1), 'unemployed'] = 1
    dataf['parttime'] = 0
    dataf.loc[(dataf['fulltime']==0)&(dataf['working']==1), 'parttime'] = 1
    cut = dataf[dataf['age'].isin(np.arange(20, 60))]


    final = cut.loc[(cut['female']==vals[0]) & (cut['east']==vals[1])]
    ret = pd.DataFrame()
    ret['age'] = sorted(final["birthyear"].unique())

    selected = final[['pid', 'birthyear']]
    selected.drop_duplicates(inplace=True)

    varlist = ['fulltime', 'parttime', 'working', 'unemployed']
    for v in varlist:
        values = final.groupby('pid',as_index=False)[v].sum()
        z = pd.merge(values, selected, on="pid")
        ret[v] = z.groupby('birthyear')[v].mean().tolist()
    return ret

def plot_age_employment(dataf):
    dataf = dataf.copy()

    plts = []
    subset_dict = {'male_west': [0, 0],
                  'male_east': [0, 1],
                  'female_west': [1, 0],
                  'female_east': [1, 1],}

    for name,vals in subset_dict.items():
        data = data_age_employment(dataf, vals)
        cols = data.columns.tolist()
        cols.remove("age")
        p=figure(title=name)

        for ind, var in enumerate(cols):
            p.line(y=data[var], x=data['age'], legend_label=var, color=Spectral6[ind], line_width=2)
        plts.append(p)

    p = gridplot([[plts[0], plts[1]], [plts[2], plts[3]]])
    return p

def plot_cohort_employment(dataf):
    dataf = dataf.copy()

    plts = []
    subset_dict = {'male_west': [0, 0],
                  'male_east': [0, 1],
                  'female_west': [1, 0],
                  'female_east': [1, 1],}

    for name,vals in subset_dict.items():
        data = data_cohort_employment(dataf, vals)
        cols = data.columns.tolist()
        cols.remove("age")
        p=figure(title=name)

        for ind, var in enumerate(cols):
            p.line(y=data[var], x=data['age'], legend_label=var, color=Spectral6[ind], line_width=2)
        plts.append(p)

    p = gridplot([[plts[0], plts[1]], [plts[2], plts[3]]])
    return p
