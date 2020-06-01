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

###############################################################################
os.chdir(sim_path)
from simulate import fill_dataf, predict
from plots import prepare_oecdplots, prepare_plots

os.chdir(estimation_path)
from standard import getdf

os.chdir(cwd)

##############################################################################


df = pd.read_pickle(input_path + 'not_imputed08').dropna()
df1 = getdf(df)

abc = fill_dataf(df1)

var_list = ['lfs', 'working', 'fulltime', 'hours', 'gross_earnings', 'birth', 'married']
oecd_data = prepare_oecdplots()

# Plotting
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
