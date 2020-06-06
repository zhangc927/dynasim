import numpy as np
import pandas as pd
import pathlib
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

###############################################################################
current_week = 22
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)
###############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"
plot_path = "/Users/christianhilscher/Desktop/dynsim/src/plotting/"
os.chdir(plot_path)
###############################################################################
from aux_plots import prepare_oecdplots, prepare_plots


###############################################################################
# Comparing variable means
dataf = pd.read_pickle(output_path + "filled_dici.pkl")

var_list = ['lfs', 'working', 'fulltime', 'hours', 'gross_earnings', 'birth', 'married']
oecd_data = prepare_oecdplots()

# Plotting
for variable in var_list:
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


########################################################
########################################################
########################################################
