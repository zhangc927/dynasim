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

if not os.path.exists(output_path):
    os.makedirs(output_path)

os.chdir(sim_path)
from simulate08 import fill_dataf, predict

os.chdir(estimation_path)
from standard08 import getdf

os.chdir(cwd)
###############################################################################
###############################################################################

def prep_countbplot(dataf):
    dataf = dataf.copy()

    counts = dataf.groupby('pid')['pid'].count()
    in_soep = np.empty(np.max(counts))

    for i in np.arange(np.max(counts)):
        in_soep[i] = sum(counts>i)

    df_counts = pd.DataFrame(in_soep, columns=['count'])
    df_counts['year'] = np.arange(1, np.max(counts)+1)
    df_counts['relative'] = df_counts['count'] / len(np.unique(df_use['pid']))

    return df_counts

###############################################################################
###############################################################################
df = pd.read_pickle(input_path + 'not_imputed08').dropna()
df1 = getdf(df)

df_use = df1.copy()




a = prep_countbplot(df_use)
b = sns.barplot(x='year', y='relative', data=a, palette="Blues_d")
