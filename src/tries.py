import os
import numpy as np
import pandas as pd
import importlib
from sklearn.neighbors import NearestNeighbors

sim_path = "/Users/christianhilscher/Desktop/dynsim/src/sim/"
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
os.chdir(sim_path)

from data_prep import SOEP_to_df
import family_module


df_base = pd.read_stata(input_path + 'SOEP_prepared_costs_2019-11-27_restricted.dta')

df = SOEP_to_df(df_base)
df_use = df[df['year'] == 2000].dropna()


df_use['age']

importlib.reload(family_module)
