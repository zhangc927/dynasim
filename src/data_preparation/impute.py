import os
import numpy as np
import pandas as pd
from numba import njit
import importlib

data_path = "/Users/christianhilscher/Desktop/dynsim/src/data_preparation/"
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"

os.chdir(data_path)
from cleaning import SOEP_to_df


df = pd.read_pickle(input_path + "new_df.pkl")
df1 = SOEP_to_df(df)

df1['education'] = df1['education'].fillna(df1.groupby('pid')['education'].transform('max'))

df1['heizkosten'] = df1['heizkosten'].fillna(df1.groupby('pid')['heizkosten'].transform('median'))

df1['bruttokaltmiete'] = df1['bruttokaltmiete'].fillna(df1.groupby('pid')['bruttokaltmiete'].transform('median'))

df1['gross_earnings'] = df1['gross_earnings'].fillna(df1.groupby('pid')['gross_earnings'].transform('median'))

df1['tenure'].fillna(0)

df1.loc[df1['hours']>0,'working'] = 1
# Setting hours to zero if not participating on labor market
df1.loc[df1['lfs'] == 0, 'hours'] = 0

df1.loc[df1['age']>18, 'child'] = 0


df1['birth'] = 0
df1.loc[(df1['hh_youngest_age']==0) & (df1['female']==1) & (df1['child']==0), 'birth'] = 1


df1.to_pickle(input_path + 'imputed09')
