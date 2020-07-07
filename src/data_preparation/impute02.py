import os
import numpy as np
import pandas as pd
from numba import njit
import importlib

data_path = "/Users/christianhilscher/Desktop/dynsim/src/data_preparation/"
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"

os.chdir(data_path)

df = pd.read_pickle(input_path + "illmitz01")

df.loc[(df["hours"].isna())&(df["lfs"]==0), "hours"] = 0
df.loc[(df["gross_earnings"].isna())&(df["lfs"]==0), "hours"] = 0



m = list(zip(df["year"], df["hid"]))
m1 = pd.MultiIndex.from_tuples(m, names=["hid", "year"])
df.set_index(m1, inplace=True)
migsum = df.groupby(level=["hid", "year"])["migback"].sum()
df["migsum"] = migsum

df.loc[(df["migback"].isna())&(df["migsum"]>0),"migback"] = 1
df.loc[(df["migback"].isna())&(df["migsum"]==0),"migback"] = 0

df.reset_index(drop=True, inplace=True)

df.loc[(df["working"]==0)&(df["gross_earnings"].isna()), "gross_earnings"] = 0

#df.loc[df['hours']>0, 'working'] = 1



# Setting hours to zero if not participating on labor market
df.loc[df['lfs'] == 0, 'hours'] = 0

df.loc[df['age']>18, 'child'] = 0


df['birth'] = 0
df.loc[(df['hh_youngest_age']==0) & (df['female']==1) & (df['child']==0), 'birth'] = 1

df.shape
df = df[df["year"]>1994]

df.to_pickle(input_path + "illmitz10_reduced")

df_ana = df[["year", "working", "lfs"]]
df_ana.loc[df_ana["lfs"]==1,:].groupby("year")["working"].mean()
