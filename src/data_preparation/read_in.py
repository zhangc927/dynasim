import numpy as np
import pandas as pd
import os
import pickle

def quick_analysis(dataf):

    print("Data Types:")
    print(dataf.dtypes)
    print("Rows and Columns:")
    print(dataf.shape)
    print("Column Names:")
    print(dataf.columns)
    print("Null Values:")
    print(dataf.apply(lambda x: sum(x.isnull()) / len(dataf)))
##############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
model_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/models/04/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"
estimation_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/"
sim_path = "/Users/christianhilscher/desktop/dynsim/src/sim/"

os.chdir("/Users/christianhilscher/desktop/dynsim/src/data_preparation/")
from cleaning import SOEP_to_df
from data_prep import SOEP_to_df_old


# Making new dataset from original SOEP
df_pgen = pd.read_stata("/Volumes/B/soep.v35/STATA_DEEN_v35/Stata/pgen.dta")
df_hgen = pd.read_stata("/Volumes/B/soep.v35/STATA_DEEN_v35/Stata/hgen.dta")
df_ppathl = pd.read_stata("/Volumes/B/soep.v35/STATA_DEEN_v35/Stata/ppathl.dta")
df_hpathl = pd.read_stata("/Volumes/B/soep.v35/STATA_DEEN_v35/Stata/hpathl.dta")
df_hbrutto = pd.read_stata("/Volumes/B/soep.v35/STATA_DEEN_v35/Stata/hbrutto.dta")

df_pgen = df_pgen[["hid", "pid" , "syear", "pglabgro", "pgemplst", "pglfs", "pgtatzeit", "pgerwzeit", "pgpsbil", "pgfamstd"]]
df_hgen = df_hgen[["hid", "syear", "hgheat", "hgrent", "hgtyp1hh"]]


df_ppathl = df_ppathl[["hid", "pid", "syear", "sex", "gebjahr", "migback", "phrf"]]
df_hpathl = df_hpathl[["hid", "syear", "hhrf"]]
df_hbrutto = df_hbrutto[["hid", "syear", "bula"]]

# Merging datasets from SOEP
person_df = pd.merge(df_pgen, df_ppathl, on=["pid", "syear"], how="left")
hh_df = pd.merge(df_hgen, df_hpathl, on=["hid", "syear"], how="left")
hh_df = pd.merge(hh_df, df_hbrutto, on=["hid", "syear"], how="left")

person_df.drop("hid_y", axis=1, inplace=True)
person_df.rename(columns={"hid_x": "hid"}, inplace=True)

full_df = pd.merge(person_df, hh_df, on=["hid", "syear"])
full_df.columns.tolist()

try1 = SOEP_to_df(full_df)
try1.drop("tenure", axis=1, inplace=True)
try1.drop("heizkosten", axis=1, inplace=True)
try1.drop("bruttokaltmiete", axis=1, inplace=True)

names_list = try1.columns.tolist()

# Reading in old dataset
old_df = pd.read_pickle("/Users/christianhilscher/Desktop/dynsim/input/old/full")
orig_df = SOEP_to_df_old(old_df)


finish = pd.merge(try1, orig_df, on=["pid", "year"], how="outer")
finish.shape
finish.columns.tolist()


names_list.remove("pid")
names_list.remove("year")
for name in names_list:
    finish[name] = finish[name+"_y"]
    finish[name].fillna(finish[name+"_x"], inplace=True)

names_list.append("year")
names_list.append("pid")
names_list
finish_small = finish[names_list]


finish_small.to_pickle(input_path + "illmitz01")
