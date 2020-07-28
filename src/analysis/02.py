import numpy as np
import pandas as pd
import pathlib
import os


###############################################################################
current_week = "29"
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)
###############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"

dici_full = pd.read_pickle(output_path + "filled_dici_illmitz_est_reduced.pkl")
dici_est = pd.read_pickle(output_path + "filled_dici_illmitz_est_reduced2.pkl")


df_full_ml = dici_full["ml"]
df_full_standard = dici_full["standard"]
df_full_ext = dici_full["ext"]

df_est_ml = dici_est["ml"]
df_est_standard = dici_est["standard"]
df_est_ext = dici_est["ext"]




real_dici = dici_full
predicted_dici = dici_est

df_real = real_dici["ml"]
df_predicted_ml = predicted_dici["ml"]
df_predicted_standard = predicted_dici["standard"]
df_predicted_ext = predicted_dici["ext"]

relevant = df_real[df_real["predicted"]==0]
together = pd.merge(relevant, df_predicted_ml, on=["pid", "year"], suffixes=["_real", "_ml"])
together = pd.merge(together, df_predicted_standard, on=["pid", "year"], suffixes=["", "_standard"])
together = pd.merge(together, df_predicted_ext, on=["pid", "year"], suffixes=["", "_ext"])
together.columns.tolist()


together.sort_values(by=["pid", "year"])
together["period_ahead"] = 0

out = pd.DataFrame()
tmp = together
j=0

together.columns.tolist()
cond = tmp["pid"].duplicated(keep="first")
not_dups = tmp[~cond]
dups = tmp[cond]







while len(tmp)>0:
    j+=1
    cond = tmp["pid"].duplicated(keep="first")
    not_dups = tmp[~cond]
    dups = tmp[cond]

    out = pd.concat([out, not_dups])
    dups["period_ahead"] += 1
    print("Done with period", j)
    tmp = dups.sort_values(by=["pid", "year"])

out.columns.tolist()
out.to_pickle(output_week + "df_analysis")
