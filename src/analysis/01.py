import numpy as np
import pandas as pd
import pathlib
import os


###############################################################################
current_week = "illmitz_est_reduced"
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)
###############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"

dici_full = pd.read_pickle(output_path + "filled_dici_illmitz_est_reduced.pkl")
dici_est = pd.read_pickle(output_path + "filled_dici_illmitz_est_reduced2.pkl")


df_full_ml = dici_full["ml"]
df_full_standard = dici_full["standard"]

df_est_ml = dici_est["ml"]
df_est_standard = dici_est["standard"]

def make_ana_df(real_dici, predicted_dici):
    df_real = real_dici["ml"]
    df_predicted_ml = predicted_dici["ml"]
    df_predicted_standard = predicted_dici["standard"]

    relevant = df_real[df_real["predicted"]==0]

    together = pd.merge(relevant, df_predicted_ml, on=["pid", "year"])
    together = pd.merge(together, df_predicted_standard, on=["pid", "year"])

    together.sort_values(by=["pid", "year"])
    together["period_ahead"] = 0

    out = pd.DataFrame()
    tmp = together
    j=0

    while len(tmp)>0:
        j+=1
        cond = tmp["pid"].duplicated(keep="first")
        not_dups = tmp[~cond]
        dups = tmp[cond]

        out = pd.concat([out, not_dups])
        dups["period_ahead"] += 1
        print("Done with period", j)
        tmp = dups.sort_values(by=["pid", "year"])

    return out


df_analysis = make_ana_df(dici_full, dici_est)



df_analysis.to_pickle(output_week + "df_analysis")
