import pandas as pd
import numpy as np

input_path = "/Users/christianhilscher/Desktop/dynsim/input/"

# Mortality
xls = pd.ExcelFile(input_path + "/parameters/periodensterbetafeln-bundeslaender-5126204177005.xlsx"
)
tmp_mortality_men = xls.parse("Deutschland männlich", header=0, usecols="A,D", index_col=0).iloc[14:115]
tmp_mortality_women = xls.parse("Deutschland weiblich", header=0, usecols="A,D", index_col=0).iloc[14:115]
mortality = tmp_mortality_men.join(tmp_mortality_women, lsuffix="_men", rsuffix="_women")
mortality.columns = ["Men", "Women"]
mortality.index.rename("Age", inplace=True)
mortality = mortality.apply(pd.to_numeric)

mortality.to_csv(input_path + "mortality")

# Age-specific fertility rates
fertility_raw = pd.read_csv(input_path + "/parameters/kohortenspezifische_geburtenziffern_12612-0012.csv",
    sep=";",
    header=7,
    decimal=",",
)
fertility_raw.drop(fertility_raw.index[range(35, 39)], axis=0, inplace=True)
fertility_raw.rename(columns={"Unnamed: 0": "Age"}, inplace=True)
fertility_raw["Age"] = fertility_raw["Age"].apply(
    lambda x: int(x.replace(" Jahre", ""))
)
fertility_raw.set_index("Age", inplace=True)
# I use the age-specific fertility rates from the 1968 cohort, the latest currently available
fertility = fertility_raw["1968"]
fertility.to_csv(input_path + "fertility")
