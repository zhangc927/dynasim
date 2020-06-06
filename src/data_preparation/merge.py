import pandas as pd
import numpy as np

input_path = "/Users/christianhilscher/Desktop/dynsim/input/"

def quick_analysis(dataf):

    print("Data Types:")
    print(dataf.dtypes)
    print("Rows and Columns:")
    print(dataf.shape)
    print("Column Names:")
    print(dataf.columns)
    print("Null Values:")
    print(dataf.apply(lambda x: sum(x.isnull()) / len(dataf)))

df_08 = pd.read_pickle(input_path + "imputed08")
df_09 = pd.read_pickle(input_path + "imputed09")

#df_09 = df_09[df_09['year']<1995]

df_08.set_index(['pid', 'year'], inplace=True)
df_09.set_index(['pid', 'year'], inplace=True)

joined= pd.concat([df_08, df_09])
joined.reset_index(inplace=True)



joined.to_pickle(input_path+'joined.pkl')
###########################################
def drop_if_missing(dataf, varlist):
    dataf = dataf.copy()

    for var in varlist:
        dataf = dataf[~dataf[var].isna()]
    return dataf

l = ["pid", "year", "hid", "personweight"]

joined1 = drop_if_missing(joined, l)

def uniqueID(dataf):
    dataf = dataf.copy()

    ID_list = dataf["pid"].astype("int").astype("str") + dataf["year"].astype("int").astype("str")
    dataf["ID"] = ID_list.astype("int64")
    return dataf

abc = uniqueID(joined1)
sum(abc["ID"].duplicated())
