import pandas as pd
import numpy as np

input_path = "/Users/christianhilscher/Desktop/dynsim/input/"

df_08 = pd.read_pickle(input_path + "imputed08")
df_09 = pd.read_pickle(input_path + "imputed09")

df_09 = df_09[df_09['year']<1995]

df_08.set_index(['pid', 'year'], inplace=True)
df_09.set_index(['pid', 'year'], inplace=True)

joined= pd.concat([df_08, df_09])
joined.reset_index(inplace=True)

joined.to_pickle(input_path+'joined.pkl')
