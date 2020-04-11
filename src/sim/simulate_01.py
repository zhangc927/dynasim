import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sim_path = "/Users/christianhilscher/Desktop/dynsim/src/sim/"
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
os.chdir(sim_path)

from data_prep import SOEP_to_df


df_base = pd.read_stata(input_path + 'SOEP_prepared_costs_2019-11-27_restricted.dta')

df = SOEP_to_df(df_base)
df1 = df[df['year'] == 2000]
sum(df1.duplicated('pid'))



df_begin = df1.dropna().copy()
dataf = df1.dropna().copy()


eligible = (dataf['in_couple'] == 0) & (dataf['child'] == 0) & (dataf['n_people'] - dataf['n_children'] == 1)
female_singles = dataf[eligible & (dataf['female'] == 1)]
male_singles = dataf[eligible & (dataf['female'] == 0)]

not_single = dataf[~eligible]

new_couples = round(0.2 * min(len(female_singles),
                              len(male_singles)))

lucky_guys = male_singles.sample(new_couples)
neigh = NearestNeighbors(n_neighbors=5)
partners = female_singles.copy()
happy_girls = pd.DataFrame()
for i in np.arange(len(lucky_guys)):
    neigh.fit(partners[['age',
                              'gross_earnings',
                              'education',
                              'employment_status']])
    p1 = lucky_guys.iloc[i,:]
    partner_choice = neigh.kneighbors((p1[['age',
                                           'gross_earnings',
                                           'education',
                                           'employment_status']].to_numpy().reshape(1,-1)))
    partner = np.random.choice(np.ravel(partner_choice[1]), 1)
    happy_girls = pd.concat([happy_girls, partners.iloc[partner,:]])
    partners.drop(partners.iloc[partner,:].index, inplace=True)



#happy_girls1 = happy_girls.copy()
#lucky_guys1 = lucky_guys.copy()


lucky_guys.loc[:,'hid'] = happy_girls['hid'].tolist()
lucky_guys.loc[:,'east'] = happy_girls['east'].tolist()
lucky_guys.loc[:,'hhweight'] = happy_girls['hhweight'].tolist()
lucky_guys.loc[:,'in_couple'] = 1
happy_girls.loc[:,'in_couple'] = 1

dici = {'all_female': female_singles,
            'all_male': male_singles,
            'happy_girls': happy_girls,
            'lucky_guys': lucky_guys}

unlucky_guys = dici['all_male'][dici['all_male'].index.isin(dici['lucky_guys'].index) == False]

unhappy_girls = dici['all_female'][dici['all_female'].index.isin(dici['happy_girls'].index) == False]


len(unlucky_guys) + len(lucky_guys) == len(male_singles)
len(unhappy_girls) + len(happy_girls) == len(female_singles)


len(happy_girls) == len(lucky_guys)
