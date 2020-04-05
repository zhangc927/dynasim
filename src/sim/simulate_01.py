import os
import numpy as np
import pandas as pd

sim_path = "/Users/christianhilscher/Desktop/dynsim/src/sim/"
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
os.chdir(sim_path)

from data_prep import SOEP_to_df


# df_base = pd.read_stata(input_path + 'SOEP_prepared_costs_2019-11-27_restricted.dta')
#
# df = SOEP_to_df(df_base)
# df_short = df[:1000]
#

def _dating_market(dataf):
    dataf = dataf.copy()

    eligible = (dataf['in_couple'] == 0) & (dataf['child'] == 0)
    female_singles = dataf[eligible & (dataf['female'] == 1)]
    male_singles = dataf[eligible & (dataf['female'] == 0)]

    not_single = dataf[~eligible]

    new_couples = round(0.2 * min(len(female_singles),
                                  len(male_singles)))

    male_singles['hid'][:new_couples] = female_singles['hid'][:new_couples]
    male_singles['east'][:new_couples] = female_singles['east'][:new_couples]
    male_singles['hhweight'][:new_couples] = female_singles['hhweight'][:new_couples]

    male_singles['in_couple'][:new_couples] = 1
    female_singles['in_couple'][:new_couples] = 1

    dataf_out = pd.concat((not_single,
                       female_singles,
                       male_singles), axis=0)

    assert(
        len(dataf_out) == len(dataf)
    ), 'Lenght of dataframe is not the same as before'

    return dataf_out, new_couples

def _separations(dataf):
    dataf = dataf.copy()

    probability = np.random.uniform(0, 1, len(dataf))
    condition_married = (dataf['married'] == 1) & (probability<0.01)
    condition_incouple = (dataf['in_couple'] == 1) & (dataf['married'] == 0) & (probability<0.02)
    condition_separation = (condition_married | condition_incouple)

    males = (condition_separation) & (dataf['female'] == 0)
    dataf.loc[condition_separation, ['married', 'in_couple']] = [[0, 0]]

    # Men moove out; resetting HID
    dataf.loc[males, 'orighid'] = dataf.loc[males, 'hid'].copy()
    dataf.loc[males, 'hid'] += np.arange(1, np.sum(males)+1)

    separations_this_period = np.sum(condition_separation)

    return dataf, separations_this_period

def _marriage(dataf):
    """
    10% of all couples get married
    """
    dataf = dataf.copy()

    marriable = (dataf['married'] == 0) & (dataf['in_couple']==1)
    probability = np.random.uniform(0, 1, len(dataf))
    condition = (marriable) & (probability<0.1)

    dataf.loc[condition, 'married'] = 1
    marriages_this_period = np.sum(condition)

    return dataf, marriages_this_period
