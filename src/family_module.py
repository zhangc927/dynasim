import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def dating_market(dataf):
    """
    New couples finding together. Right now 20% of the singles find a new partner.
    """
    dataf = dataf.copy()

    eligible = (dataf['in_couple'] == 0) & (dataf['child'] == 0) & (dataf['n_people'] - dataf['n_children'] == 1)

    female_singles = dataf[eligible & (dataf['female'] == 1)]
    male_singles = dataf[eligible & (dataf['female'] == 0)]

    not_single = dataf[~eligible]

    new_couples = round(0.2 * min(len(female_singles),
                                  len(male_singles)))

    matching_dict = _matching(female_singles,
                              male_singles,
                              new_couples)

    dataf_out = pd.concat((not_single,
                           matching_dict['girls'],
                           matching_dict['guys']), axis=0)

    assert(
        len(matching_dict['girls']) == len(female_singles)
    ), 'Lenght of dataframe is not the same as before'

    return dataf_out, new_couples

def _matching(females, males, number):
    """
    Finding the 5 best fitting matches and then choosing randomly.
    #TODO: think of a better way than this loop
    """
    partners = females.copy()
    lucky_guys = males.sample(number)

    neigh = NearestNeighbors(n_neighbors=5)

    happy_girls = pd.DataFrame()

    # Looping since as soon as one couple matched, that woman is no longer available
    for i in np.arange(len(lucky_guys)):
        neigh.fit(partners[['age',
                            'gross_earnings',
                            'education',
                            'employment_status']])
        bachelor = lucky_guys.iloc[i,:]
        bachelor = bachelor[['age',
                             'gross_earnings',
                             'education',
                             'employment_status']].to_numpy().reshape(1,-1)

        partner_choice = neigh.kneighbors(bachelor)

        partner = np.random.choice(np.ravel(partner_choice[1]), 1)
        happy_girls = pd.concat([happy_girls, partners.iloc[partner,:]])
        partners.drop(partners.iloc[partner,:].index, inplace=True)


    happy_girls, lucky_guys = _adjust_values(happy_girls, lucky_guys)

    singles_dict = {'all_female': females,
                'all_male': males,
                'happy_girls': happy_girls,
                'lucky_guys': lucky_guys}

    out_dict = _concat_singles(singles_dict)

    return out_dict

def _concat_singles(dici):
    """
    Concating those who found new partners and those who didn't
    """
    unlucky_guys = dici['all_male'][dici['all_male'].index.isin(dici['lucky_guys'].index) == False]

    unhappy_girls = dici['all_female'][dici['all_female'].index.isin(dici['happy_girls'].index) == False]

    girls = pd.concat((dici['happy_girls'], unhappy_girls), axis = 0)
    guys = pd.concat((dici['lucky_guys'], unlucky_guys), axis = 0)

    assert(
        len(unlucky_guys) + len(dici['lucky_guys']) == len(dici['all_male'])
    ), "Error in concating guys"

    assert(
        len(unhappy_girls) + len(dici['happy_girls']) == len(dici['all_female'])
    ), "Error in concating girls"

    out_dict = {'girls' : girls,
                'guys': guys}
    return out_dict

def _adjust_values(females, males):
    """
    Adjusting the values as the man moves in with the woman
    """
    females = females.copy()
    males = males.copy()

    males.loc[:,'hid'] = females['hid'].tolist()
    males.loc[:,'east'] = females['east'].tolist()
    males.loc[:,'hhweight'] = females['hhweight'].tolist()
    males.loc[:,'in_couple'] = 1
    females.loc[:,'in_couple'] = 1

    return females, males

def _separations(dataf):
    """
    Calculates the seperations in each period.
    Only those who are married or in a relationship (in_couple) can separate
    """
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
