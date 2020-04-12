import numpy as np
import pandas as pd

def SOEP_to_df(dataf):
    """
    This function takes the SOEP data as a dataframe and returns the the harmonized data such that the rest of the code can work with it. It also renames the columns etc
    """

    dataf = dataf.copy()

    # Checking whether some adjustments have already been made
    if "emplstatus" in dataf.columns.tolist():
        dataf = dataf.drop(['emplstatus', 'married'], axis = 1)
        print('Attention, this data is not the original SOEP data but already preprocessed.')
    else:
        dataf = dataf


    dataf = dataf.rename(columns={'syear': 'year',
                                  'phrf': 'personweight',
                                  'pglabgro': 'gross_earnings',
                                  'hhrf': 'hhweight',
                                  'hgheat': 'heizkosten',
                                  'kaltmiete': 'bruttokaltmiete',
                                  'kind': 'child',
                                  'pgpsbil': 'education',
                                  'married_h': 'married'})

    dataf['orighid'] = dataf['hid']
    # For now motherpid is 0 as a placeholder and maximum age is set to 99
    dataf['motherpid'] = 0
    dataf['age_max'] = 99

    dataf = _numeric_eduation(dataf)
    dataf = _numeric_employment_status(dataf)
    dataf = _numeric_laborforce(dataf)
    dataf = _numeric_working(dataf)
    dataf = _numeric_migration(dataf)
    dataf = make_hh_vars(dataf)

    return dataf

def _numeric_eduation(dataf):

    dataf = dataf.copy()

    dataf.loc[:, "educ"] = 0
    dataf.loc[(dataf['education'] == "[1] Hauptschulabschluss"), "educ"] = 0
    dataf.loc[(dataf['education'] == "[2] Realschulabschluss"), "educ"] = 1
    dataf.loc[(dataf['education'] == "[3] Fachhochschulreife"), "educ"] = 2
    dataf.loc[(dataf['education'] == "[4] Abitur"), "educ"] = 3
    dataf.loc[(dataf['education'] == "[5] Anderer Abschluss"), "educ"] = 4
    dataf.loc[(dataf['education'] == "[6] Ohne Abschluss verlassen"), "educ"] = 5
    dataf.loc[(dataf['education'] == "[7] Noch kein Abschluss"), "educ"] = 6

    dataf.drop("education", axis = 1, inplace = True)
    dataf.rename(columns={'educ': 'education'}, inplace=True)

    return dataf

def _numeric_employment_status(dataf):

    dataf = dataf.copy()

    dataf.loc[:, "emp"] = 0
    dataf.loc[(dataf['employment_status'] == "Bildung"), "emp"] = 0
    dataf.loc[(dataf['employment_status'] == "Teilzeit"), "emp"] = 1
    dataf.loc[(dataf['employment_status'] == "Vollzeit"), "emp"] = 2
    dataf.loc[(dataf['employment_status'] == "Nicht erwerbstaetig"), "emp"] = 3
    dataf.loc[(dataf['employment_status'] == "Rente"), "emp"] = 4

    dataf.drop("employment_status", axis = 1, inplace = True)
    dataf.rename(columns={'emp': 'employment_status'}, inplace=True)

    dataf['fulltime'] = 0
    dataf.loc[dataf['employment_status'] == 2, 'fulltime'] = 1

    return dataf

def _numeric_laborforce(dataf):
    dataf = dataf.copy()

    dataf.loc[:,'lfs'] = 0
    dataf.loc[dataf['pglfs'] == '[11] Working', 'lfs'] = 1

    dataf.drop("pglfs", axis = 1, inplace = True)
    return dataf

def _numeric_working(dataf):
    dataf = dataf.copy()

    dataf.loc[:,'working'] = 0
    dataf.loc[dataf['employment_status'] == 1, 'working'] = 1
    dataf.loc[dataf['employment_status'] == 2, 'working'] = 1
    return dataf

def _numeric_migration(dataf):
    dataf = dataf.copy()

    dataf['migration'] = np.NaN

    dataf.loc[dataf['migback'] == 0, 'migration'] = 1
    dataf.loc[dataf['migback'] == "[1] kein Migrationshintergrund", 'migration'] = 0

    dataf.drop('migback', axis=1, inplace=True)
    dataf.rename(columns={'migration': 'migback'}, inplace=True)

    return dataf

# Making household wide variables
def make_hh_vars(dataf):
    dataf = dataf.copy()
    dataf = _get_multiindex(dataf)

    dataf = _hh_income(dataf)
    dataf = _hh_age_youngest(dataf)
    dataf = _hh_fraction_working(dataf)
    dataf.reset_index(drop=True, inplace=True)
    return dataf


def _get_tupleindices(dataf):
    years = dataf['year'].tolist()
    hids = dataf['hid'].tolist()
    return list(zip(years, hids))

def _get_multiindex(dataf):
    dataf = dataf.copy()
    index_list = _get_tupleindices(dataf)

    mindex = pd.MultiIndex.from_tuples(index_list, names=['year' ,
                                                          'hid'])
    dataf_out = dataf.set_index(mindex)
    dataf_out = dataf_out.sort_index(level=1)

    return dataf_out

def _hh_income(dataf):
    dataf = dataf.copy()
    earnings = dataf.groupby(level=['year', 'hid'])['gross_earnings'].sum()
    dataf['hh_income'] = earnings
    return dataf

def _hh_size(dataf):
    dataf = dataf.copy()
    size = dataf.groupby(level=['year', 'hid'])['gross_earnings'].size()
    dataf['n_people'] = size
    return dataf

def _hh_children(dataf):
    dataf = dataf.copy()
    children = dataf.groupby(level=['year', 'hid'])['child'].sum()
    dataf['n_children'] = children
    return dataf

def _hh_fraction_working(dataf):
    dataf = dataf.copy()

    total = dataf.groupby(level=['year', 'hid'])['working'].sum()
    dataf['total_working'] = total

    dataf = _hh_size(dataf)
    dataf = _hh_children(dataf)

    dataf['n_adults'] = dataf['n_people'] - dataf['n_children']
    dataf['hh_frac_working'] = dataf['total_working']/dataf['n_adults']

    dataf.drop(['total_working', 'n_adults'], axis=1, inplace=True)
    return dataf

def _hh_age_youngest(dataf):
    dataf = dataf.copy()

    smallest_age = dataf.groupby(level=['year', 'hid'])['age'].min()
    dataf['hh_youngest_age'] = smallest_age
    return dataf
