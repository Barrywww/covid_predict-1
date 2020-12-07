import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('data/owid-covid-data.csv')

df = df.drop(columns=['continent', 'location'])


# making a template for the timeSeries data points
date = np.sort(pd.unique(df['date']))
iso = pd.unique(df['iso_code'])

template = pd.DataFrame(index=iso, columns=date)


def generate_constant_data():
    # get the cosntant values using the mean of all the country data
    grouped = df.groupby(['iso_code']).mean().drop(columns=['total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths', 'new_deaths', 'new_deaths_smoothed', 'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million', 'total_deaths_per_million',
                                                            'new_deaths_per_million', 'new_deaths_smoothed_per_million', 'total_tests', 'new_tests', 'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_per_case', 'positive_rate', 'stringency_index'])
    for col in grouped:
        if grouped[col].count() < 150:
            grouped = grouped.drop(columns=[col])
    # regime = _integrate_regime_data()
    grouped = pd.concat(
        [regime, grouped], axis=1, join="outer")
    grouped.sort_index(inplace=True)
    grouped.to_csv("preprocessed_data/constant_values.csv")


# def _integrate_regime_data():
#     political_regime = pd.read_csv('data/political-regime-updated2016.csv')
#     regime2015 = political_regime[political_regime['Year'] == 2015]
#     regime2015 = regime2015.set_index('iso_code')
#     regime2015 = regime2015.drop(columns=['Entity', 'Year'])
#     return regime2015


# for every feature, use this function to throw the values into the template
def _fill_owid_data(col_name):
    this = template.copy()
    for index, row in df.iterrows():
        this.loc[row['iso_code'], row['date']] = row[col_name]
    return this


def _generate_owid_csv(col_name):
    this = _fill_owid_data(col_name)
    this.to_csv("preprocessed_data/" + col_name + ".csv")


def generate_owid_csv():
    for col in ['total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths', 'new_deaths', 'new_deaths_smoothed', 'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million', 'total_deaths_per_million',
                'new_deaths_per_million', 'new_deaths_smoothed_per_million', 'total_tests', 'new_tests', 'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_per_case', 'positive_rate', 'stringency_index']:
        _generate_owid_csv(col)


# generate_owid_csv()
generate_constant_data()
