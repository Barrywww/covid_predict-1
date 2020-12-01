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
    grouped_mean = df.groupby(['iso_code']).mean()

    columns = []
    for col in ['population', 'population_density', 'median_age', 'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index']:
        columns.append(grouped_mean[col])
    constant_values = pd.concat(columns, axis=1)
    constant_values.to_csv("preprocessed_data/constant_values.csv")


# for every feature, use this function to throw the values into the template
def fill_owid_data(col_name):
    this = template.copy()
    for index, row in df.iterrows():
        this.loc[row['iso_code'], row['date']] = row[col_name]
    return this


def generate_owid_csv(col_name):
    this = fill_owid_data(col_name)
    this.to_csv("preprocessed_data/" + col_name + ".csv")


generate_constant_data()
for col in ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_cases_per_million', 'new_cases_per_million', 'total_deaths_per_million', 'new_deaths_per_million', 'total_tests', 'new_tests', 'positive_rate', 'stringency_index']:
    generate_owid_csv(col)
