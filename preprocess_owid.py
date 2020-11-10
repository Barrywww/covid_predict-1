import numpy as np
import pandas as pd

df = pd.read_csv('data/owid-covid-data.csv')

df = df.drop(columns=['continent'])

# get the cosntant values using the mean of all the country data
grouped_mean = df.groupby(['iso_code']).mean()

gdp_per_capita = grouped_mean['gdp_per_capita']
population = grouped_mean['population']
population_density = grouped_mean['population_density']
life_expectancy = grouped_mean['life_expectancy']


# import this constant_values
constant_values = pd.concat(
    [gdp_per_capita, population, population_density, life_expectancy], axis=1)

# making a template for the timeSeries data points
date = np.sort(pd.unique(df['date']))
iso = pd.unique(df['iso_code'])

template = pd.DataFrame(index=iso, columns=date)

# for every feature, use this function to throw the values into the template
def fill_owid_data(col_name):
    this = template.copy()
    for index, row in df.iterrows():
        this.loc[row['iso_code'], row['date']] = row[col_name]
    return this

def generate_owid_csv(col_name, filename):
    this = fill_owid_data(col_name)
    this.to_csv("data/" + filename + ".csv")


generate_owid_csv("new_deaths", "new_deaths")
