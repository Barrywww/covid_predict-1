import numpy as np
import pandas as pd

df = pd.read_csv('Project/raw_data/owid-covid-data.csv')

df = df.drop(columns=['continent'])

grouped_mean = df.groupby(['iso_code']).mean()

gdp_per_capita = grouped_mean['gdp_per_capita']
population = grouped_mean['population']
population_density = grouped_mean['population_density']
life_expectancy = grouped_mean['life_expectancy']


# import this constant_values
constant_values = pd.concat(
    [gdp_per_capita, population, population_density, life_expectancy], axis=1)
