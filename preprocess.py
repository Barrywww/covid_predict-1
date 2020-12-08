import numpy as np
import pandas as pd
from tqdm import tqdm

variable_list = ['total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths', 'new_deaths', 'new_deaths_smoothed', 'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million','total_deaths_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million', 'total_tests', 'new_tests', 'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_per_case', 'positive_rate']

# Extracting features
def prepare_data():
    owid_data = pd.read_csv("./data/owid-covid-data.csv", dtype=str).drop(columns=['continent', 'location', 'stringency_index', 'tests_units'])
    oxf_data = pd.read_csv("./data/OxCGRT_latest.csv", dtype=str)
    oxf_data = oxf_data[oxf_data["Jurisdiction"] == "NAT_TOTAL"]
    oxf_data.drop(columns=['CountryName', 'RegionName', 'RegionCode','Jurisdiction', 'M1_Wildcard', 'ConfirmedCases', 'ConfirmedDeaths'], inplace=True)
    oxf_data.rename(columns={'E2_Debt/contract relief': 'E2_Debt contract relief', 'Date': 'date', 'CountryCode': 'iso_code'}, inplace=True)
    cols = [c for c in oxf_data.columns if c.lower(
    )[-4:] != 'flag' and c.lower()[-7:] != 'display']
    oxf_data = oxf_data[cols]

    # Setting datatype
    for df in [owid_data, oxf_data]:
        df['date'] = pd.to_datetime(df['date'])

        for col in df.columns[2:]:
            df[col] = df[col].astype(float)

    owid_constant_features = owid_data.groupby(['iso_code']).mean().drop(columns=variable_list)
    for col in owid_constant_features:
        if owid_constant_features[col].count() < 150:
            owid_constant_features.drop(columns=[col], inplace=True)
    owid_constant_features = owid_constant_features.sort_values(
        by=['iso_code'], inplace=True)

    owid_variables = owid_data.loc[:, variable_list]

    return owid_data, oxf_data, owid_constant_features, owid_variables


def generate_country_list(df):
    iso = (pd.unique(df['iso_code'].dropna())).tolist()
    iso = [code for code in iso if code[:4] != 'OWID']
    return iso


def generate_date_series():
    return pd.Series(pd.date_range('2020', freq='D', periods=275))


def generate_country_csv():
    owid_data, oxf_data, owid_constant_features, owid_variables = prepare_data()
    world_data = pd.merge(owid_data, oxf_data, how='outer').set_index('date')

    for country in tqdm(generate_country_list(owid_data)):
        this_df = world_data[world_data["iso_code"] == country].drop(columns=['iso_code']).sort_values(by=['date'])
        this_df.to_csv("./country_csv/" + country + ".csv")

generate_country_csv()
