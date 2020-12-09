import numpy as np
import pandas as pd
from tqdm import tqdm

variable_list = ['new_cases', 'total_deaths', 'total_tests', 'tests_per_case', 'positive_rate']

# Extracting features
def prepare_data():
    owid_data = pd.read_csv("./data/owid-covid-data.csv", dtype=str).drop(columns=['continent', 'location', 'stringency_index', 'tests_units', 'weekly_icu_admissions', 'weekly_hosp_admissions', 'hosp_patients', 'total_cases', 'new_deaths', 'new_tests', 'icu_patients'])
    owid_data.drop(columns=[c for c in owid_data.columns if c.lower()[-7:] == 'million' or c.lower()[-8:] == 'smoothed' or c.lower()[-8:] == 'thousand'], inplace=True)

    oxf_data = pd.read_csv("./data/OxCGRT_latest.csv", dtype=str)
    oxf_data = oxf_data[oxf_data["Jurisdiction"] == "NAT_TOTAL"]
    oxf_data.drop(columns=['CountryName', 'RegionName', 'RegionCode', 'Jurisdiction', 'M1_Wildcard', 'ConfirmedCases', 'ConfirmedDeaths'], inplace=True)
    oxf_data.rename(columns={'E2_Debt/contract relief': 'E2_Debt contract relief', 'Date': 'date', 'CountryCode': 'iso_code'}, inplace=True)
    cols = [c for c in oxf_data.columns if c.lower()[-4:] != 'flag' and c.lower()[-7:] != 'display']
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
    

    oxf_data = oxf_data[oxf_data["date"] >= pd.Timestamp(2020, 1, 23)]

    return owid_data, oxf_data, owid_constant_features, owid_variables


def generate_country_list(series1, series2):
    # iso = (pd.unique(df['iso_code'].dropna())).tolist()
    # iso = [code for code in iso if code[:4] != 'OWID']
    # return iso
    return sorted(list(set(series1).intersection(set(series2))))


def generate_date_series():
    return pd.Series(pd.date_range('2020', freq='D', periods=275))


def generate_country_csv():
    owid_data, oxf_data, owid_constant_features, owid_variables = prepare_data()
    country_list = generate_country_list(owid_data["iso_code"], oxf_data["iso_code"])
    # owid_data = owid_data[owid_data["iso_code"].isin(country_list)].copy()
    # oxf_data = oxf_data[oxf_data["iso_code"].isin(country_list)].copy()
    world_data = pd.merge(owid_data, oxf_data, how='inner').set_index('date')

    for country in tqdm(country_list):
        this_df = world_data[world_data["iso_code"] == country].drop(columns=['iso_code']).sort_values(by=['date'])
        this_train = this_df.iloc[: 252]
        this_test = this_df.iloc[252:]
        this_train.to_csv("./country_csv/train/" + country + "_train.csv")
        this_test.to_csv("./country_csv/test/" + country + "_test.csv")


if __name__ == '__main__':
    generate_country_csv()
