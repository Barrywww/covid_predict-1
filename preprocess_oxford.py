import numpy as np
import pandas as pd


def output_csv_file(df, col, template, output_dir="./preprocessed_data/"):
    this = template.copy()
    for index, row in df.iterrows():
        this.loc[row["CountryCode"], row["Date"]] = row[col]
    this.to_csv(output_dir + col.replace(" ", "_") + ".csv")


oxford_data = pd.read_csv("./data/OxCGRT_latest.csv", dtype=str)
oxford_data = oxford_data[oxford_data["Jurisdiction"] == "NAT_TOTAL"]

# a separate index value dataframe
oxford_indices = oxford_data[["CountryName", "CountryCode", "Date",
                              "StringencyIndex", "StringencyLegacyIndex", "GovernmentResponseIndex",
                              "ContainmentHealthIndex"]].copy()

# drop irrelevant features
oxford_data.drop(["C1_Flag", "C2_Flag", "C3_Flag", "C4_Flag", "C5_Flag", "C6_Flag", "C7_Flag",
                  "E1_Flag", "H1_Flag", "H6_Flag", "M1_Wildcard"], axis=1, inplace=True)

oxford_data.drop(["RegionName", "RegionCode", "Jurisdiction", 'StringencyIndex', 'StringencyIndexForDisplay',
                  'StringencyLegacyIndex', 'StringencyLegacyIndexForDisplay',
                  'GovernmentResponseIndex', 'GovernmentResponseIndexForDisplay',
                  'ContainmentHealthIndex', 'ContainmentHealthIndexForDisplay',
                  'EconomicSupportIndex', 'EconomicSupportIndexForDisplay'], axis=1, inplace=True)

# avoid directory error
# handle column name: E2_Debt/contract relief -> E2_Debt contract relief
oxford_columns = list(oxford_data.columns)
oxford_data.columns = oxford_columns[0:12] + \
    [oxford_columns[12].replace("/", " ")] + oxford_columns[13:]

# fill empty ConfirmedCases and ConfirmedDeaths with 0
oxford_data["ConfirmedCases"] = oxford_data["ConfirmedCases"].fillna(0)
oxford_data["ConfirmedDeaths"] = oxford_data["ConfirmedDeaths"].fillna(0)

# match the format of date: from YYYYMMDD to YYYY-MM-DD
oxford_data["Date"] = oxford_data["Date"].apply(
    lambda x: x[0:4] + "-" + x[4:6] + "-" + x[6:])
oxford_indices["Date"] = oxford_indices["Date"].apply(
    lambda x: x[0:4] + "-" + x[4:6] + "-" + x[6:])

# set the datatype of numerical values as float
for col in oxford_data.columns[3:]:
    oxford_data[col] = oxford_data[col].astype(float)
for col in oxford_indices.columns[3:]:
    oxford_indices[col] = oxford_indices[col].astype(float)

# create template output dataframe
date = np.sort(pd.unique(oxford_data['Date']))
iso = np.sort(pd.unique(oxford_data['CountryCode']))
template = pd.DataFrame(index=iso, columns=date)

# output datasets
for col in oxford_data.columns[3:]:
    output_csv_file(oxford_data, col, template)

for col in oxford_indices.columns[3:]:
    output_csv_file(oxford_indices, col, template, "./preprocessed_data/I_")
