import pandas as pd


def generate_democracy_index():
    dict = pd.read_csv(
        'data/owid-covid-data.csv')[['location', 'iso_code']].drop_duplicates()
    democracy = pd.read_csv('data/democracy_index.csv').set_index('Country')
    democracy['iso_code'] = democracy['iso_code'].astype('object')

    for ind, row in democracy.iterrows():
        if len(dict[dict['location'] == ind]['iso_code'].values) == 1:
            democracy.at[ind, 'iso_code'] = dict[dict['location']
                                                 == ind]['iso_code'].values[0]

    democracy = democracy.set_index('iso_code')

    democracy.to_csv("data/preprocessed_democracy_index.csv")


generate_democracy_index()
