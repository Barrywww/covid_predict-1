import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def to_supervised(src_path, dst_path, n_in, n_out):
    country_dict = walk(src_path)
    scaler = MinMaxScaler(feature_range=(0,1))
    for k, v in country_dict.items():
        v = v.drop(["total_cases", "new_cases_smoothed","total_cases_per_million", "new_cases_per_million",
                    "new_cases_smoothed_per_million"], axis=1)
        v = v.fillna(0)
        values = v.values.astype("float64")
        scaled = scaler.fit_transform(values)
        df = generate_supervised_data(scaled, n_in, n_out)
#         y_t = df.iloc[:,index_y]
#         df = df.drop(index_y, axis=1)
#         df.insert(len(df.columns) - (n_out - 1), y_t.name, y_t)
        df = df.drop(df.columns[[i for i in range(((n_in*scaled.shape[1])+1), df.shape[1])]], axis=1)
        df.to_csv(dst_path + k + "_supervised.csv")
    
def generate_supervised_data(data, n_in=1, n_out=1, drop=True):
    n_vars = data.shape[1]
    df = pd.DataFrame(data)
    cols = []
    names = []
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if drop:
        agg = agg.dropna()
    return agg
    
    
def walk(path):
    d = {}
    for root, dir, files in os.walk(path):
        for file in files:
            d[file.split(".")[0]] = pd.read_csv(os.path.join(root, file), index_col=0)
    return d


NUM_IN = 7
NUM_OUT = 1
to_supervised("./country_csv/", "./country_csv/supervised_%d/" % NUM_IN, NUM_IN, NUM_OUT)
