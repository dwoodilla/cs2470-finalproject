import pandas as pd
import json
import os

PARAM_DICT = {
    "44201": "o3",
    "68105": "avg_temp",
    "62101": "outdoor_temp",
    "61301": "mix_height",
    "42101": "co",
    "42601": "no",
    "42602": "no2",
    "88101": "pm25",
    "86101": "pm10_25"
}

df_list:list[pd.DataFrame] = []

for dataset in os.listdir("./data/downloaded/aqs/"):
    if "json" not in dataset: continue
    with open(os.path.join("./data/downloaded/aqs/", dataset)) as f:
        raw_json = json.load(f)
    df = pd.json_normalize(raw_json['Data'])
    df['datetime_gmt'] = pd.to_datetime(df['date_gmt'] + ' ' + df['time_gmt'], utc=True)

    df = df[['datetime_gmt','parameter_code','sample_measurement']]
    df['parameter_code'] = df['parameter_code'].apply(lambda x: PARAM_DICT[x])
    df = df.pivot_table(
        index='datetime_gmt',
        columns='parameter_code',
        values='sample_measurement',
        dropna=False
    ).reset_index()
    df = df.drop(columns=['avg_temp','mix_height', 'pm10_25'], errors='ignore')
    df_list.extend([df])

ds = pd.concat(df_list)
ds['datetime_gmt'] = pd.to_datetime(ds['datetime_gmt'], utc=True, errors='raise')
ds = ds.set_index('datetime_gmt')
ds.to_pickle('./data/myron_ds.pkl')
ds.to_json('./data/myron_ds.json', index=True)