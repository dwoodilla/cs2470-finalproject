import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri

"""
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

# ==== PROCESS DOWNLOADED MYRON DATA ====

df_list:list[pd.DataFrame] = []

for dataset in os.listdir("./data/downloaded/aqs/"):

    with open(os.path.join("./data/downloaded/aqs/", dataset)) as f:
        raw_json = json.load(f)
    df = pd.json_normalize(raw_json['Data'])
    df['datetime_gmt'] = pd.to_datetime(df['date_gmt'] + ' ' + df['time_gmt'], utc=True) # type: ignore

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

aq_ds = pd.concat(df_list)
aq_ds['datetime'] = pd.to_datetime(aq_ds['datetime_gmt'], utc=True, errors='raise')
aq_ds = aq_ds.set_index('datetime')
aq_ds  = aq_ds.reset_index()

# ==== IMPORT AND CLEAN NOAA MET DATA ====

# worldmet = importr("worldmet")

# ro.r(
# '''
#     met_ds = worldmet::importNOAA(code=c("725070-14765"), year=1998:2025)
# '''
# )

# met_ds:pd.DataFrame
# with (ro.default_converter + pandas2ri.converter).context():
#     met_ds = ro.r['met_ds'] # type: ignore

# met_ds.to_csv('./data/downloaded/met_tfg.csv')

met_ds = pd.read_csv('./data/downloaded/met_tfg.csv')

met_ds = met_ds.rename(
    columns={
        "date":"datetime",
        "Uu":"wind_x",
        "Vv":"wind_y"
    }
)
met_ds = met_ds.reset_index()

met_ds = met_ds.drop(
    columns=[
        'cl_2','cl_3','cl_2_height','cl_3_height','precip_6','pwc',
        'code','station','latitude','longitude','elev','ws','wd','index', 'air_temp',
        'atmos_pres','visibility', 'cl','cl_1','cl_1_height'
    ]
)

met_ds['datetime'] = pd.to_datetime(met_ds['datetime'], utc=True)

# ==== CONSTRUCT DATASET ====

ds = pd.merge_ordered(
    met_ds,
    aq_ds,
    on='datetime'
)
ds = ds.set_index('datetime')
ds = ds.rename(columns={
    'outdoor_temp':'temp'
})

# === Normalize ===
iqr = ds.quantile(0.75)-ds.quantile(0.25)
median_normed = ds - ds.median()
ds.describe().to_json("./data/preprocessed/normalizing_constants.json")

ds = median_normed.div(iqr, axis='columns')

# === Add temporal positional encodings ===

datetime = pd.to_datetime(ds.index, utc=True)
ds['Weekday'] = datetime.weekday # These will be one-hot encoded later

unix_ts = datetime.map(pd.Timestamp.timestamp)
normed_dt = (unix_ts - unix_ts[0])/3600

encode_time = lambda f,x: f( ((2*np.pi)/x) * normed_dt )

ds['Day Sine'] = encode_time(np.sin, 24)
ds['Day Cosine'] = encode_time(np.cos, 24)
ds['Month Sine'] = encode_time(np.sin, datetime.daysinmonth * 24)
ds['Month Cosine'] = encode_time(np.cos, datetime.daysinmonth * 24)
ds['Year Sine'] = encode_time(np.sin, 24*365)
ds['Year Cosine'] = encode_time(np.cos, 24*365)

ds = ds[[
    'Day Sine', 'Day Cosine', 
    'Month Sine', 'Month Cosine',
    'Year Sine', 'Year Cosine',
    'Weekday',
    'temp','dew_point', 
    'RH', 'ceil_hgt', 
    'wind_x', 'wind_y',
    'co', 'no', 'no2', 'o3', 'pm25'
]]

ds.to_pickle('./data/preprocessed/aqmet_pd.pkl')
"""

ds = pd.read_pickle('./aqmet_pd.pkl')
ds_np = ds.to_numpy(dtype=np.float32)[:,1:]
del ds

def construct_contextualized_ds(x:np.ndarray, window_len:int =24) -> tf.data.Dataset:
    
    is_finite_bool = np.isfinite(x)
    is_finite_fl = is_finite_bool.astype(np.float32)
    finite_x = np.where(is_finite_bool, x, 0.0)

    Xs=[]; Xc=[]; Y=[]
    for i in range(len(finite_x)-window_len):
        Xs_window   = finite_x[i:i+window_len, -5:]
        Xc_window   = finite_x[i:i+window_len, :-5]
        target      = x [i+1 : i+window_len+1, -5:]

        Xs_mask = is_finite_fl[i:i+window_len, -5:]
        Xc_mask = is_finite_fl[i:i+window_len, :-5]

        Xs_cat = np.concatenate([Xs_window, Xs_mask], axis=-1)
        Xc_cat = np.concatenate([Xc_window, Xc_mask], axis=-1)

        Xs.append(Xs_cat)
        Xc.append(Xc_cat)
        Y.append(target)
    return tf.data.Dataset.from_tensor_slices(((Xs, Xc), Y))

ds = construct_contextualized_ds(ds_np)
ds.save('./data/datasets/myron_tfg_2')