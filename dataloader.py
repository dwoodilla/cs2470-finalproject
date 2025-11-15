import pandas as pd
import json
import os



with open("./data/downloaded/aqs/myron2025.json") as f:
    raw_json = json.load(f)
    df = pd.json_normalize(raw_json['Data'])
    print(df.head())

# myron2025 = pd.read_json("./data/downloaded/aqs/myron2025.json")
# print(0)