import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

# read Record.json and make the dataframe
root_dir = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(root_dir, "Records.json")
f = open(json_dir)
data = json.load(f)
f.close()

location_data = data["locations"]

df = pd.DataFrame(
    {
        "latitude": [entry["latitudeE7"] for entry in location_data],
        "longitude": [entry["longitudeE7"] for entry in location_data],
        "time": [entry["timestamp"] for entry in location_data],
    }
)
# Display the original DateFrame
print("Original dataframe")
print(df)

# Modify time
df["obtained_at"] = pd.to_datetime(df["time"].str[:19], format="%Y-%m-%dT%H:%M:%S")
df["created_at"] = pd.Timestamp.now()
df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
df = df.drop(columns="time")

# Display the updated DataFrame
print("Updated dataframe")
print(df)

# Saving the dataframe
df.to_csv(os.path.join(root_dir, "data.csv"), index=False)
