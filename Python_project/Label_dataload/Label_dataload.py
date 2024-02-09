import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

# read Record.json and make the dataframe
root_dir = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(root_dir, "라벨이 지정된 장소.json")
f = open(json_dir, encoding="UTF-8")
data = json.load(f)
f.close()

label_data = data["features"]

df = pd.DataFrame(
    {
        "latitude": [
            entry["geometry"]["coordinates"][1] * 10**7 for entry in label_data
        ],
        "longitude": [
            entry["geometry"]["coordinates"][0] * 10**7 for entry in label_data
        ],
        "name": [entry["properties"]["name"] for entry in label_data],
    }
)

# Display the original data
print("Original data")
print(df)

# Save the dataframe
df.to_csv(os.path.join(root_dir, "label.csv"), index=False, encoding="cp949")
