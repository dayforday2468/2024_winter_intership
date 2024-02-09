import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
from modules.GPS_feature import FeatureMaker

# Read data.csv
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.join(root_dir, "Data"), "data.csv")
data = pd.read_csv(data_dir)
data["obtained_at"] = pd.to_datetime(data["obtained_at"])
data["created_at"] = pd.to_datetime(data["created_at"])

# Select some data from data
selected_data = data[
    (data["obtained_at"].dt.year == 2024)
    & (data["obtained_at"].dt.month == 1)
    & (data["obtained_at"].dt.day == 17)
]

# Create an instance of featuremaker
featuremaker = FeatureMaker()

# Compute the speed
speed = featuremaker.speed(selected_data)

# Create a color map based on obtained_at
norm = plt.Normalize(speed.min(), speed.max())
cmap = plt.colormaps["viridis"]

# Plot the data
fig, ax = plt.subplots()
longitude = selected_data["longitude"]
latitude = selected_data["latitude"]
points = np.array([longitude, latitude]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(
    segments=segments,
    color=[cmap(norm(x)) for x in speed],
    linewidth=1,
)

ax.add_collection(lc)

# Add a colorbar
cbar = plt.colorbar(lc, ax=ax)
cbar.set_label("speed")
cbar.set_ticks([speed.min(), speed.max()])
cbar.set_ticklabels([str(int(speed.min())), str(int(speed.max()))])

# Set axis labels and title
ax.set_xlim(longitude.min(), longitude.max())
ax.set_ylim(latitude.min(), latitude.max())
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_title("Latitude and longitude colored by Speed")

plt.show()
