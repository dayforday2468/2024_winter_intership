import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

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

# Create time series
time_series = selected_data.set_index("obtained_at")
time_series = time_series.resample("5T").mean()
time_series = time_series.interpolate()

# Create a color map based on obtained_at
norm = plt.Normalize(
    time_series.index[0].timestamp(), time_series.index[-1].timestamp()
)
cmap = plt.colormaps["viridis"]

# Plot the data
fig, ax = plt.subplots()
longitude = time_series["longitude"]
latitude = time_series["latitude"]
points = np.array([longitude, latitude]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(
    segments=segments,
    color=[cmap(norm(date.timestamp())) for date in time_series.index],
    linewidth=1,
)

ax.add_collection(lc)

# Add a colorbar
cbar = plt.colorbar(lc, ax=ax)
cbar.set_label("obtained_at time")
cbar.set_ticks([0, 1])
cbar.set_ticklabels([time_series.index[0], time_series.index[-1]])

# Set axis labels and title
ax.set_xlim(longitude.min(), longitude.max())
ax.set_ylim(latitude.min(), latitude.max())
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_title("Latitude and longitude colored by Obtained_at")

plt.show()
