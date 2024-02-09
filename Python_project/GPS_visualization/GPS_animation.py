import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
# time_series = time_series.resample("5T").mean()
# time_series = time_series.interpolate()

# Set fixed xlim and ylim
xlim = (time_series["longitude"].min(), time_series["longitude"].max())
ylim = (time_series["latitude"].min(), time_series["latitude"].max())


def update_plot(frame):
    plt.clf()
    (line,) = plt.plot(
        time_series["longitude"].iloc[:frame], time_series["latitude"].iloc[:frame]
    )
    plt.title(f"Time: {time_series.index[frame]}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.plot(
        time_series["longitude"].loc[time_series.index[frame]],
        time_series["latitude"].loc[time_series.index[frame]],
        c="red",
        marker="D",
    )
    plt.xlim(xlim)
    plt.ylim(ylim)
    return [line]


# Create the animation
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update_plot, frames=len(time_series), repeat=False)

# Save the animation as a GIF
ani_dir = os.path.join(os.path.join(root_dir, "Animation"), "GPS_time.gif")
ani.save(ani_dir, fps=12)
