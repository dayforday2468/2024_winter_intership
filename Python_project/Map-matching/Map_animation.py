import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os

# Read data_smoothed.csv
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.join(root_dir, "Data"), "data_smoothed.csv")
data = pd.read_csv(data_dir)
data["obtained_at"] = pd.to_datetime(data["obtained_at"])
data["created_at"] = pd.to_datetime(data["created_at"])

# Preprocessing data
time_series = data.set_index("obtained_at")

place_name = "유성구"
network = ox.graph_from_place(place_name, network_type="drive")

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the network with a zoomed-in bounding box
ox.plot_graph(network, node_size=2, edge_linewidth=1, ax=ax, show=False)

# Initialize line and point plot
(lines,) = ax.plot([], [])
(point,) = ax.plot([], [], c="r", marker="o")
title = ax.set_title("")

# Set fixed xlim and ylim
xlim = (time_series["longitude"].min(), time_series["longitude"].max())
ylim = (time_series["latitude"].min(), time_series["latitude"].max())


def update(frame):
    # Update line plot with new data points
    x_data = time_series["longitude"].iloc[:frame]
    y_data = time_series["latitude"].iloc[:frame]

    lines.set_xdata(x_data)
    lines.set_ydata(y_data)

    point.set_xdata(time_series["longitude"].iloc[frame])
    point.set_ydata(time_series["latitude"].iloc[frame])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    title.set_text(f"Timestamp:{time_series.index[frame]}")

    return (lines, point, title)


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(data), interval=100, blit=True)


# Save the animation as a GIF
ani_dir = os.path.join(os.path.join(root_dir, "Animation"), "GPS_map.gif")
ani.save(ani_dir, fps=12)
