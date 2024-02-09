import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

distance_thres = 150
root_dir = os.path.dirname(os.path.abspath(__file__))
travel_dir = os.path.join(root_dir, "Travel")
projection_dir = os.path.join(root_dir, "Projection")


# edge is a tuple with start, end points. Each point is a numpy array.
def projection(edge, point):
    # Unpack the edge
    edge_start, edge_end = edge

    # Set start point
    edge_vector = edge_end - edge_start
    point_vector = point - edge_start

    # If the edge is a point, return nan
    if np.linalg.norm(edge_vector) == 0:
        return np.nan

    # Calculate t by dot product
    t = np.dot(point_vector, edge_vector) / np.dot(edge_vector, edge_vector)

    # Project the point if we can
    if 0 <= t <= 1:
        return (edge_start + t * edge_vector).tolist()
    return np.nan


# Read data_smoothed.csv
data = pd.read_csv(os.path.join(travel_dir, "travel.csv"))
data["obtained_at"] = pd.to_datetime(data["obtained_at"])
data["created_at"] = pd.to_datetime(data["created_at"])

# Load the street network
place_name = "유성구"
network = ox.graph_from_place(place_name, network_type="drive", simplify=False)
network = ox.project_graph(network)
network = network.to_undirected()

# Convert lat, lon coordinate to x,y coordinate
gpd_data = gpd.GeoDataFrame(
    data,
    geometry=gpd.points_from_xy(data["longitude"], data["latitude"]),
    crs="EPSG:4326",
)
gpd_data = gpd_data.to_crs(network.graph["crs"])

# Make project dictionary
project = dict()

for index, row in gpd_data.iterrows():
    projected_point_list = []
    edge_info_list = []

    # Check whether the data is travel or not
    if row["travel_label"] == -1:
        project[f"{index}"] = {
            "projected_points": projected_point_list,
            "edge_info": edge_info_list,
        }
        continue

    # Get point from GPS data
    point = np.array([row["geometry"].xy[0][0], row["geometry"].xy[1][0]])

    for edge in network.edges():
        # Get edge from street network
        start_node_id, end_node_id = edge
        start_point = np.array(
            [network.nodes[start_node_id]["x"], network.nodes[start_node_id]["y"]]
        )
        end_point = np.array(
            [network.nodes[end_node_id]["x"], network.nodes[end_node_id]["y"]]
        )

        # Perform projection
        projected_point = projection((start_point, end_point), point)

        # If the projection is successed,
        if not np.isnan(projected_point).any():
            # If the projected point is not so far,
            if np.linalg.norm(point - projected_point) <= distance_thres:
                projected_point_list.append(projected_point)
                edge_info_list.append(
                    {
                        "start_node": start_node_id,
                        "end_node": end_node_id,
                    }
                )
    # Save the projected point list to project
    project[f"{index}"] = {
        "projected_points": projected_point_list,
        "edge_info": edge_info_list,
    }

# Save the map projection
project = pd.DataFrame.from_dict(project, orient="index")
data["x"] = gpd_data["geometry"].x
data["y"] = gpd_data["geometry"].y
data = pd.concat([data.reset_index(drop=True), project.reset_index(drop=True)], axis=1)
data.to_csv(os.path.join(projection_dir, "Map_projection.csv"), index=False)
