import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import os
import ast
import matplotlib.pyplot as plt
import itertools

root_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(root_dir, "Projection")
distance_sigma = 30
ratio_sigma = 1.5


def create_transition_matrix(data, network, from_points, to_points):
    m, n = len(from_points), len(to_points)
    transition_matrix = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            # Get original data
            original_from_data = data.loc[int(from_points[i].split("_")[1])]
            original_to_data = data.loc[int(to_points[j].split("_")[1])]

            # Get projected_data
            projected_from_data = network.nodes[from_points[i]]
            projected_to_data = network.nodes[to_points[j]]

            # Compute distance likelihood
            to_d = np.sqrt(
                (projected_to_data["x"] - original_to_data["x"]) ** 2
                + (projected_to_data["y"] - original_to_data["y"]) ** 2
            )
            # print(f"to_d: {to_d}")
            to_d_likelihood = np.exp(-to_d / distance_sigma)

            # Compute ratio likelihood
            projected_length = np.sqrt(
                (projected_to_data["x"] - projected_from_data["x"]) ** 2
                + (projected_to_data["y"] - projected_from_data["y"]) ** 2
            )
            original_length = np.sqrt(
                (original_to_data["x"] - original_from_data["x"]) ** 2
                + (original_to_data["y"] - original_from_data["y"]) ** 2
            )
            length_ratio = projected_length / original_length
            # print(f"length_ratio: {length_ratio}")
            length_ratio_likelihood = np.exp(-np.abs(length_ratio - 1) / ratio_sigma)

            # Fill transition matrix
            transition_matrix[i, j] = to_d_likelihood * length_ratio_likelihood

    # Normalize transition matrix
    transition_matrix = transition_matrix / np.sum(
        transition_matrix, axis=1, keepdims=True
    )
    return transition_matrix


def viterbi(transition_matrices):
    nodes = [transition_matrices[0]]
    best_path = [np.argmax(nodes[0], axis=1)[0]]
    cursor = 0
    while cursor != len(transition_matrices) - 1:
        cursor += 1
        nodes.append(np.matmul(nodes[-1], transition_matrices[cursor]))
        best_path.append(np.argmax(nodes[-1], axis=1)[0])
    print(best_path)
    return best_path


# Read project.csv
project = pd.read_csv(os.path.join(project_dir, "Map_projection.csv"))

# Convert columns in project.csv
project["edge_info"] = project["edge_info"].apply(ast.literal_eval)
project["projected_points"] = project["projected_points"].apply(ast.literal_eval)

# Load the street network
place_name = "유성구"
network = ox.graph_from_place(place_name, network_type="drive", simplify=False)
network = ox.project_graph(network)
network = network.to_undirected()

for index, row in project.iterrows():
    # Iterate over projected points and edge information
    for projected_point, edge_info in zip(row["projected_points"], row["edge_info"]):
        # Extract information from edge_info
        start_node_id = edge_info["start_node"]
        end_node_id = edge_info["end_node"]
        edge_attributes = network[start_node_id][end_node_id][0]

        # Create a new node for the projected point
        new_node_id = f"projected_{index}_{start_node_id}_{end_node_id}"
        network.add_node(
            new_node_id,
            x=projected_point[0],
            y=projected_point[1],
        )

        # add edges(with length update)
        edge_attributes["length"] = ox.distance.euclidean(
            network.nodes[start_node_id]["x"],
            network.nodes[start_node_id]["y"],
            projected_point[0],
            projected_point[1],
        )
        network.add_edge(start_node_id, new_node_id, **edge_attributes)
        edge_attributes["length"] = ox.distance.euclidean(
            network.nodes[end_node_id]["x"],
            network.nodes[end_node_id]["y"],
            projected_point[0],
            projected_point[1],
        )
        network.add_edge(new_node_id, end_node_id, **edge_attributes)

# Remove the old edges
for _, row in project.iterrows():
    for edge_info in row["edge_info"]:
        start_node_id = edge_info["start_node"]
        end_node_id = edge_info["end_node"]
        if network.has_edge(start_node_id, end_node_id):
            network.remove_edge(start_node_id, end_node_id)

# Plot the updated network
node_color = ["r" if "projected" in str(node) else "b" for node in network.nodes]
fig, ax = ox.plot_graph(network, node_color=node_color, node_size=1, bgcolor="w")
plt.show()

# Get the number of travels
n_travel = project["travel_label"].max() + 1

# Perform viterbi algorithm for each travel
for index in range(n_travel):
    transition_matrices = []
    transition_matrix = None

    # Get travel index
    travel_index = np.where(project["travel_label"] == index)[0]

    # Get projected points
    projected_points_list = []
    for i in travel_index:
        projected_points = []
        for node in network.nodes:
            if f"projected_{i}" in str(node):
                projected_points.append(node)
        projected_points_list.append(projected_points)

    # Make transition matrices
    for i, projected_points in enumerate(projected_points_list[:-1]):
        # Skip if the projected points list is empty
        if not projected_points:
            continue

        # Handle the initial case
        if i == 0:
            transition_matrix = np.zeros((1, len(projected_points)))

            last_non_travel_data = project.loc[
                int(projected_points[0].split("_")[1]) - 1
            ]

            for j in range(len(projected_points)):
                original_to_data = project.loc[int(projected_points[j].split("_")[1])]
                projected_to_data = network.nodes[projected_points[j]]

                to_d = np.sqrt(
                    (projected_to_data["x"] - original_to_data["x"]) ** 2
                    + (projected_to_data["y"] - original_to_data["y"]) ** 2
                )
                to_d_likelihood = np.exp(-to_d / distance_sigma)

                length_ratio = np.sqrt(
                    (
                        (projected_to_data["x"] - last_non_travel_data["x"]) ** 2
                        + (projected_to_data["y"] - last_non_travel_data["y"]) ** 2
                    )
                    / (
                        (original_to_data["x"] - last_non_travel_data["x"]) ** 2
                        + (original_to_data["y"] - last_non_travel_data["y"]) ** 2
                    )
                )
                length_ratio_likelihood = np.exp(
                    -np.abs(length_ratio - 1) / ratio_sigma
                )

                transition_matrix[0, j] = to_d_likelihood * length_ratio_likelihood
            transition_matrices.append(transition_matrix)

        # Make transitino matrix
        from_points = projected_points
        to_points = projected_points_list[i + 1]
        transition_matrix = create_transition_matrix(
            project, network, from_points, to_points
        )

        # Append transition matrix
        transition_matrices.append(transition_matrix)

    # Backtrace the best projection
    viterbi(transition_matrices)
