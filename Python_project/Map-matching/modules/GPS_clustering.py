import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ClusterGenerator:
    def __init__(self, seed=None):
        np.random.seed(seed)

    def generate_cluster(self, mu, sigma, N):
        return np.random.multivariate_normal(mu, sigma, N)

    def plot_clusters(self, clusters, title="Cluster visualization", save_path=None):
        # Create the new plot
        plt.figure()

        for i, cluster_points in enumerate(clusters):
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}"
            )

        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.title(title)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def show_clusters(self, clusters):
        self.plot_clusters(clusters)

    def save_clusters(self, clusters, path, title="Cluster visualization"):
        self.plot_clusters(clusters, title, save_path=path)


class Classifier:
    def __init__(self):
        pass

    def k_mean(self, data, k, iteration=10):
        # Randomly assign initial label and create the empty centroid
        label = np.random.choice(k, data.size // 2, replace=True)
        centroid = [np.array([]) for _ in range(k)]

        # Calculate the initial centroid
        for i in range(k):
            centroid[i] = np.mean(data[label == i], axis=0)

        # Iterate to find the new label
        for _ in range(iteration):
            # Calculate distances for each centroid
            distances = np.array([np.linalg.norm(data - c, axis=1) for c in centroid])

            # Update labels based on the minimum distance
            label = np.argmin(distances, axis=0)

            # Update centroids based on the assigned samples
            for i in range(k):
                centroid[i] = np.mean(data[label == i], axis=0)

        return label

    def DBSCAN(self, data, rad: float = 3, thres=4):
        label_counter = 0
        label = np.array([-1] * (data.size // 2))
        # Determine core data and noise
        distance = np.array([np.linalg.norm(data - c, axis=1) for c in data])
        core_data = np.sum(distance < rad, axis=1) >= thres
        noise = np.sum(distance < rad, axis=1) < thres

        # Assign core data
        unassigned_data = np.where(core_data == True)[0]
        cluster = np.array([unassigned_data[0]])
        while True:
            cluster = self.__extend_cluster(data, cluster, unassigned_data, rad, thres)
            label[cluster] = label_counter
            label_counter += 1
            unassigned_data = np.setdiff1d(unassigned_data, cluster)
            if unassigned_data.size == 0:
                break
            cluster = np.array([unassigned_data[0]])

        # Assign noise
        unassigned_data = np.where(noise == True)[0]
        i = 0
        while i < len(unassigned_data):
            current_i = unassigned_data[i]
            neighbor = np.where(
                (distance[current_i] < rad) & (0 < distance[current_i])
            )[0]
            nearest = np.argsort(distance[current_i][neighbor])
            updated = 0
            for j in nearest:
                if label[neighbor[j]] == -1:
                    continue
                if label[current_i] != label[neighbor[j]]:
                    label[current_i] = label[neighbor[j]]
                    updated = 1
                    break
            i += 1
            if updated:
                i = 0
        return label

    def __extend_cluster(self, data, cluster, unassigned_data, rad, thres):
        update = 0
        i = 0
        while i < cluster.size:
            current_point = cluster[i]
            distance = np.linalg.norm(
                data[unassigned_data] - data[current_point], axis=1
            )
            neighbor = unassigned_data[np.where(distance < rad)[0]]
            if neighbor.size != 0:
                update = 1
                cluster = np.union1d(cluster, neighbor)
                unassigned_data = np.setdiff1d(unassigned_data, neighbor)
            i += 1
        return cluster

    def adaptive_k_mean(self, data, k):
        # initial clusters
        nrow = data.shape[0]
        seed = np.random.choice(np.arange(nrow), k, replace=False)
        label = np.array([-1] * nrow)
        label[seed] = np.arange(k)
        centroid = data[seed]
        cluster_dis = np.array([np.linalg.norm(centroid - c, axis=1) for c in centroid])
        np.fill_diagonal(cluster_dis, np.inf)
        cluster_min_dis = np.min(cluster_dis)
        closest_clusters = np.unravel_index(cluster_dis.argmin(), cluster_dis.shape)

        # assign unassigned data
        for i in range(nrow):
            if label[i] != -1:
                continue
            to_cluster_dis = np.array([np.linalg.norm(data[i] - c) for c in centroid])
            if np.min(to_cluster_dis) <= cluster_min_dis:
                # assign unassigned data to the closest cluster
                updated_cluster = np.argmin(to_cluster_dis)
                label[i] = updated_cluster

                # update
                centroid[updated_cluster] = np.mean(data[label == updated_cluster])
            else:
                # merge two closest clusters and make new cluster with the unsigned data
                merged_cluster = closest_clusters[0]
                new_cluster = closest_clusters[1]
                label[label == new_cluster] = merged_cluster
                label[i] = new_cluster

                # update
                centroid[merged_cluster] = np.mean(data[label == merged_cluster])
                centroid[new_cluster] = data[i]

            # common update
            cluster_dis = np.array(
                [np.linalg.norm(centroid - c, axis=1) for c in centroid]
            )
            np.fill_diagonal(cluster_dis, np.inf)
            cluster_min_dis = np.min(cluster_dis)
            closest_clusters = np.unravel_index(cluster_dis.argmin(), cluster_dis.shape)
        return label

    def neighbor(self, data, point, rad):
        return (np.linalg.norm(data - point, axis=1) <= rad).astype(int)

    def plot_clusters(self, data, label, title="Cluster visualization", save_path=None):
        # Create the new plot
        plt.figure()

        # Plot unassigned data with label -1
        unassigned_data = data[label == -1]
        if len(unassigned_data) > 0:
            plt.scatter(
                unassigned_data[:, 0], unassigned_data[:, 1], label="unassigned"
            )

        # Plot data for each cluster
        for i in range(max(label) + 1):
            cluster_points = data[label == i]
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}"
            )
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.title(title)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def show_clusters(self, data, label):
        self.plot_clusters(data, label)

    def save_clusters(self, data, label, path, title="Cluster visualization"):
        self.plot_clusters(data, label, title, save_path=path)
