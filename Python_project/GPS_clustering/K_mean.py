from modules.GPS_clustering import *

# Set parameters for the clusters
mu1 = [1, 1]
sigma1 = [[1, 0], [0, 1]]
N1 = 50

mu2 = [-1, -1]
sigma2 = [[1, 0], [0, 1]]
N2 = 50

# Create an instance of the ClusterGenerator and an instance of the Classifier
cluster_gen = ClusterGenerator(seed=42)
classifier = Classifier()

# Create two clusters
cluster1 = cluster_gen.generate_cluster(mu1, sigma1, N1)
cluster2 = cluster_gen.generate_cluster(mu2, sigma2, N2)

# Combine the clusters for k-means testing
combined_data = np.concatenate([cluster1, cluster2])

# test k-mean
label = classifier.k_mean(combined_data, 2)

# show the initial data
cluster_gen.show_clusters([combined_data])

# show the clustered data
classifier.show_clusters(combined_data, label)

# show true clusters
cluster_gen.show_clusters([cluster1, cluster2])
