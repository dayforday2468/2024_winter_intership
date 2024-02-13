from modules.GPS_clustering import *

# Set parameters for the clusters
mu1 = [10, 10]
sigma1 = [[10, 5], [5, 10]]
N1 = 50

mu2 = [-10, -10]
sigma2 = [[10, -5], [-5, 10]]
N2 = 50

# Create an instance of the ClusterGenerator and an instance of the Classifier
cluster_gen = ClusterGenerator(seed=42)
classifier = Classifier(seed=42)

# Create two clusters
cluster1 = cluster_gen.generate_cluster(mu1, sigma1, N1)
cluster2 = cluster_gen.generate_cluster(mu2, sigma2, N2)

# Combine the clusters for DBSCAN testing
combined_data = np.concatenate([cluster1, cluster2])

# test DBSCAN
label = classifier.DBSCAN(combined_data, rad=4, thres=3)

# show the true clusters
cluster_gen.show_clusters([cluster1, cluster2], title="Before DBSCAN clustering")

# show the clusterd data
classifier.show_clusters(combined_data, label, title="After DBSCAN clustering")
