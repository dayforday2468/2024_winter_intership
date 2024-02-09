from modules.GPS_clustering import *

# Set the parameters for the clusters
mu = [0, 0]
sigma = [[1, 0], [0, 1]]
N = 100

# Create an instance of the ClusterGenerator and an instance of the Classifier
cluster_gen = ClusterGenerator(seed=42)
classifier = Classifier()

# Create two clusters
cluster = cluster_gen.generate_cluster(mu, sigma, N)

# test neighbor
label = classifier.neighbor(cluster, np.array([0, 0]), 1)

# show the initial data
cluster_gen.show_clusters([cluster])

# show the clustered data
classifier.show_clusters(cluster, label)
