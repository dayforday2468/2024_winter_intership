from modules.GPS_feature import *

# Read data.csv
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.join(root_dir, "Data"), "location_230110.csv")
data = pd.read_csv(data_dir)
data = data.iloc[:, [1, 2]].to_numpy() * 10**6
print("Original data")
print(data)

# Show the clustering
classifier = Classifier()
label = classifier.DBSCAN(data, rad=20000, thres=3)
classifier.show_clusters(data, label)
