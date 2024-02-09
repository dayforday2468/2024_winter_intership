from modules.GPS_clustering import *
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, "Data")
travel_dir = os.path.join(root_dir, "Travel")

# Read data_smoothed.csv
data = pd.read_csv(os.path.join(data_dir, "data_smoothed.csv"))
data["obtained_at"] = pd.to_datetime(data["obtained_at"])
data["created_at"] = pd.to_datetime(data["created_at"])

# Proprocess to perform DBSCAN clustering
np_data = data[["longitude", "latitude"]].to_numpy()

# Create the classifier instance
classifier = Classifier()

# Perform DBSCAN clustering
label = classifier.DBSCAN(np_data, rad=0.0025, thres=8)

# Plot clustering
classifier.show_clusters(np_data, label)


def travel_label(label):
    # Get travel indicies
    travel = np.where(label == -1)[0]

    # Split the travel
    split_travel = np.split(travel, np.where(np.diff(travel) != 1)[0] + 1)

    # Make travel label
    result = np.array([-1] * len(label))

    for i, indices in enumerate(split_travel):
        result[indices] = i

    return result


# Save the data
data["travel_label"] = travel_label(label)
data.to_csv(os.path.join(travel_dir, "travel.csv"), index=False)
