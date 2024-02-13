from modules.GPS_feature import *

# Read data.csv
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.join(root_dir, "Data"), "data_smoothed.csv")
data = pd.read_csv(data_dir)
data["obtained_at"] = pd.to_datetime(data["obtained_at"])
data["created_at"] = pd.to_datetime(data["created_at"])
print("data")
print(data)

# Read label.csv
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.join(root_dir, "Data"), "label.csv")
label = pd.read_csv(data_dir)

# select home label and target data
home_label = label[label["name"] == "Home"]
print("home_label")
print(home_label)

# Create an instance of FeatureMaker
featuremaker = FeatureMaker(rad=0.002, thres=5, home_rad=0.002)

# Create a report dataframe
report = pd.DataFrame().reindex(
    columns=[
        "location_variance",
        "total_distance",
        "speed_variance",
        "speed_mean",
        "entropy",
        "normalized_entropy",
        "number_of_clusters",
        "home_stay",
        "transition_time",
        "activity_percentile",
    ]
)

# Calculate features
location_variance = featuremaker.location_variance(data)
total_distance = featuremaker.total_distance(data)
speed_variance = featuremaker.speed_variance(data)
speed_mean = featuremaker.speed_mean(data)
entropy = featuremaker.entropy(data)
normalized_entropy = featuremaker.normalized_entropy(data)
number_of_clusters = featuremaker.number_of_clusters(data)
home_stay = featuremaker.home_stay(data, home_label=home_label)
transition_time = featuremaker.transition_time(data)
activity_percentile = featuremaker.activity_percentile(data, 0.25)

# Record features to report
report.loc[len(report)] = np.array(
    [
        location_variance,
        total_distance,
        speed_variance,
        speed_mean,
        entropy,
        normalized_entropy,
        number_of_clusters,
        home_stay,
        transition_time,
        activity_percentile,
    ]
)
print(report)

# Make plot directory
plot_dir = os.path.join(root_dir, "Plot")

# Save the home neighbor plot
classifier = Classifier()
np_data = data[["latitude", "longitude"]].to_numpy()
label = classifier.neighbor(
    np_data,
    point=home_label[["latitude", "longitude"]].to_numpy(),
    rad=featuremaker.home_rad,
)
classifier.save_clusters(
    np_data,
    label,
    path=os.path.join(plot_dir, "home_neighbor.png"),
    title="home_neighbor",
)

# Save the DBSCAN clustering plot
classifier = Classifier()
np_data = data[["latitude", "longitude"]].to_numpy()
label = classifier.DBSCAN(np_data, rad=featuremaker.rad, thres=featuremaker.thres)
classifier.save_clusters(
    np_data,
    label,
    path=os.path.join(plot_dir, "DBSCAN_clustering.png"),
    title="DBSCAN_clustering",
)

# save the report
feature_dir = os.path.join(os.path.join(root_dir, "Feature"), "feature.csv")
report.to_csv(feature_dir, index=False)
