from modules.GPS_feature import *

# Read data.csv
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.join(root_dir, "Data"), "data.csv")
data = pd.read_csv(data_dir)
data["obtained_at"] = pd.to_datetime(data["obtained_at"])
data["created_at"] = pd.to_datetime(data["created_at"])

# Read label.csv
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.join(root_dir, "Data"), "label.csv")
label = pd.read_csv(data_dir, encoding="cp949")

# select home label and target data
home_label = label[label["name"] == "홈"]
print("home location")
print(home_label)
selected_data = data[
    (data["obtained_at"].dt.year == 2024)
    & (data["obtained_at"].dt.month == 1)
    & (data["obtained_at"].dt.day == 1)
]
print("Selected_data")
print(selected_data)

# Create an instance of FeatureMaker
featuremaker = FeatureMaker()

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
location_variance = featuremaker.location_variance(selected_data)
total_distance = featuremaker.total_distance(selected_data)
speed_variance = featuremaker.speed_variance(selected_data)
speed_mean = featuremaker.speed_mean(selected_data)
entropy = featuremaker.entropy(selected_data)
normalized_entropy = featuremaker.normalized_entropy(selected_data)
number_of_clusters = featuremaker.number_of_clusters(selected_data)
home_stay = featuremaker.home_stay(selected_data, home_label=home_label, rad=5000)
transition_time = featuremaker.transition_time(selected_data)
activity_percentile = featuremaker.activity_percentile(selected_data, 0.25)

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
np_data = selected_data[["latitude", "longitude"]].to_numpy()
label = classifier.neighbor(
    np_data,
    point=home_label[["latitude", "longitude"]].to_numpy(),
    rad=5000,
)
classifier.save_clusters(
    np_data,
    label,
    path=os.path.join(plot_dir, "home_neighbor.png"),
    title="home_neighbor",
)

# Save the DBSCAN clustering plot
classifier = Classifier()
np_data = selected_data[["latitude", "longitude"]].to_numpy()
label = classifier.DBSCAN(np_data, rad=20000, thres=4)
classifier.save_clusters(
    np_data,
    label,
    path=os.path.join(plot_dir, "DBSCAN_clustering.png"),
    title="DBSCAN_clustering",
)

# save the report
feature_dir = os.path.join(os.path.join(root_dir, "Feature"), "feature.csv")
report.to_csv(feature_dir, index=False)
