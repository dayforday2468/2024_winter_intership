from modules.GPS_feature import *

distance_threshold = 0.75
jump_threshold = 4
same_place_threshold = 0.3

# Read data.csv
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.join(root_dir, "Data"), "data.csv")
data = pd.read_csv(data_dir)
data["obtained_at"] = pd.to_datetime(data["obtained_at"])
data["created_at"] = pd.to_datetime(data["created_at"])

# select target data
selected_data = data[
    (data["obtained_at"].dt.year == 2024)
    & (data["obtained_at"].dt.month == 1)
    & (data["obtained_at"].dt.day == 16)
]

# Create an instance of featuremaker
featuremaker = FeatureMaker()

# Preprocess the data
selected_data.loc[:, "obtained_at"] = pd.to_datetime(selected_data["obtained_at"])
selected_data.loc[:, "longitude"] = selected_data["longitude"] / (10**7)
selected_data.loc[:, "latitude"] = selected_data["latitude"] / (10**7)

# Plot the selected data
plt.subplot(1, 2, 1)
plt.title("Before cleaning")
plt.plot(selected_data["longitude"], selected_data["latitude"])

# Calculate the distance
distance = list()
for i in range(selected_data.shape[0]):
    if i == selected_data.shape[0] - 1:
        break
    distance.append(
        featuremaker.harversine(selected_data.iloc[i], selected_data.iloc[i + 1])
    )

# Split the data to segment by distance
segments = list()
i = 0
segment = []
while i < len(distance):
    if distance[i] > distance_threshold:
        segments.append(segment)
        segment = []
    segment.append(i)
    i += 1
segment.append(i)
segments.append(segment)

# Calculate the mean for each segment
segments_mean = [
    selected_data[["longitude", "latitude"]].iloc[segment].mean()
    for segment in segments
]


# Indicate the jump
jumps = list()
for i in range(1, len(segments_mean) - 1):
    if (
        len(segments_mean[i]) <= jump_threshold
        and featuremaker.harversine(segments_mean[i - 1], segments_mean[i + 1])
        < same_place_threshold
    ):
        jumps.append(i)

# Filter the selected_data only if there are jumps
jump_index = np.array([])
if len(jumps) > 0:
    jump_index = np.concatenate([segments[jump] for jump in jumps])
    jump_index += 1
    selected_data_filtered = pd.DataFrame(np.delete(selected_data, jump_index, axis=0))
    selected_data_filtered.columns = selected_data.columns
else:
    selected_data_filtered = selected_data.copy()

# Plot the filtered data
plt.subplot(1, 2, 2)
plt.title("After cleaning")
plt.plot(selected_data_filtered["longitude"], selected_data_filtered["latitude"])
for i in jump_index:
    plt.scatter(
        selected_data["longitude"].iloc[i],
        selected_data["latitude"].iloc[i],
        c="r",
    )
plt.show()

# Save the filtered data
filtered_dir = os.path.join(root_dir, "Filtered_data")
selected_data_filtered.to_csv(
    os.path.join(filtered_dir, "data_cleaned.csv"), index=False
)
