from modules.GPS_feature import *

sigma = 90

# Read data_cleaned.csv
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.join(root_dir, "Filtered_data"), "data_cleaned.csv")
data = pd.read_csv(data_dir)
data["obtained_at"] = pd.to_datetime(data["obtained_at"])
data["created_at"] = pd.to_datetime(data["created_at"])

# # Plot the filtered data
# plt.subplot(1, 2, 1)
# plt.plot(data["longitude"], data["latitude"])
# plt.title("Before smoothing")

# Make Gaussian kernel
diff_time = np.array(
    [data["obtained_at"] - t for t in data["obtained_at"]]
) / pd.Timedelta(seconds=sigma)
Gaussian = np.exp(-(diff_time**2) / 2)
Gaussian = np.array([i / np.sum(i) for i in Gaussian])

# Perform Gaussian kernel smoothing on longitude
data["longitude"] = np.array(
    [np.matmul(Gaussian[i], data["longitude"]) for i in range(len(Gaussian))]
)

# Perform Gaussian kernel smoothing on latitude
data["latitude"] = np.array(
    [np.matmul(Gaussian[i], data["latitude"]) for i in range(len(Gaussian))]
)

# # Plot the filtered data
# plt.subplot(1, 2, 2)
# plt.plot(data["longitude"], data["latitude"])
# plt.title("After smoothing")
# plt.show()

# Save the filtered data
filtered_dir = os.path.join(root_dir, "Filtered_data")
data.to_csv(os.path.join(filtered_dir, "data_smoothed.csv"), index=False)
