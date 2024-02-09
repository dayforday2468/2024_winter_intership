from modules.GPS_feature import *

# Read data.csv
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.join(root_dir, "Data"), "data.csv")
data = pd.read_csv(data_dir)
data["obtained_at"] = pd.to_datetime(data["obtained_at"])
data["created_at"] = pd.to_datetime(data["created_at"])

# Select some data from data
selected_data = data[
    (data["obtained_at"].dt.year == 2023) & (data["obtained_at"].dt.month == 12)
]
print("Selected_data")
print(selected_data)

# Show the selected data
plt.plot(selected_data["obtained_at"], selected_data["latitude"])
plt.title("latitude")
plt.show()
plt.plot(selected_data["obtained_at"], selected_data["longitude"])
plt.title("longitude")
plt.show()

# Create the instance of FeatureMakter
featuremaker = FeatureMaker()

# Compute the circadian movement
circadian_movement = featuremaker.circadian_movement(selected_data)
print("Compute the circadian movement")
print(circadian_movement)
