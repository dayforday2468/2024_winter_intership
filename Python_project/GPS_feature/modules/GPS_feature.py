import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from .GPS_clustering import *


class FeatureMaker:
    def __init__(self):
        pass

    def location_variance(self, data):
        longitude_variance = data["longitude"].std()
        latitude_variance = data["latitude"].std()
        return np.log2(longitude_variance + latitude_variance)

    def total_distance(self, data):
        diff_longitude = (data["longitude"].diff()).iloc[1:]
        diff_latitude = (data["latitude"].diff()).iloc[1:]
        return np.sum(np.sqrt(np.square(diff_longitude) + np.square(diff_latitude)))

    def speed(self, data):
        diff_longitude = (data["longitude"].diff()).iloc[1:]
        diff_latitude = (data["latitude"].diff()).iloc[1:]
        diff_time = (data["obtained_at"].diff()).iloc[1:] / pd.Timedelta(seconds=1)
        diff_longitude = diff_longitude[diff_time != 0]
        diff_latitude = diff_latitude[diff_time != 0]
        diff_time = diff_time[diff_time != 0]
        return np.sqrt(
            np.square(diff_longitude / diff_time) + np.square(diff_latitude / diff_time)
        )

    def speed_mean(self, data):
        return np.mean(self.speed(data))

    def speed_variance(self, data):
        return np.std(self.speed(data))

    def number_of_clusters(self, data, rad=20000, thres=4):
        classifier = Classifier()
        np_data = data[["latitude", "longitude"]].to_numpy()
        label = classifier.DBSCAN(np_data, rad, thres)
        return np.max(label) + 1

    def __time(self, data, label):
        N = np.max(label) + 1
        time_list = np.array([])
        for i in range(N):
            index = np.where(label == i)[0]
            index = np.setdiff1d(index, np.array([len(label) - 1]))
            time = np.sum(
                np.diff(data["obtained_at"].to_numpy())[index] / pd.Timedelta(seconds=1)
            )
            time_list = np.append(time_list, [time])
        return time_list

    def entropy(self, data, rad=20000, thres=4):
        classifier = Classifier()
        np_data = data[["latitude", "longitude"]].to_numpy()
        label = classifier.DBSCAN(np_data, rad, thres)
        time_list = self.__time(data, label)
        percentage = time_list / np.sum(time_list)
        return -np.sum(percentage * np.log2(percentage))

    def normalized_entropy(self, data, rad=20000, thres=4):
        classifier = Classifier()
        np_data = data[["latitude", "longitude"]].to_numpy()
        label = classifier.DBSCAN(np_data, rad, thres)
        time_list = self.__time(data, label)
        percentage = time_list / np.sum(time_list)
        return -np.sum(percentage * np.log2(percentage)) / np.log2(len(time_list))

    def home_stay(self, data, home_label, rad):
        home_location = home_label[["latitude", "longitude"]].to_numpy()
        classifier = Classifier()
        np_data = data[["latitude", "longitude"]].to_numpy()
        label = classifier.neighbor(np_data, home_location, rad)
        home_index = np.where(label == 1)[0]
        diff_time = (data["obtained_at"].diff()).iloc[1:].reset_index(
            drop=True
        ) / pd.Timedelta(seconds=1)

        # Remove the last index if it is contained.
        if diff_time.shape[0] <= np.max(home_index):
            home_index = home_index[:-1]
        return np.sum(diff_time.iloc[home_index]) / 86400

    def transition_time(self, data, rad=20000, thres=4):
        classifier = Classifier()
        np_data = data[["latitude", "longitude"]].to_numpy()
        label = classifier.DBSCAN(np_data, rad, thres)
        transition_index = np.where(label == -1)[0]
        diff_time = (data["obtained_at"].diff()).iloc[1:].reset_index(
            drop=True
        ) / pd.Timedelta(seconds=1)

        # Remove the last index if it is contained.
        if diff_time.shape[0] <= np.max(transition_index):
            transition_index = transition_index[:-1]
        return np.sum(diff_time[transition_index]) / 86400

    def circadian_movement(self, data):
        # Preprocess the data
        data["obtained_at"] = pd.to_datetime(data["obtained_at"])
        time_series = data.set_index("obtained_at")
        time_series = time_series.resample(
            "60T"
        ).mean()  # measure data for every 1 hour
        time_series = time_series.interpolate()
        time_series["latitude"] = (
            time_series["latitude"] - time_series["latitude"].median()
        )
        time_series["longitude"] = (
            time_series["longitude"] - time_series["longitude"].median()
        )

        # Perform fft
        latitude_fft = np.fft.fft(time_series["latitude"])
        longitude_fft = np.fft.fft(time_series["longitude"])

        # Compute the power spectrums
        latitude_ps = np.abs(latitude_fft) ** 2
        longitude_ps = np.abs(longitude_fft) ** 2

        # Get frequencies
        frequencies = np.fft.fftfreq(
            len(time_series), 1
        )  # since data is measured for every 1 hour

        # Plot power spectrums
        plt.figure(figsize=(12, 6))
        plt.plot(1 / frequencies, latitude_ps, label="Latitude_ps")
        plt.plot(1 / frequencies, longitude_ps, label="Longitude_ps")
        plt.xlabel("Period(hour)")
        plt.ylabel("Power spectrum")
        plt.legend()
        plt.xticks(np.arange(0, 25, step=1))
        plt.xlim(0, 25)
        plt.title("Power spectrum versus Period")
        plt.show()

        # Compute the circadian movement
        index_24 = np.where((1 / frequencies >= 23.5) & (1 / frequencies <= 24.5))[0]
        E_latitude = np.sum(latitude_ps[index_24]).astype(float)
        E_longtitude = np.sum(longitude_ps[index_24]).astype(float)
        return np.log2(E_latitude + E_longtitude)

    def __harversine(self, location1, location2):
        # Radius of the Earth in kilometers
        R = 6371.0

        # Convert latitude and longitude from degrees to radians
        lat1, lon1 = np.radians(location1["latitude"]), np.radians(
            location1["longitude"]
        )
        lat2, lon2 = np.radians(location2["latitude"]), np.radians(
            location2["longitude"]
        )

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distance = R * c

        return distance

    def activity_percentile(self, data, percentile):
        # Preprocess the data
        data.loc[:, "obtained_at"] = pd.to_datetime(data["obtained_at"])
        time_series = data.set_index("obtained_at")
        time_series = time_series.resample("10T").mean()
        time_series = time_series.interpolate()
        time_series["latitude"] = time_series["latitude"] / (10**7)
        time_series["longitude"] = time_series["longitude"] / (10**7)

        # Calculate the distance
        distance = list()
        for i in range(time_series.shape[0]):
            if i == time_series.shape[0] - 1:
                break
            distance.append(
                self.__harversine(time_series.iloc[i], time_series.iloc[i + 1])
            )

        # Calculate the activity percentile
        activity_percentile = np.cumsum(distance) / np.sum(distance)

        # Plot the activity percentile
        plt.plot(time_series.index[:-1].hour, activity_percentile)
        plt.xlabel("time")
        plt.ylabel("activity percentile")
        plt.show()

        # Calculate the index when the activity percentile is archived
        index = 0
        while True:
            if activity_percentile[index] >= percentile:
                break
            index += 1

        return time_series.index[index]
