import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from .GPS_clustering import *


class FeatureMaker:
    def __init__(self, rad=0.002, thres=5, home_rad=0.002):
        self.rad = rad
        self.thres = thres
        self.home_rad = home_rad

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
        return np.sqrt(
            np.square(diff_longitude / diff_time) + np.square(diff_latitude / diff_time)
        )

    def speed_mean(self, data):
        return np.mean(self.speed(data))

    def speed_variance(self, data):
        return np.std(self.speed(data))

    def number_of_clusters(self, data):
        classifier = Classifier()
        np_data = data[["latitude", "longitude"]].to_numpy()
        label = classifier.DBSCAN(np_data, self.rad, self.thres)
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
        return time_list / np.sum(time_list)

    def entropy(self, data):
        classifier = Classifier()
        np_data = data[["latitude", "longitude"]].to_numpy()
        label = classifier.DBSCAN(np_data, self.rad, self.thres)
        percentage = self.__time(data, label)
        return -np.sum(percentage * np.log2(percentage))

    def normalized_entropy(self, data):
        classifier = Classifier()
        np_data = data[["latitude", "longitude"]].to_numpy()
        label = classifier.DBSCAN(np_data, self.rad, self.thres)
        percentage = self.__time(data, label)
        return -np.sum(percentage * np.log2(percentage)) / np.log2(len(percentage))

    def home_stay(self, data, home_label):
        home_location = home_label[["latitude", "longitude"]].to_numpy()
        classifier = Classifier()
        np_data = data[["latitude", "longitude"]].to_numpy()
        label = classifier.neighbor(np_data, home_location, self.home_rad)
        home_index = np.where(label == 1)[0]
        diff_time = (data["obtained_at"].diff()).iloc[1:].reset_index(
            drop=True
        ) / pd.Timedelta(seconds=1)

        # Remove the last index if it is contained.
        if diff_time.shape[0] <= np.max(home_index):
            home_index = home_index[:-1]
        return np.sum(diff_time.iloc[home_index]) / 86400

    def transition_time(self, data):
        classifier = Classifier()
        np_data = data[["latitude", "longitude"]].to_numpy()
        label = classifier.DBSCAN(np_data, self.rad, self.thres)
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(1 / frequencies, latitude_ps, label="Latitude_ps")
        ax1.set_xlabel("Period (hour)")
        ax1.set_ylabel("Power spectrum")
        ax1.legend()
        ax1.set_xticks(np.arange(0, 25, step=1))
        ax1.set_xlim(0, 25)
        ax1.set_ylim(0, 0.1)
        ax1.set_title("Latitude Power Spectrum versus Period")

        ax2.plot(1 / frequencies, longitude_ps, label="Longitude_ps")
        ax2.set_xlabel("Period (hour)")
        ax2.set_ylabel("Power spectrum")
        ax2.legend()
        ax2.set_xticks(np.arange(0, 25, step=1))
        ax2.set_xlim(0, 25)
        ax2.set_ylim(0, 1)
        ax2.set_title("Longitude Power Spectrum versus Period")

        plt.tight_layout()
        plt.show()

        # Compute the circadian movement
        index_24 = np.where((1 / frequencies >= 23.5) & (1 / frequencies <= 24.5))[0]
        E_latitude = np.sum(latitude_ps[index_24]).astype(float)
        E_longtitude = np.sum(longitude_ps[index_24]).astype(float)
        return np.log2(E_latitude + E_longtitude)

    def harversine(self, location1, location2):
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
        c = 2 * np.arcsin(np.sqrt(a))

        distance = R * c

        return distance

    def activity_percentile(self, data, percentile):
        # Preprocess the data
        data.loc[:, "obtained_at"] = pd.to_datetime(data["obtained_at"])
        time_series = data.set_index("obtained_at")
        time_series = time_series.resample("10T").mean()
        time_series = time_series.interpolate()

        # Calculate the distance
        distance = list()
        for i in range(time_series.shape[0]):
            if i == time_series.shape[0] - 1:
                break
            distance.append(
                self.harversine(time_series.iloc[i], time_series.iloc[i + 1])
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
