import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import cv2
import requests
from PIL import Image
import random
import os
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import albumentations as A


num_bins = 25
samples_per_bin = 200
datadir = "C:\\Users\\USER\\Desktop\\SmartTechCA2_XuTeckTan\\DataForCar\\"

def main():
    data = load_data()
    print(f"Total driving samples: {len(data)}")
    bin_and_plot_data(data)

def bin_and_plot_data(data):
    hist, bins = np.histogram(data["steering"], num_bins)
    print(bins)
    centre = (bins[:-1] + bins[1:]) * 0.5
    plt.bar(centre, hist, width=0.05)
    plt.show()
    return bins, centre

def load_data():
    columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
    data = pd.read_csv(os.path.join(datadir, "driving_log.csv"), names=columns)
    pd.set_option("display.width", None)
    data["center"] = data["center"].apply(path_leaf)
    data["left"] = data["left"].apply(path_leaf)
    data["right"] = data["right"].apply(path_leaf)
    return data

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail

if __name__ == "__main__":
    main()