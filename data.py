
import os
import glob
from matplotlib import pyplot

import numpy as np
# import signal_processing as sp
from itertools import tee
from skimage import util
import csv
from sklearn import preprocessing
from sklearn.utils import shuffle
from scipy.signal import resample


INPUT_SIGNAL_TYPES = [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z"
]

def make_dataset(gesture_path, save_dir, pattern, winSize = 128,step = 2):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = load_data(gesture_path)
    # data = sp.despikeSeries(data)
    x_train_signals_paths = [
        os.path.join(save_dir ,signal + ".csv" )for signal in INPUT_SIGNAL_TYPES
    ]
    x_train_files = [csv.writer(open( x_train_signals_path,"a",newline=''))for x_train_signals_path in x_train_signals_paths]
    y_train_file = csv.writer(open(os.path.join(save_dir,"y.csv"),"a",newline=''))

    for i ,signals in enumerate(data):
        segmented_signal = util.view_as_windows(signals, (winSize,),step = step)
        for signal in segmented_signal:
            x_train_files[i].writerow(signal)
            if i ==0:
                y_train_file.writerow([pattern])


def load_x_dataset(x_signals_paths):
    x_signals = []

    for signal_type_path in x_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        x_signals.append(
            [resample(np.array(serie, dtype=np.float32),128) for serie in [
                row.split(',') for row in file
            ]]
        )
        file.close()

    x_data=  np.array(x_signals,dtype=np.float32)
    x_data = np.transpose(x_data, (1, 0, 2))

    scaled_x_data = []
    for row in x_data:
        scaled_x = []
        for d in row:
            d = preprocessing.scale(d)
            scaled_x.append(d)
        scaled_x= np.array(scaled_x,dtype=np.float32)
        scaled_x_data.append(scaled_x)
    return np.array(scaled_x_data,dtype=np.float32)
def load_y_dataset(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]],
        dtype=np.int32
    )
    file.close()
    # Substract 1 to each output class for friendly 0-based indexing
    return y_

def load_data(gesture_path):
    x_data = []
    with open(gesture_path) as f:
        for row in f:
            x_data.append(np.array(row.strip().split(","))[1:].astype(float))
    x_data = np.array(x_data,dtype=np.float32).transpose()
    return x_data
    # scaled_x_data = []
    # for x in x_data:
    #     scaled_x = preprocessing.scale(x)
    #     scaled_x_data.append(scaled_x)
    # return np.array(scaled_x_data,dtype=np.float32)


def one_hot(y_):
    """
    Function to encode output labels from number indexes.
    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

import math
def load_dataset(gesture_paths, y_path):
    data = load_x_dataset(gesture_paths)
    y = one_hot(load_y_dataset(y_path))

    data, y = shuffle(data, y, random_state=0)


    return data ,y


def load_dataset(gesture_paths, y_path,split=80):
    data = load_x_dataset(gesture_paths)
    y = one_hot(load_y_dataset(y_path))

    data, y = shuffle(data, y, random_state=0)

    split_index = int(math.floor(len(data) * 0.8))

    x_train = data[:split_index]
    x_test = data[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return x_train, x_test , y_train, y_test

import random
def random_crop(data,y,max_gen = 10, crop_w = 10):
    gen_x_data = []
    gen_y_data = []
    for i in range(len(data)):
        x_signals = data[i]
        y_signals = y[i]

        n_gen = 1
        for n in range(n_gen):
            left_tw = random.randint(0, crop_w)
            right_tw = random.randint(0, crop_w)
            gen_x_signals = np.resize(x_signals[:, crop_w:x_signals.shape[1] - right_tw],(6,128))
            gen_x_data.append(gen_x_signals)
            gen_y_data.append(y_signals)
    return np.array(gen_x_data,dtype=np.float32), np.array(gen_y_data,dtype=np.float32)
def data_resize(x_data,size):
    resized_x_data = []
    for row in x_data:
        resized_x = []
        for d in row:
            d = np.resize(d,size)
            resized_x.append(d)
        resized_x = np.array(resized_x, dtype=np.float32)
        resized_x_data.append(resized_x)
    return np.array(resized_x_data, dtype=np.float32)

def load_lawdata(gesture_path):
    x_signals = []
    file = open(gesture_path, 'r')
    x_signals.append(
        [np.array(serie, dtype=np.float32) for serie in [
            row.strip().split(',') for row in file
        ]]
    )
    file.close()
    x_signals = np.array(x_signals)
    return x_signals
if __name__ == "__main__":
    save_dir = './train'
    x_train_signals_paths = [
        os.path.join(save_dir, signal + ".csv") for signal in INPUT_SIGNAL_TYPES
    ]
    y_path = os.path.join(save_dir, "y.csv")
    x_train, y_train = load_dataset(x_train_signals_paths,y_path)

    x_train, y_train = random_crop(x_train,y_train)
    print(x_train)
    # data = load_x_dataset(x_train_signals_paths)
    # y = one_hot(load_y_dataset(y_path))

    # print(data)