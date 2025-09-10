#pytorch imports

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchview import draw_graph
import torchaudio

#accuracy metrics

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    log_loss,
    recall_score,
    confusion_matrix
)

#model selection imports

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
    learning_curve,
    StratifiedShuffleSplit
)


import wave
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import os
import matplotlib.pyplot as plt
import matplotlib

from skimage.transform import resize
from scipy.io import wavfile
import sklearn
from tqdm.notebook import trange, tqdm

if __name__ == '__main__':
        # load data from folder into numpy array
    folder_path = os.path.join(os.path.dirname(__file__), r"data\WAV")
    wave_array = [] # wave_array contains [ y , Filename ]
    for filename in os.listdir(folder_path):  # looping through every file in WAV folder
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            y, sr = librosa.load(file_path, sr=16000)  # obtaining y (amplitude), and sr (sample rate)
            wave_array.append([y, filename])
            print(f"Loaded: {filename}, Shape: {y.shape}")

    spk_filepath = os.path.join(os.path.dirname(__file__), r"data\SPKINFO.txt")
    df = pd.read_csv(spk_filepath, sep="\t")  # convert txt to dataframe

    # categorize dataframe into each category
    info_array = df.to_numpy()
    spk_id = df["SPEAKER_ID"].to_numpy()
    genders = df["GENDER"].to_numpy()
    ages = df["AGE"].to_numpy()
    spk_region = df["PLACE_OF_BIRTH"].to_numpy()

    # print(spk_id[:5])
    # print(spk_region[:5])

    # matching wav amplitudes to speaker region
    train_test_array = [] #train_test_array contains [ y , spk_region ]
                          #y contains [ amplitudes at 16,000 sr , y.shape , dtype]
    for array in wave_array:
        filename = array[1]
        for idx, sid in enumerate(spk_id):
            if sid in filename:
                train_test_array.append([array[0], spk_region[idx]])
                break
    # print(train_test_array[:5])

    # waveform display
    #   plt.figure(figsize=(10, 4))
    # librosa.display.waveshow(wavearray[int(len(wavearray)*random.random())][0], sr = 16000)
    # plt.title("Waveform")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.show()









