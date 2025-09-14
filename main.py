#pytorch imports

import torch
import torchvision
from numba.np.npyfuncs import np_datetime_isnat_impl
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import random_split
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

from dataloader import load_data




if __name__ == '__main__':
    model = resnet18(weights = ResNet18_Weights.DEFAULT)
    full_dataset = load_data()

    split_percentage = 0.8

    train_size = int(split_percentage * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)







#Deprecated:

    #print(f"Labeled_data check: {labeled_data}")

    #print("Data Splitting commencing")
    #np_data = np.array([])
    #np_label = np.array([])
    #for i, array in enumerate(labeled_data):
    #    np_data = np.append(np_data, array[0])
    #    np_label = np.append(np_label, array[1])
    #    print(f"Splitting: {i}")

    #print(f"np_data contains: {np_data}")
    #print(f"np_label contains: {np_label}")











