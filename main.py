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
from torch import optim
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
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = resnet18(weights = ResNet18_Weights.DEFAULT)
    model = model.to(device)
    full_dataset = load_data()

    for param in model.parameters():
        param.requires_grad = False

    split_percentage = 0.8

    train_size = int(split_percentage * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.fc = nn.Linear(512, full_dataset.label_length())
    model.fc.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(train_dataloader,0):
            inputs, label = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, label)
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print("Finished training")

    PATH = os.path.join(os.getcwd(), r"\MODELS")
    torch.save(model.state_dict(), PATH)

    model.load_state_dict(torch.load(PATH, weights_only=True))

    for i, data in enumerate(test_dataloader):
        spectrogram, label = data
        outputs = model(spectrogram)

        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join(f'{label[predicted[j]]:5s}'
                                      for j in range(4)))












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











