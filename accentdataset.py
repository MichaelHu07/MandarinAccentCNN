import os
import numpy  as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data.dataset
import torch.utils.data.dataloader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class AccentDataSet(Dataset):
    def __init__(self, stft_path, label_file, transform = None, target_transform = None, sr = 16000, chunk_duration = 3.0):
        self.stft_path = stft_path
        self.files = [f for f in os.listdir(stft_path) if f.endswith(".npy")] # array of .npy filenames
        labels = pd.read_csv(label_file, sep= "\t") # converts csv file into a table with columns and headers
        self.spk_id2region = dict(zip(labels["SPEAKER_ID"], labels["PLACE_OF_BIRTH"]))
        self.transform = transform
        self.target_transform = target_transform


        # mapping id to each unique region/PLACE_OF_BIRTH
        unique_regions = sorted(set(self.spk_id2region.values()))
        self.region2id = { region: i for i, region in enumerate(unique_regions) }
        self.id2region = { i: region for region, i in self.region2id.items()}

    def __len__(self):
        return len(self.files)

    def label_length(self):
        return len(self.region2id)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.stft_path, file_name)
        stft_file = np.load(file_path)
        stft_file = torch.as_tensor(stft_file, dtype=torch.float32)
        if stft_file.ndim == 2:
            stft_file.unsqueeze(0)
        spk_id = file_name.split(sep = "_")[3] # filenames are *****_*****_*_spkid_1.npy
        region = self.spk_id2region.get(spk_id, None) # grabbing region from spkid to region dict, None otherwise
        if region is None:
            raise KeyError(f"No label found for {spk_id}")

        stft_label = self.region2id[region] # grabbing id from region

        if self.transform:
            stft_file = self.transform(stft_file)
        else:
            stft_file = torch.tensor(stft_file, dtype = torch.float32).unsqueeze(0) # convert np spectrogram into float32 tensor
        if self.target_transform:
            stft_label = self.target_transform(stft_label)

        stft_label = torch.tensor(stft_label, dtype = torch.long) # convert label file into tensor

        return stft_file, stft_label


