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
    def __init__(self, stft_path, label_file, transform = None, target_transform = None):
        self.stft_path = stft_path
        self.files = [f for f in os.listdir(stft_path) if f.endswith(".npy")]
        labels = pd.read_csv(label_file, sep= "\t")
        self.spk_id2region = dict(zip(labels["SPEAKER_ID"], labels["PLACE_OF_BIRTH"]))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.stft_path, file_name)
        stft_file = np.load(file_path)
        spk_id = file_name.split(sep = "_")[3]
        stft_label = self.spk_id2region.get(spk_id, None)
        if stft_label is None:
            raise KeyError(f"No label found for {spk_id}")


        if self.transform:
            stft_file = self.transform(stft_file)
        else:
            stft_file = torch.tensor(stft_file, dtype = torch.float32).unsqueeze(0)
        if self.target_transform:
            stft_label = self.target_transform(stft_label)

        return stft_file, stft_label


