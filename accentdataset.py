import os
import numpy  as np
import pandas as pd
import torch
import torch.utils.data.dataset
import torch.utils.data.dataloader
from torch.utils.data import Dataset



class AccentDataSet(Dataset):
    def __init__(self, stft_path, label_file, hop_length, transform = None, target_transform = None, sr = 16000, chunk_duration = 1):
        self.stft_path = stft_path
        self.files = [f for f in os.listdir(stft_path) if f.endswith(".npy")] # array of .npy filenames
        labels = pd.read_csv(label_file, sep= "\t") # converts csv file into a table with columns and headers
        self.spk_id2region = dict(zip(labels["SPEAKER_ID"], labels["PLACE_OF_BIRTH"]))
        self.transform = transform
        self.target_transform = target_transform
        self.segment_length = int(sr * chunk_duration)
        self.hop_length = hop_length


        # mapping id to each unique region/PLACE_OF_BIRTH
        unique_regions = sorted(set(self.spk_id2region.values()))
        self.region2id = { region: i for i, region in enumerate(unique_regions) }
        self.id2region = { i: region for region, i in self.region2id.items()}

    def __len__(self):
        total_segments = 0
        for file in self.files:
            data = np.load(os.path.join(self.stft_path, file))
            length = data.shape[-1]
            if self.segment_length is None or length < self.segment_length:
                total_segments += 1
            else:
                total_segments += 1 + int(max(0, (length - self.segment_length) // self.hop_length))
        return total_segments

    def label_length(self):
        return len(self.region2id)

    def __getitem__(self, idx):
        cumulative = 0
        for file in self.files:
            d = np.load(os.path.join(self.stft_path, file))
            length = d.shape[-1]
            if self.segment_length is None or length <= self.segment_length:
                n_segs = 1
            else:
                n_segs = 1 + int(max(0, (length - self.segment_length) // self.hop_length))

            if idx < cumulative + n_segs:
                seg_idx = idx - cumulative
                seg_start = int(seg_idx * self.hop_length)
                seg_end = seg_start + self.segment_length
                segment = d[..., seg_start:seg_end]
                if segment.shape[-1] < (self.segment_length or length):
                    pad_amt = (self.segment_length or length) - segment.shape[-1]
                    segment = np.pad(segment, ((0,0),(0,pad_amt)), mode = 'constant')
                tensor_seg = torch.as_tensor(segment, dtype=torch.float32).unsqueeze(0)
                filename = os.path.basename(file)
                spk_id = filename.split("_")[3]
                region = self.spk_id2region[spk_id]
                label = self.region2id[region]
                if self.transform:
                    tensor_seg = self.transform(tensor_seg)
                if self.target_transform:
                    label = self.target_transform(label)
                label = torch.tensor(label, dtype = torch.long)
                return tensor_seg, label
            else:
                cumulative += n_segs
        raise IndexError(f"Idx {idx} Not Found")

