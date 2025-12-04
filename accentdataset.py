import struct
import io 
import os
import numpy  as np
import lmdb 
import pandas as pd
import torch
import torch.utils.data.dataset
import torch.utils.data.dataloader
from torch.utils.data import Dataset
import threading


class AccentDataSet(Dataset):
    def __init__(self, lmdb_path, label_file, transform = None, target_transform = None):
        self.lmdb_path= lmdb_path
        self.files = [f for f in os.listdir(lmdb_path) if f.endswith(".npy")] 
        labels = pd.read_csv(label_file, sep= "\t") 
        self.spk_id2region = dict(zip(
            labels["SPEAKER_ID"].str[3:].astype(int), #SPEAKER_ID format example: SPK002, convert to 2
            labels["PLACE_OF_BIRTH"]
            ))
        self.transform = transform
        self.target_transform = target_transform
        
        self.env = lmdb.open(self.lmdb_path, readonly = True, lock = False)
    
        # mapping id to each unique region/PLACE_OF_BIRTH
        unique_regions = sorted(set(self.spk_id2region.values()))
        self.region2id = { region: i for i, region in enumerate(unique_regions) }
        self.id2region = { i: region for region, i in self.region2id.items()}

    def __len__(self):
        txn = self.env.begin(write = False)
        stats = txn.stat()
        txn.abort()
        self._length = stats['entries']
        return self._length
    
    def label_length(self):
        return len(self.region2id)

    def __getitem__(self, idx):
        txn = self.env.begin(write=False)
        combined_data = txn.get(f"{idx}".encode()) #lmdb key is are bytes of the form: 138 (IDX)
        txn.abort()

        spk_id_int = struct.unpack('I', combined_data[:4])[0] #
        data = np.load(io.BytesIO(combined_data[4:]))
            
        spk_label = self.spk_id2region[spk_id_int]
        region_id = self.region2id[spk_label]
        
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            region_id = self.target_transform(region_id)

        return data, region_id
        

            


