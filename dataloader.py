import os
import librosa
import librosa.display
import numpy as np
import torch
import io
import torch.utils.data.dataset
import torch.utils.data.dataloader
import torch.nn.functional as F
import tarfile 
from pathlib import Path
import pandas as pd
import lmdb, struct

from accentdataset import AccentDataSet

def collate_fn(batch):
    waveforms, labels = zip(*batch)
    
    waveforms = [torch.tensor(w).squeeze() for w in waveforms]
    
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    
    batch_labels = torch.tensor(labels, dtype=torch.long)
    return padded_waveforms, batch_labels


def load_data():
    folder_path = os.path.join(os.getcwd(), r"data\WAV")

    segment_length = 64000
    hop_length = 32000 

    lmdb_path = os.path.join(Path(__file__).parent, "data", "LMDB_WAV")
    os.makedirs(lmdb_path, exist_ok=True)
    
    has_data = False
    if os.path.exists(os.path.join(lmdb_path, "data.mdb")):
        env_check = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env_check.begin() as txn:
            if txn.stat()['entries'] > 0:
                has_data = True
        env_check.close()

    if not has_data:
        print("Creating LMDB from WAV files...")
        with lmdb.open(lmdb_path, map_size=10737418240) as env: # 10GB disk space
            with env.begin(write=True) as txn:
                idx = 0
                for filename in os.listdir(folder_path):
                    if not filename.endswith(".wav"):
                        continue
                        
                    file_path = os.path.join(folder_path, filename)
                    try:
                        y, sr = librosa.load(file_path, sr=16000)
                        
                        intervals = librosa.effects.split(y, top_db=30)
                        processed = np.concatenate([y[start:end] for start, end in intervals])

                        # Filename format: G0001_S0001_0_SPK002.wav
                        spk_id_str = filename.split("_")[3]
                        spk_id = spk_id_str.replace(".wav", "")
                        spk_id_int = int(spk_id[3:]) # SPK002 -> 2

                        length = len(processed)
                        
                        current_idx = 0
                        while current_idx < length:
                            seg_end = current_idx + segment_length
                            segment = processed[current_idx:seg_end]
                            
                            if len(segment) > 0:
                                if len(segment) < segment_length:
                                    segment = np.pad(segment, (0, segment_length - len(segment)), mode='constant')
                                
                                segment = segment.astype(np.float32)
                                
                                buf = io.BytesIO()
                                np.save(buf, segment, allow_pickle=False)
                                
                                key = f"{idx}"
                                combined_data = struct.pack('I', spk_id_int) + buf.getbuffer()
                                txn.put(key.encode(), combined_data)
                                idx += 1
                            
                            current_idx += hop_length
                            
                        print(f"Processed: {filename}")
                        
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        continue

    spk_filepath = os.path.join(Path(__file__).parent, "data", "SPKINFO.txt")
    training_dataset = AccentDataSet(lmdb_path, spk_filepath)

    print("Loading complete")

    return training_dataset
