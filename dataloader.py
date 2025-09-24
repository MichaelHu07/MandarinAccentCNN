import os
import librosa
import librosa.display
import numpy  as np
import torch
import torch.utils.data.dataset
import torch.utils.data.dataloader
import torch.nn.functional as F

from accentdataset import AccentDataSet

def collate_fn(batch):
    spectrograms, labels = zip(*batch)
    max_length = max(s.shape[-1] for s in spectrograms)

    padded_spectrograms = []
    for s in spectrograms:
        if s.shape[-1] < max_length:
            pad_size = max_length - s.shape[-1]
            s = F.pad(s, (0, pad_size))
        padded_spectrograms.append(s)

    spectrograms_tensor = torch.stack(padded_spectrograms)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return spectrograms_tensor, labels_tensor



def load_data():
    # load data from folder into numpy array
    folder_path = os.path.join(os.getcwd(), r"data\WAV")
    stft_path = os.path.join(os.getcwd(), r"data\STFT")
    os.makedirs(stft_path, exist_ok=True)


    for filename in os.listdir(folder_path):  # looping through every file in WAV folder
        if filename.endswith(".wav"):
            stft_file = os.path.join(stft_path, filename.replace(".wav", "_1.npy"))
            file_path = os.path.join(folder_path, filename)

            if os.path.exists(stft_file):
                continue
            else:
                y, sr = librosa.load(file_path, sr=16000)  # obtaining y (amplitude), and sr (sample rate)
                intervals = librosa.effects.split(y, top_db=30)
                processed = np.concatenate([y[start:end] for start, end in intervals])
                d = np.abs(librosa.stft(processed, n_fft=400, hop_length=160, win_length=400))
                np.save(stft_file, d)
                print(f"Processed: {filename}")


    for filename in os.listdir(stft_path):
        path = os.path.join(stft_path, filename)
        if filename.endswith(".npy"):
            print(f"\'{filename}\' Loaded")

    spk_filepath = os.path.join(os.getcwd(), r"data\SPKINFO.txt")

    training_dataset = AccentDataSet(stft_path, spk_filepath, 8000)

    print("Loading complete")

    return training_dataset

