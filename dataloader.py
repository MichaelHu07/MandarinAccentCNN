import os
import librosa
import librosa.display
import numpy  as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data.dataset
import torch.utils.data.dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor
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
    #df = pd.read_csv(spk_filepath, sep="\t")  # convert txt to dataframe
    #spk_id = df["SPEAKER_ID"]
    #spk_region = df["PLACE_OF_BIRTH"]

    training_dataset = AccentDataSet(stft_path, spk_filepath, 8000)

    print("Loading complete")

    return training_dataset







#Deprecated Snippets from array loading version:

    #wave_path = os.path.join(os.getcwd(), r"data\WAVE")
    #os.makedirs(wave_path, exist_ok=True)

    #wave_array = []  # wave_array contains [ y , Filename ]
    #spectro_array = []  # spectro_array contains [ D ]
            # wave_file = os.path.join(wave_path, filename.replace(".wav", ".npy"))
            #if os.path.exists(wave_file):
                #continue
            #else:
                #y, sr = librosa.load(file_path, sr=16000)  # obtaining y (amplitude), and sr (sample rate)
                #np.save(wave_file, y)
                #print(f"Loaded: {filename}, Shape: {y.shape}")

    #for filename in os.listdir(wave_path):
        #path = os.path.join(wave_path, filename)
        #if filename.endswith(".npy"):
            #wave_array.append([np.load(path), filename])
            #print(f"\'{filename}\' Loaded")

    #fig, ax = plt.subplots()
    #img = librosa.display.specshow(librosa.amplitude_to_db(spectro_array[4][0],
    #                                                       ref=np.max),
    #                               y_axis='log', x_axis='time', ax=ax)
    #ax.set_title('Power Spectrogram')
    #fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # categorize dataframe into each category
    #info_array = df.to_numpy()
    #genders = df["GENDER"].to_numpy()
    #ages = df["AGE"].to_numpy()

    # print(spk_id[:5])
    # print(spk_region[:5])

    # matching wav amplitudes to speaker region
    #train_test_array = []  # train_test_array contains [ D , spk_region ]
    #for i, (D, filename) in enumerate(spectro_array):
    #    for idx, sid in enumerate(spk_id):
    #        if sid in filename:
    #            train_test_array.append([D, spk_region[idx]])
    #            break



    # waveform display
    #   plt.figure(figsize=(10, 4))
    # librosa.display.waveshow(wavearray[int(len(wavearray)*random.random())][0], sr = 16000)
    # plt.title("Waveform")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.show()

