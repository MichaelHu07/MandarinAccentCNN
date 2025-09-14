import os
import librosa
import librosa.display
import numpy  as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    # load data from folder into numpy array
    folder_path = os.path.join(os.getcwd(), r"data\WAV")
    stft_path = os.path.join(os.getcwd(), r"data\STFT")
    #wave_path = os.path.join(os.getcwd(), r"data\WAVE")
    os.makedirs(stft_path, exist_ok=True)
    #os.makedirs(wave_path, exist_ok=True)

    #wave_array = []  # wave_array contains [ y , Filename ]
    #spectro_array = []  # spectro_array contains [ D ]
    for filename in os.listdir(folder_path):  # looping through every file in WAV folder
        if filename.endswith(".wav"):
            #wave_file = os.path.join(wave_path, filename.replace(".wav", ".npy"))
            stft_file = os.path.join(stft_path, filename.replace(".wav", "_1.npy"))
            file_path = os.path.join(folder_path, filename)

            #if os.path.exists(wave_file):
                #continue
            #else:
                #y, sr = librosa.load(file_path, sr=16000)  # obtaining y (amplitude), and sr (sample rate)
                #np.save(wave_file, y)
                #print(f"Loaded: {filename}, Shape: {y.shape}")

            if os.path.exists(stft_file):
                continue
            else:
                y, sr = librosa.load(file_path, sr=16000)  # obtaining y (amplitude), and sr (sample rate)
                D = np.abs(librosa.stft(y, n_fft=400, hop_length=160, win_length=400))
                np.save(stft_file, D)
                print(f"Processed: {filename}")

    #for filename in os.listdir(wave_path):
        #path = os.path.join(wave_path, filename)
        #if filename.endswith(".npy"):
            #wave_array.append([np.load(path), filename])
            #print(f"\'{filename}\' Loaded")
    for filename in os.listdir(stft_path):
        path = os.path.join(stft_path, filename)
        if filename.endswith(".npy"):
                #Will use pytorch Dataset instead of array
            #spectro_array.append([np.load(path), filename])
            print(f"\'{filename}\' Loaded")

    #fig, ax = plt.subplots()
    #img = librosa.display.specshow(librosa.amplitude_to_db(spectro_array[4][0],
    #                                                       ref=np.max),
    #                               y_axis='log', x_axis='time', ax=ax)
    #ax.set_title('Power Spectrogram')
    #fig.colorbar(img, ax=ax, format="%+2.0f dB")

    spk_filepath = os.path.join(os.getcwd(), r"data\SPKINFO.txt")
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
    train_test_array = []  # train_test_array contains [ D , spk_region ]
    #for i, (D, filename) in enumerate(spectro_array):
    #    for idx, sid in enumerate(spk_id):
    #        if sid in filename:
    #            train_test_array.append([D, spk_region[idx]])
    #            break


    print("Loading complete")
    return train_test_array
    # waveform display
    #   plt.figure(figsize=(10, 4))
    # librosa.display.waveshow(wavearray[int(len(wavearray)*random.random())][0], sr = 16000)
    # plt.title("Waveform")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.show()

