import os
import itertools
import shutil
import numpy as np
import scipy
from scipy.io.wavfile import write, read
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


def move(destination):
    all_files = []
    for root, _dirs, files in itertools.islice(os.walk(destination), 1, None):
        try:
            for filename in files:
                all_files.append(os.path.join(root, filename))
        except:
            print("DUP")
    for filename in all_files:
        try:
            shutil.move(filename, destination)
        except:
            print("DUP")


def normalize_concatenate_save():
    onlyfiles = [f for f in listdir("MusicData/") if isfile(join("MusicData/", f))]

    big_music = np.zeros((1, 1))

    for i in tqdm(range(len(onlyfiles))):
        file = onlyfiles[i]
        if "-converted" in file:
            x = scipy.io.wavfile.read(f"MusicData/{file}")
            x = x[1]
            x = x.reshape((-1, 1))

            min_max_scaler = MinMaxScaler()
            x = (min_max_scaler.fit_transform(x) - .5) * 2

            print(f"Max: {x.max()}, Min: {x.min()}")

            big_music = np.concatenate([big_music, x])


    #scipy.io.wavfile.write("big_music.wav", 44100, big_music)

    with open("big_music.npy", 'wb') as file:
        np.save(file, big_music)


normalize_concatenate_save()