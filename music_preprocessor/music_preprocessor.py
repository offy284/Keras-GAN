import itertools
import shutil
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np
import scipy
from scipy.io.wavfile import write, read
from sklearn.preprocessing import MinMaxScaler
import pickle


def flatten_dir(dir):
    print("Flattening MusicData directory...")
    all_files = []
    dups = 0

    for root, _dirs, files in itertools.islice(os.walk(dir), 1, None):
        try:
            for filename in files:
                all_files.append(os.path.join(root, filename))
        except:
            dups += 1
    for filename in all_files:
        try:
            shutil.move(filename, dir)
        except:
            dups += 1

    print(f"{dups} duplicate files removed")


def generate_big_music():
    print("Generating big_music from MusicData directory...")
    onlyfiles = [f for f in listdir("MusicData/") if isfile(join("MusicData/", f))]

    print("Normalizing big_music...")
    big_music = []

    for i in tqdm(range(len(onlyfiles))):
        file = onlyfiles[i]
        if "-converted" in file:
            x = scipy.io.wavfile.read(f"MusicData/{file}")
            x = x[1]

            x = x.reshape(-1, 1)

            min_max_scaler = MinMaxScaler()
            x = (min_max_scaler.fit_transform(x) - .5) * 2

            samples = np.zeros((int(x.shape[0] / 28 / 28), 28, 28, 1))
            rows = np.zeros((28, 28, 1))
            cols = np.zeros((28, 1))

            for samplei in tqdm(range(samples.shape[0])):
                for yi in range(28):
                    for xi in range(28):
                        cols[xi] = x[xi + yi * 28 + samplei * 28 * 28]
                    rows[yi] = cols
                samples[samplei] = rows

            big_music.append(samples)
            #print(f"Max: {max(x)}, Min: {max(min)}")

    #scipy.io.wavfile.write("big_music.wav", 44100, big_music)

    print("Numpyifying big_music...")
    big_music = np.asarray(big_music)

    big_music = big_music.reshape((big_music.shape[1], 28, 28, 1))

    print(f"big_music is of shape {big_music.shape}")

    filename = "big_music.npy"
    print(f"Saving {filename}...")
    np.save(f"{filename}", big_music )


if __name__ == '__main__':
    print("Music Preprocessor v0.1")
    #flatten_dir()
    generate_big_music()