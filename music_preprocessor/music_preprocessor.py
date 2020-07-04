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


def normalize_concatenate_save():
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

    filename = "big_music.npy"
    print(f"Saving {filename}...")
    np.save(f"{filename}", big_music )


def reshape_save():
    print("Reshaping and saving")

    filename = "big_music.dat"
    print(f"Loading {filename}...")
    big_music = pickle.load(open(f"{filename}", "rb"))


    rows = []
    samples = []

    print("Reshaping big_music...")
    for sample in tqdm(range(int(len(big_music) / 28 / 28))):
        row = []
        for i in range(28):
            for j in range(28):
                row.append([big_music[j + (i * 28) + (sample * 28 * 28)]])
            rows.append(row)
        samples.append(rows)

    filename = "samples.dat"
    print(f"Saving {filename}...")
    pickle.dump(samples, open(f"{filename}", "wb"))


if __name__ == '__main__':
    print("Music Preprocessor v0.1")
    #flatten_dir()
    normalize_concatenate_save()
    #reshape_save()