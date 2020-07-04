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

RESOLUTION_SCALE = 5


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


def generate_big_music(resolution_scale=RESOLUTION_SCALE):
    print("Generating big_music from MusicData directory...")
    onlyfiles = [f for f in listdir("MusicData/") if isfile(join("MusicData/", f))]

    print("Normalizing big_music...")
    square_size = 28 * resolution_scale
    big_music = np.empty((len(onlyfiles), square_size, square_size, 1))

    for i in tqdm(range(len(onlyfiles))):
        file = onlyfiles[i]
        if "-converted" in file:
            x = scipy.io.wavfile.read(f"MusicData/{file}")
            x = x[1]

            x = x.reshape(-1, 1)

            min_max_scaler = MinMaxScaler()
            x = (min_max_scaler.fit_transform(x) - .5) * 2

            samples = list(np.empty((int(x.shape[0] / square_size / square_size), square_size, square_size, 1)))
            rows = np.zeros((square_size, square_size, 1))
            cols = np.zeros((square_size, 1))

            for samplei in tqdm(range(len(samples))):
                for yi in range(square_size):
                    for xi in range(square_size):
                        cols[xi] = x[xi + yi * square_size + samplei * square_size * square_size]
                    rows[yi] = cols
                samples[samplei] = rows

            print("Numpyifying samples...")
            samples = np.asarray(samples)

            big_music = np.concatenate([big_music, samples])
            #print(f"Max: {max(x)}, Min: {max(min)}")

    #scipy.io.wavfile.write("big_music.wav", 44100, big_music)

    print(big_music.shape)
    big_music = big_music.reshape((big_music.shape[0], square_size, square_size, 1))

    print(f"big_music is of shape {big_music.shape}")

    filename = "big_music.npy"
    print(f"Saving {filename}...")
    np.save(f"{filename}", big_music )


if __name__ == '__main__':
    print("Music Preprocessor v0.1")
    #flatten_dir()
    generate_big_music()