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
    big_music = list()

    print("Normalizing big_music...")

    for i in tqdm(range(len(onlyfiles))):
        file = onlyfiles[i]
        if "-converted" in file:
            x = scipy.io.wavfile.read(f"MusicData/{file}")
            x = x[1]

            x = x.reshape(-1, 1)

            min_max_scaler = MinMaxScaler()
            x = (min_max_scaler.fit_transform(x) - .5) * 2

            for sample in x:
                sample = sample[0]

            for sample in x:
                for channel in sample:
                    big_music.append(float(channel))

            #print(f"Max: {max(x)}, Min: {max(min)}")

    #scipy.io.wavfile.write("big_music.wav", 44100, big_music)

    print("Saving big_music.str...")

    with open("big_music.npy", 'w') as file:
        file.write(str(big_music))


def reshape_save():
    print("Reshaping and saving")

    print("Loading big_music.str...")
    big_music = []
    with open("big_music.str", "r") as file:
        big_music = file.readlines()

    print("Listifying big_music...")
    big_music = list(big_music)

    print("Building colums...")
    cols = []
    start = 0
    end = 28

    for sample in tqdm(range(len(big_music))):
        col = big_music[start:end]
        cols.append(col)

        start += 28
        end += 28

    print("Building rows...")
    rows = []
    start = 0
    end = 28

    for col in tqdm(range(len(cols))):
        row = cols[start:end]

        start += 28
        end += 28

    print("Building samples...")
    samples = []
    start = 0
    end = 28

    for row in tqdm(range(len(rows))):
        samples.append(rows[start:end])

        start += 28
        end += 28

    print("Saving big_music_imgs.list...")
    with open("big_music_imgs.list", "w") as file:
        file.write("big_music_imgs.npy", samples)



if __name__ == '__main__':
    print("Music Preprocessor v0.1")

    #flatten_dir()
    normalize_concatenate_save()
    reshape_save()