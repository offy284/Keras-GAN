import os
import itertools
import shutil

import scipy as scipy


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


import numpy as np
import scipy
from scipy.io.wavfile import write, read
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("MusicData/") if isfile(join("MusicData/", f))]


big_music = np.zeros((1,2))

i = 0
for file in onlyfiles:
    i += 1
    if i % 10 == 0:
        print(f"{i/len(onlyfiles) * 100}% complete")
    if "-converted" in file:
        x = scipy.io.wavfile.read(f"MusicData/{file}")
        x = x[1]
        big_music = np.concatenate([big_music, x])


test_out = scipy.io.wavfile.write("big_music.wav", 44100, big_music)