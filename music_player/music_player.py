import numpy as np
import scipy
import simpleaudio as sa
import scipy.signal as signal
import numpy as np
from inverse_spectrogram import inverse_spectrogram


def play_epoch(epoch=0):
    r, c = 5, 5
    for i in range(r):
        for j in range(c):
            Sxx = np.load(f"music/song_Sxx_{i * c + j}-epoch_{epoch}.npy")
            f = np.load(f"music/song_f_{i * c + j}-epoch_{epoch}.npy", allow_pickle=True)
            t = np.load(f"music/song_t_{i * c + j}-epoch_{epoch}.npy")

            t = t - t[0]

            print("Inverting spectrogram...")
            music = inverse_spectrogram(f=f, t=t, Sxx=Sxx, fs=44100)
            print(f"Playing sample {i * c + j}")
            sa.play_buffer(audio_data=music, sample_rate=44100, num_channels=1, bytes_per_sample=2).wait_done()


if __name__ == "__main__":
    print("-Music Player v0.1-")
    play_epoch(3)