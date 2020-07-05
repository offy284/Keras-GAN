import numpy as np
import simpleaudio as sa


def play_epoch(epoch=0):
    r, c = 5, 5
    for i in range(r):
        for j in range(c):
            test_song = np.load(f"music/song_{i * c + j}-epoch_{epoch}.npy")
            sa.play_buffer(audio_data=test_song, sample_rate=44100, num_channels=1, bytes_per_sample=2).wait_done()


if __name__ == "__main__":
    play_epoch(1043)