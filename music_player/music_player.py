import numpy as np
import simpleaudio as sa


r, c = 5, 5
for i in range(r):
    for j in range(c):
        test_song = np.load(f"song_r{i}-c{j}.npy")
        sa.play_buffer(audio_data=test_song, sample_rate=44100, num_channels=1, bytes_per_sample=2).wait_done()