from __future__ import print_function, division
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import scipy
from scipy.io.wavfile import write, read
import numpy as np
import pickle
from tqdm import tqdm


RESOLUTION_SCALE = 10
RUN_LEN = 100


class GAN():
    def __init__(self, resolution_scale=20):
        self.img_shape = (None, RUN_LEN, 129)
        self.latent_dim = (RUN_LEN, 2)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=self.latent_dim)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(LSTM(256, input_shape=self.latent_dim, return_sequences=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LSTM(512, return_sequences=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LSTM(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape[1:]), activation='tanh'))
        model.add(Reshape(self.img_shape[1:]))

        model.summary()

        noise = Input(shape=self.latent_dim,)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(LSTM(1024, input_shape=self.img_shape[1:], return_sequences=True))
        model.add(LSTM(512, return_sequences=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(LSTM(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape[1:])
        validity = model(img)

        return Model(img, validity)

    def load_data(self):
        filename = f"spectrogram.npy"
        print(f"Loading {filename}...")
        spectrogram = np.load(filename)

        filename = f"freqs.npy"
        print(f"Loading {filename}...")
        freqs = np.load(filename)

        filename = f"times.npy"
        print(f"Loading {filename}...")
        times = np.load(filename)

        print(f"spectrogram is of shape {spectrogram.shape}")
        print(f"freqs is of shape {freqs.shape}")
        print(f"times is of shape {times.shape}")

        print("Chunking spectrogram...")
        runs = np.empty((int(spectrogram.shape[0] / RUN_LEN) + 1, RUN_LEN, spectrogram.shape[1]))
        time_runs = []

        featureseti = 0

        for runi in tqdm(range(len(runs))):
            run = runs[runi]
            time_run = []
            for samplei in range(len(run)):
                run[samplei] = spectrogram[featureseti]
                if runi * len(run) + samplei < len(times):
                    time_run.append(times[runi * len(run) + samplei])
            while len(time_run) < runs.shape[1]:
                    time_run.append(0)
            time_runs.append(np.array(time_run))

        runs = np.array(runs)
        time_runs = np.array(time_runs)

        print(f"runs is of shape {runs.shape}")
        print(f"freqs is of shape {freqs.shape}")
        print(f"time_runs is of shape {time_runs.shape}")

        return runs, freqs, time_runs

    def train(self, epochs, batch_size=128, sample_interval=50):
        (X_train, freqs, times) = self.load_data()

        #X_train = X_train / np.linalg.norm(X_train)
        #X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)

            t = times[idx]
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, ((batch_size,) + self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            #print(imgs)
            #print(np.asarray(gen_imgs))
            #print(valid)
            #print(fake)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, ((batch_size,) + self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, freqs, t[0])

    def sample_images(self, epoch, f, t):
        r, c = 5, 5
        noise = np.random.normal(0, 1, ((r * c,) + self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :], cmap='gray')
                axs[i, j].axis('off')

                print(f"t shape is of shape {t.shape}")
                np.save(f"music/song_Sxx_{i * c + j}-epoch_{epoch}", gen_imgs[cnt, :, :])
                np.save(f"music/song_f_{i * c + j}-epoch_{epoch}", f)
                np.save(f"music/song_t_{i * c + j}-epoch_{epoch}", t)

                cnt += 1
        fig.savefig("music_images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    print("music_gan.py v0.1")


    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


    gan = GAN(resolution_scale=RESOLUTION_SCALE)
    gan.train(epochs=50000, sample_interval=1)
