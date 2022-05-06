"""
Creating a data generator for super resolution.
"""
import os
import iris
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

from data_preprocessing import generate_LR, normalisation, make_stash_string
from functions import SRCallback
from super_resolution_models import sr_cnn_functional


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras - https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    TODO: finish
    """

    def __init__(self, file_list, batch_size=32, dim=(70, 480, 640), n_channels=1, shuffle=True):
        self.file_list = file_list
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        stash_codes = {"specific humidity": (0, 10),
                       "pressure": (0, 408),
                       "temperature": (16, 4)}
        self.tmp = make_stash_string(*stash_codes["temperature"])['stashstr_iris']

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.dim[0] // self.batch_size  # For this you need to know the amount of levels in each file.

    def __getitem__(self, i):
        for file in self.file_list:
            data = self.__loadFile__(file)
            batch = self.samples[i * self.batch_size:(i + 1) * self.batch_size]

    #     return generate_LR(hr), hr
    #
    # def __getBatch__(self, index):
    #     for file in self.file_list:
    #         data = self.__loadFile__(file)
    #         for i in index:
    #             X.append(librosa.feature.mfcc(wav, self.sr).T)
    #             y.append(self.label[i])
    #         return generate_LR(hr), hr

    def __loadFile__(self, file):
        data = iris.load_cube(os.fsdecode(file), iris.AttributeConstraint(STASH=self.tmp)).data
        data = tf.random.shuffle(tf.reshape(tf.convert_to_tensor(data), (*data.shape, 1)))
        data = normalisation(data, "zscore")
        return data

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)


def tf_data_generator_fill_batch(file_list, batch_size=10):
    """

    :param file_list:
    :param batch_size:

    TODO: Create even batch sizes when each files dataset isn't divisible by the batch size.
    """
    i = 0
    stash_codes = {"specific humidity": (0, 10),
                   "pressure": (0, 408),
                   "temperature": (16, 4)}
    tmp = make_stash_string(*stash_codes["temperature"])['stashstr_iris']
    while True:  # This loop makes the generator an infinite loop
        if i >= len(file_list):
            i = 0
            np.random.shuffle(file_list)
        else:
            for file in file_list:
                temp = iris.load_cube(os.fsdecode(file), iris.AttributeConstraint(STASH=tmp)).data
                lvl, lon, lat = temp.shape
                if lvl % batch_size == 0:
                    for j in range(lvl // batch_size):
                        data = temp[j * batch_size:(j + 1) * batch_size]
                        data = tf.random.shuffle(tf.reshape(tf.convert_to_tensor(data), (batch_size, lon, lat, 1)))
                        data = normalisation(data, "minmax")
                        yield generate_LR(data), data
                else:  # Store unfinished batches!!!
                    for j in range(lvl // batch_size):
                        data = temp[j * batch_size:(j + 1) * batch_size]
                        data = tf.random.shuffle(tf.reshape(tf.convert_to_tensor(data), (batch_size, lon, lat, 1)))
                        data = normalisation(data, "minmax")
                        yield generate_LR(data), data
                i += 1


def tf_data_generator(file_list, batch_size=10, variable="temperature"):
    """
    Simple generator that reads in the temperature data on levels.
    Each cube has 70 levels, so non-divisible batch sizes with give
    varying batches.
    :param file_list: list of file paths.
    :param batch_size: int batch size.
    :param variable: string of either "specific humidity", "pressure" or "temperature",
    depending of what you are training.
    """
    i = 0
    stash_codes = {"specific humidity": (0, 10),
                   "pressure": (0, 408),
                   "temperature": (16, 4)}
    tmp = make_stash_string(*stash_codes[variable])['stashstr_iris']
    while True:  # This loop makes the generator an infinite loop
        if i == 0:
            np.random.shuffle(file_list)
            print(file_list[i])
        file = file_list[i]

        i = (i + 1) % len(file_list)
        data = iris.load_cube(os.fsdecode(file), iris.AttributeConstraint(STASH=tmp)).data
        data = tf.random.shuffle(tf.reshape(tf.convert_to_tensor(data), (*data.shape, 1)))
        data = normalisation(data, "minmax")
        # data = normalisation(data, "zscore")
        for local_index in range(0, data.shape[0], batch_size):
            hr = data[local_index:(local_index + batch_size)]
            yield generate_LR(hr), hr


if __name__ == '__main__':
    upscale_factor = 16
    epochs = 201
    batch_size = 35

    # directory_str = '/data/users/jbowyer/cbh_challenge_data/'
    # directory_str = '/data/nwp1/frme/ML_minichallenge/train/'
    # directory_str = '/data/users/lbroad/Machine_Learning/super_resolution/test_data/'
    file_list = glob.glob('/data/users/jbowyer/cbh_challenge_data/*')
    dataset = tf.data.Dataset.from_generator(tf_data_generator, args=(file_list[:300], batch_size),
                                             output_types=(tf.dtypes.float32, tf.dtypes.float32))

    validationset = tf.data.Dataset.from_generator(tf_data_generator, args=(file_list[300:], batch_size),
                                                   output_types=(tf.dtypes.float32, tf.dtypes.float32))

    for x, callback_data in dataset.take(1):
        print(x.shape)
        print(callback_data.shape)
        callback_data = tf.reshape(callback_data[0], (1, *callback_data[0].shape))

    model = sr_cnn_functional(upscale_factor=upscale_factor, channels=1)
    model.summary()

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)

    checkpoint_filepath = "./checkpoints"

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    callbacks = [SRCallback(callback_data=callback_data), early_stopping_callback, model_checkpoint_callback]

    opt = Adam(learning_rate=1.0e-4)
    model.compile(optimizer=opt, loss='mse', metrics=[RootMeanSquaredError()])

    model_json = model.to_json()
    with open("model_{}.json".format(epochs), "w") as json_file:
        json_file.write(model_json)
        json_file.close()

    model_output = model.fit(x=dataset,
                             validation_data=validationset,
                             epochs=epochs,
                             callbacks=callbacks, 
                             steps_per_epoch=300*round(70 / batch_size),
                             validation_steps=len(file_list[300:])*round(70 / batch_size),
                             verbose=2)

    model.save_weights("model_saved_weights_{}.h5".format(epochs))
