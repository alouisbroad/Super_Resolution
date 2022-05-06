"""
https://www.tensorflow.org/guide/data_performance
https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
"""
import os
import iris
import time
import glob
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

from functions import SRCallback
from data_generators import tf_data_generator
from super_resolution_models import super_resolution_upscaling
from data_preprocessing import generate_LR, normalisation, make_stash_string


if __name__ == '__main__':
    # - Data Prep -
    epochs = 2
    batch = 10
    upscale_factor = 16
    file_list = glob.glob('/data/users/jbowyer/cbh_challenge_data/*')
    dataset = tf.data.Dataset.from_generator(tf_data_generator, args=(file_list[:300], batch),
                                             output_types=(tf.dtypes.float32, tf.dtypes.float32))

    validationset = tf.data.Dataset.from_generator(tf_data_generator, args=(file_list[300:], batch),
                                                   output_types=(tf.dtypes.float32, tf.dtypes.float32))
    for x, callback_data in dataset.take(1):
        print(x.shape)
        print(callback_data.shape)
        callback_data = tf.reshape(callback_data[0], (1, *callback_data[0].shape))

    # - Model -
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)
    checkpoint_filepath = "./checkpoints/checkpoints"  # this is what needs to be passed to model.load_weights
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    callbacks = [SRCallback(callback_data=callback_data), early_stopping_callback, model_checkpoint_callback]
    opt = Adam(learning_rate=1.0e-4)

    # - Baseline -
    model = super_resolution_upscaling(upscale_factor=upscale_factor, channels=1)
    model.compile(optimizer=opt, loss='mse', metrics=[RootMeanSquaredError()])
    start = time.time()
    model_output = model.fit(x=dataset,
                             validation_data=validationset,
                             epochs=epochs,
                             callbacks=callbacks,
                             steps_per_epoch=20,
                             validation_steps=20)
    print("Baseline: ", time.time() - start)

    # - Autotune -
    model = super_resolution_upscaling(upscale_factor=upscale_factor, channels=1)
    model.compile(optimizer=opt, loss='mse', metrics=[RootMeanSquaredError()])
    start = time.time()
    model_output = model.fit(x=dataset.prefetch(tf.data.AUTOTUNE),
                             validation_data=validationset,
                             epochs=epochs,
                             callbacks=callbacks,
                             steps_per_epoch=20,
                             validation_steps=20)
    print("Autotune: ", time.time() - start)

    # - Parallelizing data transformation -
    def transform_generator(file_list, batch_size=10, variable="temperature"):
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

            for local_index in range(0, data.shape[0], batch_size):
                hr = data[local_index:(local_index + batch_size)]
                yield hr


    def transform_function(data):
        hr = normalisation(data, "minmax")
        return generate_LR(hr), hr

    dataset = tf.data.Dataset.from_generator(transform_generator, args=(file_list[:300], batch),
                                             output_types=(tf.dtypes.float32))

    validationset = tf.data.Dataset.from_generator(transform_generator, args=(file_list[300:], batch),
                                                   output_types=(tf.dtypes.float32))

    model = super_resolution_upscaling(upscale_factor=upscale_factor, channels=1)
    model.compile(optimizer=opt, loss='mse', metrics=[RootMeanSquaredError()])
    start = time.time()
    model_output = model.fit(x=dataset.map(transform_function),
                             validation_data=validationset.map(transform_function),
                             epochs=epochs,
                             callbacks=callbacks,
                             steps_per_epoch=20,
                             validation_steps=20)
    print("Baseline: ", time.time() - start)
