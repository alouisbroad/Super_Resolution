"""
Script to run machine learning scripts using generators created previously.
"""
import glob
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

from functions import SRCallback
from data_generators import tf_data_generator
from super_resolution_models import super_resolution_upscaling


def train(epochs, batch):
    # - Data Prep -
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
    model = super_resolution_upscaling(upscale_factor=upscale_factor, channels=1)
    model.summary()

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
    model.compile(optimizer=opt, loss='mse', metrics=[RootMeanSquaredError()])

    model_json = model.to_json()
    with open("model_{}.json".format(epochs), "w") as json_file:
        json_file.write(model_json)
        json_file.close()

    model_output = model.fit(x=dataset.prefetch(tf.data.AUTOTUNE),
                             validation_data=validationset,
                             epochs=epochs,
                             callbacks=callbacks,
                             steps_per_epoch=len(file_list)*round(70 / batch),
                             validation_steps=len(file_list[300:])*round(70 / batch),
                             verbose=2
                             )

    # - Results -
    model.save_weights("model_saved_weights_{}.h5".format(epochs))

    fileout = "/data/users/lbroad/Machine_Learning/super_resolved_outputs/{}_e{}_b{}.txt"
    np.savetxt(fileout.format("rmse", epochs, batch), np.ones((1, 1)) * model_output.history["root_mean_squared_error"], fmt='%10.7f')
    np.savetxt(fileout.format("lrmse", epochs, batch), np.ones((1, 1)) * model_output.history["val_root_mean_squared_error"], fmt='%10.7f')


if __name__ == '__main__':
    # - Inputs -
    parser = argparse.ArgumentParser(description='Super resolution machine learning script.')
    parser.add_argument('-epochs', type=int, required=True, help='Number of epochs to train for.')
    parser.add_argument('-batch', type=int, required=True, help='Size of batch during training.')

    args = parser.parse_args()

    train(args.epochs, args.batch)
