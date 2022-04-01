"""
Script to run machine learning scripts.
"""
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

from functions import SRCallback
from super_resolution_models import super_resolution_upscaling
from data_preprocessing import generate_LR, normalisation, load_file


def main():
    # - Inputs -
    parser = argparse.ArgumentParser(description='Super resolution machine learning script.')
    parser.add_argument('-epochs', type=int, required=True, help='Number of epochs to train for.')
    parser.add_argument('-batch', type=int, required=True, help='Size of batch during training.')

    args = parser.parse_args()

    # - Data Prep -
    upscale_factor = 16

    directory_str = '/data/users/jbowyer/cbh_challenge_data/'
    # directory_str = '/data/nwp1/frme/ML_minichallenge/train/'
    # directory_str = '/data/users/lbroad/Machine_Learning/super_resolution/test_data/'

    variable = "temperature"
    hr = load_file(directory_str, variable)
    hr = normalisation(hr, "minmax")

    callback_data = tf.reshape(hr[0], (1, *hr[0].shape))

    hr = tf.random.shuffle(hr, seed=None)
    lr = generate_LR(hr, upscale_factor)

    train_size = 0.8
    n_samples, rows, cols, n_channels = lr.shape
    X_train, X_test = tf.split(lr, (round(n_samples * train_size), round(n_samples * (1 - train_size))), axis=0)
    y_train, y_test = tf.split(hr, (round(n_samples * train_size), round(n_samples * (1 - train_size))), axis=0)

    # - Model -
    model = super_resolution_upscaling(upscale_factor=upscale_factor, channels=1)
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
    with open("model_{}.json".format(args.epochs), "w") as json_file:
        json_file.write(model_json)
        json_file.close()

    model_output = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                             verbose=2, epochs=args.epochs, batch_size=args.batch,
                             callbacks=callbacks)
    # model_output = model.fit(X_train, y_train, validation_data=(X_test, y_test),
    #                          epochs=args.epochs, batch_size=args.batch,
    #                          callbacks=callbacks)

    model.save_weights("model_saved_weights_{}.h5".format(args.epochs))


if __name__ == '__main__':
    main()

