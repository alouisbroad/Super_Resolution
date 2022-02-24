"""
Script to train a super resolution machine learning model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import UpSampling2D
# from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Add
from tensorflow.keras.metrics import RootMeanSquaredError

# import torch
# from torch_model import SuperResolution

# import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_preprocessing import generate_LR, normalisation, load_file
from subclassing import SuperResolution


def interpolate_cbh(data, upsample, interp_type="linear"):
    """
    bi-cubic - https://en.wikipedia.org/wiki/Bicubic_interpolation
    bicubic and bilinear interpolation.
    :return:
    """
    ygrid = np.arange(data.shape[0])
    xgrid = np.arange(data.shape[1])
    if interp_type == "linear":
        # Single image.
        f = interpolate.interp2d(xgrid, ygrid, data, kind="linear")
        upscaled = f(np.linspace(0, len(xgrid), len(xgrid) * upsample),
                     np.linspace(0, len(ygrid), len(ygrid) * upsample))
    elif interp_type == "cubic":
        # Single image
        f = interpolate.interp2d(xgrid, ygrid, data, kind="cubic")
        upscaled = f(np.linspace(0, len(xgrid), len(xgrid) * upsample),
                     np.linspace(0, len(ygrid), len(ygrid) * upsample))
    elif interp_type == "nearest":
        # For full batch
        upscaled = UpSampling2D(size=(upsample,upsample), interpolation="nearest")(data)
    elif interp_type == "bilinear":
        # For full batch.
        upscaled = UpSampling2D(size=(upsample, upsample), interpolation="bilinear")(data)
    elif interp_type == "bicubic":
        # https://www.geeksforgeeks.org/python-opencv-bicubic-interpolation-for-resizing-image/
        upscaled = None
    else:
        print("Incorrect interpolation type chosen.")
        upscaled = None

    return upscaled


def cnn_upsampling(upscale_factor=16, channels=1):
    """
    Using upsampling and convolutions. This very slow.
    :return: model
    """
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(filters=64, kernel_size=5, **conv_args)(inputs)
    if upscale_factor%2 == 0:
        for i in range(int(upscale_factor/2)):
            x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
            x = layers.Conv2D(filters=32, kernel_size=3, **conv_args)(x)
    else:
        print("Use even upscale.")
    outputs = layers.Conv2D(filters=1, kernel_size=3, **conv_args)(x)
    return keras.Model(inputs, outputs)


def sr_cnn_sequential(input_shape=(30, 40, 1), output_shape=(480, 640, 1), upscale_factor=16, channels=1):
    """
    Does not run well.
    :return: model
    """
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, **conv_args))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), **conv_args))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), **conv_args))
    model.add(Conv2D(filters=channels*(upscale_factor ** 2), kernel_size=(3, 3), **conv_args))
    # model.add(Flatten())
    # model.add(Dense(input_shape[0]*input_shape[1]*channels*(upscale_factor ** 2), activation="relu"))
    # model.add(Reshape(output_shape))

    return model


def sr_cnn_functional(upscale_factor=16, channels=1):
    """
    Model similar to the one in this link.
    Works well.
     https://keras.io/examples/vision/super_resolution_sub_pixel/
    :return:
    """
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(filters=64, kernel_size=5, **conv_args)(inputs)
    x = layers.Conv2D(filters=64, kernel_size=3, **conv_args)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, **conv_args)(x)
    x = layers.Conv2D(filters=channels * (upscale_factor ** 2), kernel_size=3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)


def sr_conv_cnn():
    """
    Based on Jack's code layout.

    Can't do very large or high res images.
    :return:
    """
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(3, 3), activation='relu', input_shape=(30, 40, 1)))
    model.add(Conv2D(filters=10, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(90 * 120, activation='relu'))
    model.add(Reshape((96, 120, 1)))

    return model


def sr_transpose_2():
    """
    Using Conv2dTranspose in the model. Increases the image by a factor
    of 2 with each transpose layer.
    """
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, 1))
    x = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(1, 1), **conv_args)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), **conv_args)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), **conv_args)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), **conv_args)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(filters=18, kernel_size=(5, 5), strides=(2, 2), **conv_args)(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(1,1), **conv_args)(x)

    return keras.Model(inputs, outputs)


def sr_transpose_4():
    """
    Using Conv2dTranspose in the model. Increases the image by a factor
    of 4 with each transpose layer.
    """
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, 1))
    x = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(1, 1), **conv_args)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(4, 4), **conv_args)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(4, 4), **conv_args)(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(1, 1), **conv_args)(x)

    return keras.Model(inputs, outputs)


def pre_upsampling(upscale_factor=16):
    """
    Using pre-upsampling
    https://beyondminds.ai/blog/an-introduction-to-super-resolution-using-deep-learning/
    """
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, 1))
    x = UpSampling2D(size=(upscale_factor, upscale_factor), interpolation="bilinear")(inputs)
    x = layers.Conv2D(filters=128, kernel_size=3, **conv_args)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, **conv_args)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, **conv_args)(x)
    outputs = Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(1,1), **conv_args)(x)

    return keras.Model(inputs, outputs)


def post_upsampling(upscale_factor=16):
    """

    :return:
    """
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, 1))
    x = layers.Conv2D(filters=128, kernel_size=3, **conv_args)(inputs)
    x = layers.Conv2D(filters=64, kernel_size=3, **conv_args)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, **conv_args)(x)
    x = UpSampling2D(size=(upscale_factor, upscale_factor), interpolation="bilinear")(x)
    outputs = Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(1,1), **conv_args)(x)

    return keras.Model(inputs, outputs)


def prog_upsampling():
    """

    :return:
    """
    # See cnn_upsampling
    pass


def inter_up_down_sampling():
    """

    :return:
    """
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, 1))
    x = layers.Conv2D(filters=128, kernel_size=3, **conv_args)(inputs)
    x = Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(3, 3), **conv_args)(x)

    x = layers.Conv2D(filters=64, kernel_size=2, strides=(2, 2), **conv_args)(x)
    x = Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), **conv_args)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, **conv_args)(x)
    outputs = Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(1,1), **conv_args)(x)

    return keras.Model(inputs, outputs)


if __name__ == '__main__':
    # directory_str = '/data/users/jbowyer/cbh_challenge_data/'
    directory_str = '/data/users/lbroad/Machine_Learning/super_resolution/test_data/'
    hr = load_file(directory_str)
    hr = normalisation(hr, "minmax")
    hr = tf.random.shuffle(hr, seed=None)
    lr = generate_LR(hr, 16)

    train_size = 0.8
    n_samples, rows, cols, n_channels = lr.shape
    X_train, X_test = tf.split(lr, (round(n_samples * train_size), round(n_samples * (1 - train_size))), axis=0)
    y_train, y_test = tf.split(hr, (round(n_samples * train_size), round(n_samples * (1 - train_size))), axis=0)

    op_samples, op_rows, op_cols, op_channels = hr.shape
    model = sr_cnn_sequential(input_shape=(rows, cols, n_channels), output_shape=(op_rows, op_cols, op_channels), upscale_factor=16, channels=1)
    model.summary()

    # Works well, and using the
    model = sr_cnn_functional(upscale_factor=16, channels=1)
    model.summary()

    # Exceptionally slow.
    model = cnn_upsampling()
    model.summary()

    # mr = generate_LR(hr, 5)
    # ... would need to create a mr version.
    model = sr_conv_cnn()
    model.summary()

    model = sr_transpose_2()
    model.summary()

    model = sr_transpose_4()
    model.summary()

    model = pre_upsampling(upscale_factor=16)
    model.summary()

    # from subclassing or torch_model
    model = SuperResolution(upscale_factor=16, channels=1)
    # model.summary()

    opt = Adam(learning_rate=1.0e-4)
    model.compile(optimizer=opt, loss='mse', metrics=[RootMeanSquaredError()])
    model_output = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=20)
    # model_output = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=2, epochs=50, batch_size=10)

    n = 93
    testing = tf.reshape(lr[n], (1, 30, 40, 1))
    prediction = model.predict(testing)

    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(1, 2)
    gs.update(left=0.10, right=0.975, bottom=0.10, top=0.975, wspace=1e-1, hspace=1e-3)

    ax1 = plt.subplot(gs[0])
    cmap1 = ax1.imshow(np.flipud(prediction[0, :, :, 0]))
    cmap1.set_clim([0, 1])
    plt.colorbar(cmap1, orientation='horizontal')

    ax2 = plt.subplot(gs[1])
    cmap2 = ax2.imshow(np.flipud(hr[n, :, :, 0]))
    cmap2.set_clim([0, 1])
    plt.colorbar(cmap2, orientation='horizontal')
