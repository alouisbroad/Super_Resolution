"""
Script to train a super resolution machine learning model.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU
from keras.metrics import RootMeanSquaredError
from keras.utils.vis_utils import plot_model

import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data_preprocessing import generate_LR, normalisation, load_file


def interpolate_cbh(data, upsample, interp_type="linear"):
    """
    bi-cubic - https://en.wikipedia.org/wiki/Bicubic_interpolation
    bicubic and bilinear interpolation.
    :return:
    """
    ygrid = np.arange(data.shape[0])
    xgrid = np.arange(data.shape[1])
    if interp_type == "linear":
        f = interpolate.interp2d(xgrid, ygrid, data, kind="linear")
    elif interp_type == "cubic":
        f = interpolate.interp2d(xgrid, ygrid, data, kind="cubic")
    elif interp_type == "bicubic":
        # https://www.geeksforgeeks.org/python-opencv-bicubic-interpolation-for-resizing-image/
        pass
    upscaled = f(np.linspace(0, len(xgrid), len(xgrid) * upsample),
                 np.linspace(0, len(ygrid), len(ygrid) * upsample))

    return upscaled


def sr_cnn():
    """
    https://keras.io/examples/vision/super_resolution_sub_pixel/
    :return: model
    """
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(3, 3), input_shape=(12, 16, 1), **conv_args))
    model.add(Conv2D(filters=10, kernel_size=(3, 3), **conv_args))

    return model


def sr_trans_cnn():
    """

    :return:
    """
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    model = Sequential()
    model.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(1, 1), input_shape=(12, 16, 1), **conv_args))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(2, 2), **conv_args))
    model.add(LeakyReLU(alpha=0.2))

    return model


def pre_upsampling():
    """
    Using pre-upsampling
    https://beyondminds.ai/blog/an-introduction-to-super-resolution-using-deep-learning/
    """
    pass


def post_upsampling():
    """

    :return:
    """
    pass


def prog_upsampling():
    """

    :return:
    """
    pass


def inter_up_down_sampling():
    """

    :return:
    """
    pass


if __name__ == '__main__':
    directory_str = '/data/users/jbowyer/cbh_challenge_data/'
    # directory_str = '/data/users/lbroad/Machine_Learning/super_resolution/test_data/'
    file = os.listdir(directory_str)[0]
    filename = directory_str + os.fsdecode(file)
    result = load_file(filename)
    data_high_res = result['data_hr']
    data_med_res = result['data_mr']
    data_low_res = result['data_lr']

    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(3, 3), input_shape=(12, 16, 1), **conv_args))
    model.summary()
    plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)

    opt = Adam(learning_rate=1.0e-4)
    model.compile(optimizer=opt, loss='mse', metrics=[RootMeanSquaredError()])

    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(3, 2)
    gs.update(left=0.10, right=0.975, bottom=0.10, top=0.975, wspace=1e-1, hspace=1e-3)

    ax1 = plt.subplot(gs[0])
    cmap1 = ax1.imshow(np.flipud())
    cmap1.set_clim([0, 1])
    plt.colorbar(cmap1, orientation='horizontal')

    ax2 = plt.subplot(gs[2])
    cmap2 = ax2.imshow(np.flipud())
    cmap2.set_clim([0, 1])
    plt.colorbar(cmap2, orientation='horizontal')


    def get_model(upscale_factor=3, channels=1):
        conv_args = {
            "activation": "relu",
            "kernel_initializer": "Orthogonal",
            "padding": "same",
        }
        inputs = keras.Input(shape=(None, None, channels))
        x = Conv2D(64, 5, **conv_args)(inputs)
        x = Conv2D(64, 3, **conv_args)(x)
        x = Conv2D(32, 3, **conv_args)(x)
        x = Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
        outputs = tf.nn.depth_to_space(x, upscale_factor)

        return keras.Model(inputs, outputs)


    model = get_model(upscale_factor=3, channels=1)
    model.summary()
