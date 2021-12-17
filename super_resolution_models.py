"""
Script to train a super resolution machine learning model.
"""
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU
import os
import numpy as np
from scipy import interpolate
from data_preprocessing import load_file_cnn_sr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
    file = os.listdir(directory_str)[0]
    filename = directory_str + os.fsdecode(file)
    result = load_file_cnn_sr(filename)
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

        return keras.Model#(inputs, outputs)


    model = get_model(upscale_factor=3, channels=1)
    model.summary()
