"""
--
"""
# import keras
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
import math
from data_preprocessing import load_file_cnn_sr

if __name__ == '__main__':
    directory_str = '/data/users/jbowyer/cbh_challenge_data/'
    file = os.listdir(directory_str)[0]
    filename = directory_str + os.fsdecode(file)
    result = load_file_cnn_sr(filename)
    data_high_res = result['data_hr']
    data_med_res = result['data_mr']
    data_low_res = result['data_lr']

    _shape = data_high_res.shape
    upscale_factor = 20
    input_size = _shape // upscale_factor

