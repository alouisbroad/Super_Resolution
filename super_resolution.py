"""
Script for running random search on Jack's super resolution code found
here: /home/h05/jbowyer/PycharmProjects/MachineLearning/super_res_cnn_model.py
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Flatten, Reshape, LeakyReLU
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
import warnings
from load_files import load_file_cnn_sr
import sys
sys.path.append('/home/h05/jbowyer/PycharmProjects/tools')
from functions import generate_lower_resolution_data


warnings.filterwarnings('ignore', '.*HybridHeightFactory*.')
# Define the deep (convolutional) neural network
# channels_last corresponds to inputs with shape (batch_size, height, width, channels)
# channels_first corresponds to inputs with shape (batch_size, channels, height, width)

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(3, 3), activation='relu', input_shape=(12, 16, 1)))
model.add(Conv2D(filters=10, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(24*32, activation='relu'))
model.add(Reshape((24, 32, 1)))
# model = Sequential()
# model.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(12, 16, 1)))
# model.add(LeakyReLU(alpha=0.2))
# model.add(Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(2, 2), padding='same'))
# model.add(LeakyReLU(alpha=0.2))
model.summary()

opt = Adam(learning_rate=1.0e-4)
model.compile(optimizer=opt, loss='mse', metrics=[RootMeanSquaredError()])

# Write things out to file (for reading back in later to make predictions).
model_json = model.to_json()
with open("models_and_weights/cnn_sr_model.json", "w") as json_file:
    json_file.write(model_json)
    json_file.close()

# Define directory where training data is stored.
directory_str = '/data/users/jbowyer/cbh_challenge_data/'
directory = os.fsencode(directory_str)

# Set up arrays to store the rmse as one loops through multiple sets of multiple epochs.
keep_rmse_train = np.empty([1, 1]) + np.nan
keep_rmse_val = np.empty([1, 1]) + np.nan
s = np.random.permutation(np.arange(0, 210, 1))

n_super_loop = 5
# You might want to try changing the number of super_loops.
for super_loop in np.arange(0, n_super_loop):
    file_no = 0
    for file in os.listdir(directory):
        file_no = file_no + 1
        filename = directory_str + os.fsdecode(file)
        print('Super_Loop = ', super_loop, 'Loading file_no', file_no, filename)
        print('Loading file_no', file_no, filename)
        result = load_file_cnn_sr(filename)
        data_high_res = result['data_hr']
        data_med_res = result['data_mr']
        data_low_res = result['data_lr']

        # stash_code = 'm01s16i004'
        # data_high_res, data_low_res = generate_lower_resolution_data(filename, stash_code, 40)
        # # _,             data_med_res = generate_lower_resolution_data(filename, stash_code, 5)

        data_low_res_shuffled = data_low_res[s]
        data_med_res_shuffled = data_med_res[s]
        data_high_res_shuffled = data_low_res[s]
        # Must shuffle array and *then* take X and y from it
        # (rather than take X and y and then shuffle) to make sure things line up
        # X is the long vector of inputs (temp, qv, pres)
        # y is the one-hot encoded vector of where cloud base is
        X = data_low_res_shuffled.reshape(210, 12, 16, 1)
        # print(X.shape)
        y = data_med_res_shuffled.reshape(210, 24, 32, 1)
        # y = data_med_res_shuffled.reshape(210, 1200)
        # y = data_high_res_shuffled.reshape(210, n_nodes)
        # n_samples, rows, cols = X.shape
        n_samples, rows, cols, n_channels = X.shape
        # Use 80% of the data for training and the rest for validation
        n_train = int(0.8*n_samples)
        trainX = X[:n_train]
        testX = X[n_train:]
        trainy = y[:n_train]
        testy = y[n_train:]

        history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1, batch_size=10)
        keep_rmse_train = np.append(keep_rmse_train, history.history['root_mean_squared_error'])
        keep_rmse_val = np.append(keep_rmse_val, history.history['val_root_mean_squared_error'])
        # Write out some metrics to track progress. You need to read these back in later and plot them
        file_out = 'timeseries/cnn_sr_timeseries_rmse_train.txt'
        np.savetxt(file_out, np.ones((1, 1))*keep_rmse_train, fmt='%10.7f')
        file_out = 'timeseries/cnn_sr_timeseries_rmse_val.txt'
        np.savetxt(file_out, np.ones((1, 1))*keep_rmse_val, fmt='%10.7f')
        # Write out the weights once per super_loop
        model.save_weights("models_and_weights/cnn_sr_saved_weights.h5")


print('All training completed successfully!')



