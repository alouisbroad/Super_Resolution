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
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
import warnings
from load_files import load_file_cnn_sr
from scipy.stats import loguniform
import datetime
import sys


def build_model(learn_rate):
    """
    Model for super resolution machine learning.
    Add variables for random/grid search.

    :return: ml model.
    """
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(3, 3), activation='relu', input_shape=(12, 16, 1)))
    model.add(Conv2D(filters=10, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(24 * 32, activation='relu'))
    model.add(Reshape((24, 32, 1)))
    # model = Sequential()
    # model.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(12, 16, 1)))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    # model.add(LeakyReLU(alpha=0.2))
    model.summary()
    opt = Adam(learning_rate=learn_rate)
    model.compile(optimizer=opt, loss='mse', metrics=[RootMeanSquaredError()])

    return model

def main():
    """
    Function runs a super resolution machine learning script using random search.
    :return: None
    """
    warnings.filterwarnings('ignore', '.*HybridHeightFactory*.')
    today = datetime.datetime.today().strftime("%Y%m%d")

    # Define the deep (convolutional) neural network
    # channels_last corresponds to inputs with shape (batch_size, height, width, channels)
    # channels_first corresponds to inputs with shape (batch_size, channels, height, width)

    classifier = KerasRegressor(build_fn=build_model, batch_size=10)

    parameters = {'epochs': [100, 200],
                  # 'batch_size': [2000],
                  'learn_rate': loguniform(1e-5, 100)}
    random_search = RandomizedSearchCV(estimator=classifier,
                                       param_grid=parameters,
                                       scoring="neg_root_mean_squared_error",
                                       cv=5,
                                       n_jobs=-1)

    # Define directory where training data is stored.
    directory_str = '/data/users/jbowyer/cbh_challenge_data/'
    directory = os.fsencode(directory_str)

    file = os.listdir(directory)[0]
    filename = directory_str + os.fsdecode(file)
    print('Loading file', filename)
    result = load_file_cnn_sr(filename)
    data_high_res = result['data_hr']
    data_med_res = result['data_mr']
    data_low_res = result['data_lr']

    s = np.random.permutation(np.arange(0, 210, 1))

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
    n_train = int(0.8 * n_samples)
    trainX = X[:n_train]
    testX = X[n_train:]
    trainy = y[:n_train]
    testy = y[n_train:]

    model = random_search.fit(trainX, trainy, validation_data=(testX, testy), epochs=1, batch_size=10)

    # Write things out to file (for reading back in later to make predictions).
    best_model = model.best_estimator_.model
    model_json = best_model.to_json()
    with open("models_and_weights/cnn_sr_model.json", "w") as json_file:
        json_file.write(model_json)
        json_file.close()

    best_model.save_weights("models_and_weights/cnn_sr_saved_weights.h5")

    fileout = 'Text_outputs/cbh_ml_gridsearch_best_parameters' + today + '.txt'
    file_object = open(fileout, 'a')
    file_object.write("Best: %f using %s" % (model.best_score_, model.best_params_) + "\n")
    file_object.close()

    fileout = 'Text_outputs/cbh_ml_gridsearch_all_parameters' + today + '.txt'
    file_object = open(fileout, 'a')
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    params = model.cv_results_['params']
    file_object.write("Model \n")
    for mean, stdev, param in zip(means, stds, params):
        file_object.write("%f (%f) with: %r" % (mean, stdev, param) + "\n")
    file_object.write("\n")
    file_object.close()

    # # Set up arrays to store the rmse as one loops through multiple sets of multiple epochs.
    # keep_rmse_train = np.empty([1, 1]) + np.nan
    # keep_rmse_val = np.empty([1, 1]) + np.nan
    # s = np.random.permutation(np.arange(0, 210, 1))
    #
    #
    # keep_rmse_train = np.append(keep_rmse_train, history.history['root_mean_squared_error'])
    # keep_rmse_val = np.append(keep_rmse_val, history.history['val_root_mean_squared_error'])
    # # Write out some metrics to track progress. You need to read these back in later and plot them
    # file_out = 'timeseries/cnn_sr_timeseries_rmse_train.txt'
    # np.savetxt(file_out, np.ones((1, 1)) * keep_rmse_train, fmt='%10.7f')
    # file_out = 'timeseries/cnn_sr_timeseries_rmse_val.txt'
    # np.savetxt(file_out, np.ones((1, 1)) * keep_rmse_val, fmt='%10.7f')

    print('All training completed successfully!')


if __name__ == '__main__':
    main()
