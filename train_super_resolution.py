"""
Script to run machine learning scripts.
"""
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from data_preprocessing import generate_LR, normalisation, load_file
from super_resolution_models import sr_cnn_functional

def main():
    upscale_factor = 16

    # Load and prepare data.
    # directory_str = '/data/users/jbowyer/cbh_challenge_data/'
    directory_str = '/data/users/lbroad/Machine_Learning/super_resolution/test_data/'
    hr = load_file(directory_str)
    hr = normalisation(hr, "minmax")
    hr = tf.random.shuffle(hr, seed=None)
    lr = generate_LR(hr, upscale_factor)

    train_size = 0.8
    n_samples, rows, cols, n_channels = lr.shape
    X_train, X_test = tf.split(lr, (round(n_samples * train_size), round(n_samples * (1 - train_size))), axis=0)
    y_train, y_test = tf.split(hr, (round(n_samples * train_size), round(n_samples * (1 - train_size))), axis=0)

    # Create model
    model = sr_cnn_functional(upscale_factor=upscale_factor, channels=1)
    model.summary()

    opt = Adam(learning_rate=1.0e-4)
    model.compile(optimizer=opt, loss='mse', metrics=[RootMeanSquaredError()])
    model_output = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=2, epochs=50, batch_size=10)

if __name__ == '__main__':
    main()

