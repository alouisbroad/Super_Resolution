"""
Supporting functions
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import sys

from data_preprocessing import generate_LR


class SRCallback(tf.keras.callbacks.Callback):
    """
    https: // keras.io / examples / vision / super_resolution_sub_pixel /
    """
    def __init__(self, callback_data):
        super(SRCallback, self).__init__()
        self.test_data = generate_LR(callback_data)

    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        if epoch % 20 == 0:
            prediction = upscale_image(self.model, self.test_data)
            plot_results(prediction, "epoch-" + str(epoch), "Prediction")

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))

    def on_predict_end(self, logs=None):
        self.psnr_final = 10 * math.log10(1 / logs["loss"])
        print("PSNR: {}".format(self.psnr_final))


def upscale_image(model, dataset):
    """
    Upscales a data.

    TODO: Implement denormalisation.
    """
    # de-normalise
    # data = denormalisation(dataset, mu_var=mu_var)
    return model.predict(dataset)


def plot_results(prediction, prefix, title):
    """
    Create an image showing upscaled data.
    """
    fig = plt.figure(figsize=(12, 18))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.10, right=0.975, bottom=0.10, top=0.975, wspace=1e-1, hspace=1e-3)

    ax1 = plt.subplot(gs[0])
    cmap1 = ax1.imshow(np.flipud(prediction[0, :, :, 0]))
    cmap1.set_clim([0, 1])
    plt.title("{}".format(title))
    plt.colorbar(cmap1, orientation='horizontal')
    plt.savefig("/data/users/lbroad/Machine_Learning/training_images/{}_{}.png".format(prefix, title))


# Interpolation kernel
def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a + 2) * (abs(s) ** 3) - (a + 3) * (abs(s) ** 2) + 1

    elif (abs(s) > 1) & (abs(s) <= 2):
        return a * (abs(s) ** 3) - (5 * a) * (abs(s) ** 2) + (8 * a) * abs(s) - 4 * a
    return 0


# Padding for interpolation
def padding(img, H, W, C):
    zimg = np.zeros((H + 4, W + 4, C))
    zimg[2:H + 2, 2:W + 2, :C] = img

    # Pad the first/last two col and row
    zimg[2:H + 2, 0:2, :C] = img[:, 0:1, :C]
    zimg[H + 2:H + 4, 2:W + 2, :] = img[H - 1:H, :, :]
    zimg[2:H + 2, W + 2:W + 4, :] = img[:, W - 1:W, :]
    zimg[0:2, 2:W + 2, :C] = img[0:1, :, :C]

    # Pad the missing eight points
    zimg[0:2, 0:2, :C] = img[0, 0, :C]
    zimg[H + 2:H + 4, 0:2, :C] = img[H - 1, 0, :C]
    zimg[H + 2:H + 4, W + 2:W + 4, :C] = img[H - 1, W - 1, :C]
    zimg[0:2, W + 2:W + 4, :C] = img[0, W - 1, :C]

    return zimg


# Bicubic operation
def bicubic(img, ratio, a):
    """
    Function for bi-cubic interpolation of a square grid.
    Found and edited from here https://www.geeksforgeeks.org/python-opencv-bicubic-interpolation-for-resizing-image/

    :param img:
    :param ratio:
    :param a:
    :return:

    TODO: Make adjustments for use on rectangular grids.
    """
    # Get image size
    H, W, C = img.shape

    # Here H = Height, W = weight,
    # C = Number of channels if the
    # image is coloured.
    img = padding(img, H, W, C)

    # Create new image
    dH = math.floor(H * ratio)
    dW = math.floor(W * ratio)

    # Converting into matrix
    dst = np.zeros((dH, dW, C))

    # np.zeroes generates a matrix
    # consisting only of zeroes
    # Here we initialize our answer
    # (dst) as zero

    h = 1 / ratio

    print('Start bicubic interpolation')
    print('It will take a little while...')
    inc = 0

    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                # Getting the coordinates of the
                # nearby values
                x, y = i * h + 2, j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                # Considering all nearby 16 values
                mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.matrix([[img[int(y - y1), int(x - x1), c],
                                    img[int(y - y2), int(x - x1), c],
                                    img[int(y + y3), int(x - x1), c],
                                    img[int(y + y4), int(x - x1), c]],
                                   [img[int(y - y1), int(x - x2), c],
                                    img[int(y - y2), int(x - x2), c],
                                    img[int(y + y3), int(x - x2), c],
                                    img[int(y + y4), int(x - x2), c]],
                                   [img[int(y - y1), int(x + x3), c],
                                    img[int(y - y2), int(x + x3), c],
                                    img[int(y + y3), int(x + x3), c],
                                    img[int(y + y4), int(x + x3), c]],
                                   [img[int(y - y1), int(x + x4), c],
                                    img[int(y - y2), int(x + x4), c],
                                    img[int(y + y3), int(x + x4), c],
                                    img[int(y + y4), int(x + x4), c]]])
                mat_r = np.matrix(
                    [[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])

                # Here the dot function is used to get the dot
                # product of 2 matrices
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

    # If there is an error message, it
    # directly goes to stderr
    sys.stderr.write('\n')

    # Flushing the buffer
    sys.stderr.flush()
    return dst


def denormalisation(data, mu_var):
    """
    :param data: tensor of normalised data
    :param mu_var: mu and var
    :return: de-normalised data
    """
    return data * tf.math.sqrt(mu_var[1]) + mu_var[0]
