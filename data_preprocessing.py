"""
Viewing and preparing data for super resolution.
"""
import numpy as np
import iris
import iris.analysis
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.layers import AveragePooling2D
import tensorflow as tf
import gc


def make_stash_string(stashsec, stashcode):
    """

    :param stashsec:
    :param stashcode:
    :return:
    """
    #
    stashsecstr = str(stashsec)
    if stashsec < 10:
        stashsecstr = '0' + stashsecstr
    # endif
    #
    stashcodestr = str(stashcode)
    if stashcode < 100:
        stashcodestr = '0' + stashcodestr
    # endif
    if stashcode < 10:
        stashcodestr = '0' + stashcodestr
    # endif
    stashstr_iris = 'm01s' + stashsecstr + 'i' + stashcodestr
    stashstr_fout = stashsecstr + stashcodestr
    return {'stashstr_iris': stashstr_iris, 'stashstr_fout': stashstr_fout}


def generate_LR(original_data, factor=16):
    """
    Use average pooling layer to shrink by 1/factor
    :param original_data:
    :param factor:
    :return:
    """
    return AveragePooling2D(pool_size=(factor, factor))(original_data)


def normalisation_by_channels(data, ntype):
    """
    :param data: input data - tensor or array
    :param ntype: Min-max normalisation or z-score (standard score) normalisation
    :return: normalised data
    """
    _, _, _, c = data.shape
    result = []
    if ntype == "minmax":
        for i in range(c):
            norm = np.array(data[:, :, :, i])
            norm = (norm - norm.min()) / (norm.max() - norm.min())
            result.append(tf.convert_to_tensor(norm))
        return tf.stack(result, axis=3)
    elif ntype == "zscore":
        mu, variance = tf.nn.moments(data, axes=[0, 1, 2])
        for i in range(c):
            norm = (data - mu) / tf.math.sqrt(variance)
            result.append(norm)
        return tf.stack(result, axis=3)
    else:
        "Incorrect arguments, either minmax or zscore."


def normalisation(data, ntype):
    """
    :param data: input data - tensor or array
    :param ntype: Min-max normalisation or z-score (standard score) normalisation
    :return: normalised data
    """
    if ntype == "minmax":
        norm = np.array(data)
        norm = (norm - norm.min()) / (norm.max() - norm.min())
        return tf.convert_to_tensor(norm)
    elif ntype == "zscore":
        mu, variance = tf.nn.moments(data, axes=[0, 1, 2, 3])
        return (data - mu) / tf.math.sqrt(variance)
    else:
        "Incorrect arguments, either minmax or zscore."


def denormalisation(data, mu_var):
    """
    :param data: tensor of normalised data
    :param mu_var: mu and var
    :return: de-normalised data
    """
    return data * tf.math.sqrt(mu_var[1]) + mu_var[0]


def load_file(path, variable="temperature"):
    """
    Loads and converts data to tensor with 1 channel.
    :param path: Directory where your files are.
    :param variable: variable you want to extract.
    :return Tensor of data.
    """
    stash_codes = {"specific humidity": (0, 10),
                   "pressure": (0, 408),
                   "temperature": (16, 4)}
    if variable == "all":
        tmp = make_stash_string(*stash_codes["temperature"])['stashstr_iris']
        prs = make_stash_string(*stash_codes["pressure"])['stashstr_iris']
        sph = make_stash_string(*stash_codes["specific humidity"])['stashstr_iris']
        tmp_list = []
        prs_list = []
        sph_list = []
        for foldername, _, filenames in os.walk(path):
            for file in filenames:
                temperature = np.array(iris.load_cube(os.path.join(foldername, file),
                                                      iris.AttributeConstraint(STASH=tmp)).data)
                pressure = np.array(iris.load_cube(os.path.join(foldername, file),
                                                   iris.AttributeConstraint(STASH=prs)).data)
                sphumidity = np.array(iris.load_cube(os.path.join(foldername, file),
                                                     iris.AttributeConstraint(STASH=sph)).data)
                tmp_list.append(temperature)
                prs_list.append(pressure)
                sph_list.append(sphumidity)
        del temperature, pressure, sphumidity
        gc.collect()
        lvl, la, lo = tmp_list[0].shape
        tmp_list = tf.convert_to_tensor(np.array(tmp_list).reshape(len(tmp_list)*lvl, la, lo, 1))
        prs_list = tf.convert_to_tensor(np.array(prs_list).reshape(len(prs_list)*lvl, la, lo, 1))
        sph_list = tf.convert_to_tensor(np.array(sph_list).reshape(len(sph_list)*lvl, la, lo, 1))
        print("Converting to Tensor")
        return tf.concat([tmp_list, prs_list, sph_list], axis=3)
    else:
        result = make_stash_string(*stash_codes[variable])
        data_list = []
        for foldername, _, filenames in os.walk(path):
            for file in filenames:
                data = iris.load_cube(os.path.join(foldername, file),
                                      iris.AttributeConstraint(STASH=result['stashstr_iris'])).data
                data_list.append(data)
        c, lvl, lon, lat = np.array(data_list).shape
        print("Converting to Tensor.")
        return tf.reshape(tf.convert_to_tensor(data_list), (lvl*c, lon, lat, 1))


if __name__ == '__main__':
    # directory_str = '/data/users/jbowyer/cbh_challenge_data/'
    # directory_str = '/data/nwp1/frme/ML_minichallenge/train/'
    directory_str = '/data/users/lbroad/Machine_Learning/super_resolution/test_data/'

    high_res = load_file(directory_str)
    high_res_mm = normalisation(high_res, "minmax")
    high_res_zs = normalisation(high_res, "zscore")

    lr_mm = generate_LR(high_res_mm, 16)
    lr_zs = generate_LR(high_res_zs, 16)
    print(lr_zs.shape)

    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 2)
    gs.update(left=0.10, right=0.975, bottom=0.10, top=0.975, wspace=1e-1, hspace=1e-3)

    n = 0
    ax1 = plt.subplot(gs[0])
    cmap1 = ax1.imshow(np.flipud(high_res_mm[n, :, :, 0]))
    cmap1.set_clim([0, 1])
    plt.colorbar(cmap1, orientation='horizontal')

    ax2 = plt.subplot(gs[1])
    cmap2 = ax2.imshow(np.flipud(lr_mm[n, :, :, 0]))
    cmap2.set_clim([0, 1])
    plt.colorbar(cmap2, orientation='horizontal')

    ax3 = plt.subplot(gs[2])
    cmap3 = ax3.imshow(np.flipud(high_res_zs[n, :, :, 0]))
    cmap3.set_clim([-4, 4])
    plt.colorbar(cmap3, orientation='horizontal')

    ax4 = plt.subplot(gs[3])
    cmap4 = ax4.imshow(np.flipud(lr_zs[n, :, :, 0]))
    cmap4.set_clim([-4, 4])
    plt.colorbar(cmap4, orientation='horizontal')

    # 3 Channels
    channeled = load_file(directory_str, variable="all")

    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 2)
    gs.update(left=0.10, right=0.975, bottom=0.10, top=0.975, wspace=1e-1, hspace=1e-3)

    n = 0
    ax1 = plt.subplot(gs[0])
    cmap1 = ax1.imshow(np.flipud(channeled[n, :, :, 0]))
    plt.colorbar(cmap1, orientation='horizontal')

    ax2 = plt.subplot(gs[1])
    cmap2 = ax2.imshow(np.flipud(channeled[n, :, :, 1]))
    plt.colorbar(cmap2, orientation='horizontal')

    ax3 = plt.subplot(gs[2])
    cmap3 = ax3.imshow(np.flipud(channeled[n, :, :, 2]))
    plt.colorbar(cmap3, orientation='horizontal')

    channeled = normalisation_by_channels(channeled, "minmax")
    # high_res_zs = normalisation_by_channels(channeled, "zscore")

    lr_channeled = generate_LR(channeled, 16)
