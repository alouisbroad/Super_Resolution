"""
--
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

from super_resolution_models import interpolate_cbh

if __name__ == '__main__':
    directory_str = '/data/users/jbowyer/cbh_challenge_data/'
    file = os.listdir(directory_str)[0]
    filename = directory_str + os.fsdecode(file)
    result = load_file_cnn_sr(filename)
    data_high_res = result['data_hr']
    data_med_res = result['data_mr']
    data_low_res = result['data_lr']

    # Using interpolation
    n = 0
    low_res_lin = interpolate_cbh(data_low_res[n], 40, interp_type="linear")
    low_res_cub = interpolate_cbh(data_low_res[n], 40, interp_type="cubic")
    med_res_lin = interpolate_cbh(data_med_res[n], 20, interp_type="linear")
    med_res_cub = interpolate_cbh(data_med_res[n], 20, interp_type="cubic")

    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(3, 2)
    gs.update(left=0.10, right=0.975, bottom=0.10, top=0.975, wspace=1e-1, hspace=1e-3)

    ax1 = plt.subplot(gs[0])
    cmap1 = ax1.imshow(np.flipud(low_res_lin))
    cmap1.set_clim([0, 1])
    plt.colorbar(cmap1, orientation='horizontal')

    ax2 = plt.subplot(gs[2])
    cmap2 = ax2.imshow(np.flipud(low_res_cub))
    cmap2.set_clim([0, 1])
    plt.colorbar(cmap2, orientation='horizontal')

    ax3 = plt.subplot(gs[1])
    cmap3 = ax3.imshow(np.flipud(med_res_lin))
    cmap3.set_clim([0, 1])
    plt.colorbar(cmap3, orientation='horizontal')

    ax4 = plt.subplot(gs[3])
    cmap4 = ax4.imshow(np.flipud(med_res_cub))
    cmap4.set_clim([0, 1])
    plt.colorbar(cmap4, orientation='horizontal')

    ax5 = plt.subplot(gs[4])
    cmap5 = ax5.imshow(np.flipud(data_low_res[n]))
    cmap5.set_clim([0, 1])
    plt.colorbar(cmap5, orientation='horizontal')

    ax6 = plt.subplot(gs[5])
    cmap6 = ax6.imshow(np.flipud(data_med_res[n]))
    cmap6.set_clim([0, 1])
    plt.colorbar(cmap6, orientation='horizontal')

    # Deep Learning