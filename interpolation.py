"""
Using several forms of interpolation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from data_preprocessing import load_file, normalisation, generate_LR
from super_resolution_models import interpolate_cbh

if __name__ == '__main__':
    directory_str = '/data/users/lbroad/Machine_Learning/super_resolution/test_data/'
    hr = load_file(directory_str)
    hr = normalisation(hr, "minmax")
    lr = generate_LR(hr, 16)

    # Using interpolation
    lr_rs = tf.reshape(lr, (2, 70, 30, 40))
    m = 0
    n = 3
    low_res_lin = interpolate_cbh(lr_rs[m, n], 16, interp_type="linear")
    low_res_cub = interpolate_cbh(lr_rs[m, n], 16, interp_type="cubic")
    low_res_bilin = interpolate_cbh(lr, 16, interp_type="bilinear")
    # low_res_bicub = interpolate_cbh(lr, 16, interp_type="bicubic")
    # low_res_nn = interpolate_cbh(lr, 16, interp_type="nearest")  # Dreadful

    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 3)
    gs.update(left=0.10, right=0.975, bottom=0.10, top=0.975, wspace=1e-1, hspace=1e-3)

    ax1 = plt.subplot(gs[0])
    cmap1 = ax1.imshow(np.flipud(low_res_lin))
    cmap1.set_clim([0, 1])
    plt.colorbar(cmap1, orientation='horizontal')

    ax2 = plt.subplot(gs[1])
    cmap2 = ax2.imshow(np.flipud(low_res_cub))
    cmap2.set_clim([0, 1])
    plt.colorbar(cmap2, orientation='horizontal')

    ax3 = plt.subplot(gs[2])
    cmap3 = ax3.imshow(np.flipud(low_res_bilin[n, :, :, 0]))
    cmap3.set_clim([0, 1])
    plt.colorbar(cmap3, orientation='horizontal')

    # ax4 = plt.subplot(gs[3])
    # cmap4 = ax4.imshow(np.flipud(low_res_bicub[n, :, :, 0]))
    # cmap4.set_clim([0, 1])
    # plt.colorbar(cmap4, orientation='horizontal')

    ax5 = plt.subplot(gs[4])
    cmap5 = ax5.imshow(np.flipud(lr[n, :, :, 0]))
    cmap5.set_clim([0, 1])
    plt.colorbar(cmap5, orientation='horizontal')

    ax6 = plt.subplot(gs[5])
    cmap6 = ax6.imshow(np.flipud(hr[n, :, :, 0]))
    cmap6.set_clim([0, 1])
    plt.colorbar(cmap6, orientation='horizontal')
