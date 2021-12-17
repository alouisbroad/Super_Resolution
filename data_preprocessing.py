"""
Viewing and preparing data for super resolution.
"""
import numpy as np
import iris
import iris.analysis
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def load_file_cnn_sr(pp_file):
    """
    Load in model diagnostics from pp file (for machine learning of cloud base height).
    Each file contains 3d data (no time index).
    :param pp_file:
    :return:
    """
    warnings.filterwarnings("ignore", "HybridHeightFactory")
    warnings.filterwarnings("ignore", "orography")

    # Load in specific humidity
    result = make_stash_string(0, 10)
    cube = iris.load_cube(pp_file, iris.AttributeConstraint(STASH=result['stashstr_iris']))
    qv_hr = cube.data

    latitudes_hr = cube.coord('latitude').points
    longitudes_hr = cube.coord('longitude').points
    latitudes_mr = np.linspace(latitudes_hr[0], latitudes_hr[-1], len(latitudes_hr) // 20)
    longitudes_mr = np.linspace(longitudes_hr[0], longitudes_hr[-1], len(longitudes_hr) // 20)
    latitudes_lr = np.linspace(latitudes_hr[0], latitudes_hr[-1], len(latitudes_hr) // 40)
    longitudes_lr = np.linspace(longitudes_hr[0], longitudes_hr[-1], len(longitudes_hr) // 40)

    cube = cube.interpolate([('latitude', latitudes_mr),
                             ('longitude', longitudes_mr)],
                            iris.analysis.Linear())
    qv_mr = cube.data

    cube = cube.interpolate([('latitude', latitudes_lr),
                             ('longitude', longitudes_lr)],
                            iris.analysis.Linear())
    qv_lr = cube.data

    # Load in pressure on theta levels
    result = make_stash_string(0, 408)
    cube = iris.load_cube(pp_file, iris.AttributeConstraint(STASH=result['stashstr_iris']))
    pres_hr = cube.data

    cube = cube.interpolate([('latitude', latitudes_mr),
                             ('longitude', longitudes_mr)],
                            iris.analysis.Linear())
    pres_mr = cube.data

    cube = cube.interpolate([('latitude', latitudes_lr),
                             ('longitude', longitudes_lr)],
                            iris.analysis.Linear())
    pres_lr = cube.data

    # Load in temperature on theta levels
    result = make_stash_string(16, 4)
    cube = iris.load_cube(pp_file, iris.AttributeConstraint(STASH=result['stashstr_iris']))
    temp_hr = cube.data

    cube = cube.interpolate([('latitude', latitudes_mr),
                             ('longitude', longitudes_mr)],
                            iris.analysis.Linear())
    temp_mr = cube.data

    cube = cube.interpolate([('latitude', latitudes_lr),
                             ('longitude', longitudes_lr)],
                            iris.analysis.Linear())
    temp_lr = cube.data

    # Alternatively, load saved values for all 70 levels
    path = "/data/users/lbroad/Machine_Learning/super_resolution/"
    max_temp = np.load(path + 'max_levels_temp_3m.npy')
    min_temp = np.load(path + 'min_levels_temp_3m.npy')
    max_qv = np.load(path + 'max_levels_qv_3m.npy')
    max_pres = np.load(path + 'max_levels_pres_3m.npy')
    # Normalise/standardise
    for n in range(70):
        temp_hr[n, :] = (temp_hr[n, :] - min_temp[n]) / (max_temp[n] - min_temp[n])
        temp_mr[n, :] = (temp_mr[n, :] - min_temp[n]) / (max_temp[n] - min_temp[n])
        temp_lr[n, :] = (temp_lr[n, :] - min_temp[n]) / (max_temp[n] - min_temp[n])
        qv_hr[n, :] = qv_hr[n, :] / max_qv[n]
        qv_mr[n, :] = qv_mr[n, :] / max_qv[n]
        qv_lr[n, :] = qv_lr[n, :] / max_qv[n]
        pres_hr[n, :] = pres_hr[n, :] / max_pres[n]
        pres_mr[n, :] = pres_mr[n, :] / max_pres[n]
        pres_lr[n, :] = pres_lr[n, :] / max_pres[n]

    # Combine all the variables together into a big array
    data_hr = np.append(np.append(temp_hr, qv_hr, axis=0), pres_hr, axis=0)
    data_mr = np.append(np.append(temp_mr, qv_mr, axis=0), pres_mr, axis=0)
    data_lr = np.append(np.append(temp_lr, qv_lr, axis=0), pres_lr, axis=0)
    return {'data_hr': data_hr,
            'data_mr': data_mr,
            'data_lr': data_lr}


if __name__ == '__main__':
    directory_str = '/data/users/jbowyer/cbh_challenge_data/'
    file = os.listdir(directory_str)[0]
    filename = directory_str + os.fsdecode(file)
    result = load_file_cnn_sr(filename)
    data_high_res = result['data_hr']
    data_med_res = result['data_mr']
    data_low_res = result['data_lr']

    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 2)
    gs.update(left=0.10, right=0.975, bottom=0.10, top=0.975, wspace=1e-1, hspace=1e-3)

    n = 0
    ax1 = plt.subplot(gs[0])
    cmap1 = ax1.imshow(np.flipud(data_low_res[n]))
    cmap1.set_clim([0, 1])
    plt.colorbar(cmap1, orientation='horizontal')

    ax2 = plt.subplot(gs[1])
    cmap2 = ax2.imshow(np.flipud(data_med_res[n]))
    cmap2.set_clim([0, 1])
    plt.colorbar(cmap2, orientation='horizontal')

    ax4 = plt.subplot(gs[2])
    cmap4 = ax4.imshow(np.flipud(data_high_res[n]))
    cmap4.set_clim([0, 1])
    plt.colorbar(cmap4, orientation='horizontal')

