"""
Old used functions.
"""
import warnings
import glob
import iris
import numpy as np
from ..data_processing import make_stash_string


def load_file_cnn_sr_hml(pp_file):
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
    max_temp = np.load(path + 'max_levels_temp_3m_16004.npy')
    min_temp = np.load(path + 'min_levels_temp_3m_16004.npy')
    max_qv = np.load(path + 'max_levels_qv_3m_00010.npy')
    max_pres = np.load(path + 'max_levels_pres_3m_00408.npy')
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


def load_file_cnn_sr_hl(pp_file, path, variable=(16, 4), downscale=10):
    """

    :param pp_file:
    :param path:
    :param variable:
    :param downscale:
    """
    result = make_stash_string(*variable)
    cube = iris.load_cube(pp_file, iris.AttributeConstraint(STASH=result['stashstr_iris']))
    hr = cube.data
    latitudes_hr = cube.coord('latitude').points
    longitudes_hr = cube.coord('longitude').points
    latitudes_lr = np.linspace(latitudes_hr[0], latitudes_hr[-1], len(latitudes_hr) // downscale)
    longitudes_lr = np.linspace(longitudes_hr[0], longitudes_hr[-1], len(longitudes_hr) // downscale)

    cube = cube.interpolate([('latitude', latitudes_lr),
                             ('longitude', longitudes_lr)],
                            iris.analysis.Linear())
    lr = cube.data
    if result["stashstr_fout"] in ["16004"]:
        max_levels = np.load(glob.glob(path + 'max_levels_*_3m_{}.npy'.format(result["stashstr_fout"]))[0])
        min_levels = np.load(glob.glob(path + 'min_levels_*_3m_{}.npy'.format(result["stashstr_fout"]))[0])
        # Normalise
        for n in range(max_levels.shape[0]):
            hr[n, :] = (hr[n, :] - min_levels[n]) / (max_levels[n] - min_levels[n])
            lr[n, :] = (lr[n, :] - min_levels[n]) / (max_levels[n] - min_levels[n])
    else:
        max_levels = np.load(glob.glob(path + '*_3m_{}.npy'.format(result["stashstr_fout"]))[0])
        # Normalise
        for n in range(max_levels.shape[0]):
            hr[n, :] = hr[n, :] / max_levels[n]
            lr[n, :] = lr[n, :] / max_levels[n]

    return hr, lr


def load_cnn_sr_hl(pp_file, variable="all", downscale=10):
    """
    Load in model diagnostics from pp file (for machine learning of cloud base height).
    Each file contains 3d data (no time index).
    :param pp_file:
    :param variable:
    :param downscale:
    :return:
    """
    warnings.filterwarnings("ignore", "HybridHeightFactory")
    warnings.filterwarnings("ignore", "orography")

    path = "/data/users/lbroad/Machine_Learning/super_resolution/"
    stash_codes = {"specific humidity": (0, 10),
                   "pressure": (0, 408),
                   "temperature": (16, 4)}

    if variable == "all":
        hr_shape = 0
        for vars in stash_codes.keys():
            hr, lr = load_file(pp_file=pp_file, path=path, variable=stash_codes[vars], downscale=downscale)
            if hr_shape is 0:
                hr_shape = 1
                hr_arr = np.empty(hr.shape, float)
                lr_arr = np.empty(lr.shape, float)
            hr_arr = np.append(hr_arr, hr, axis=0)
            lr_arr = np.append(lr_arr, lr, axis=0)
        hr = hr_arr
        lr = lr_arr
    else:
        hr, lr = load_file(pp_file=pp_file, path=path, variable=stash_codes[variable], downscale=downscale)

    return {'data_hr': hr,
            'data_lr': lr}