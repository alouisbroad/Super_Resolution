"""
Jack's code from here: /home/h05/jbowyer/PycharmProjects/MachineLearning/load_in_one_file_for_cbh_ml_jbowyer.py
"""
import numpy as np
import iris
import iris.analysis
import warnings


def make_stash_string(stashsec, stashcode):
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
    return {'stashstr_iris': stashstr_iris, 'stashstr_fout': stashstr_fout};


def load_in_one_file_for_cbh_ml(pp_file):
    warnings.filterwarnings("ignore", "HybridHeightFactory")
    warnings.filterwarnings("ignore", "orography")
    # Load in model diagnostics from pp file (for machine learning of cloud base height).
    # Each file contains 3d data (no time index)
    #
    # Load in specific humidity
    result = make_stash_string(0, 10)
    data = iris.load_cube(pp_file, iris.AttributeConstraint(STASH=result['stashstr_iris']))
    qv = data.data
    # Load in bulk cloud fraction
    result = make_stash_string(0, 266)
    data = iris.load_cube(pp_file, iris.AttributeConstraint(STASH=result['stashstr_iris']))
    bcf = data.data
    # Load in pressure on theta levels
    result = make_stash_string(0, 408)
    data = iris.load_cube(pp_file, iris.AttributeConstraint(STASH=result['stashstr_iris']))
    pres = data.data
    # Load in temperature on theta levels
    result = make_stash_string(16, 4)
    data = iris.load_cube(pp_file, iris.AttributeConstraint(STASH=result['stashstr_iris']))
    temp = data.data
    # Data is 3d, but it will be easier to work with if the lat/lon are unfolded
    # and we have a 2d distance-height curtain.
    nz, ny, nx = temp.shape
    temp = np.reshape(temp, (nz, ny * nx))
    qv = np.reshape(qv, (nz, ny * nx))
    bcf = np.reshape(bcf, (nz, ny * nx))
    pres = np.reshape(pres, (nz, ny * nx))
    # Create new array to hold cloud base height
    nz, ntotal = bcf.shape
    cbh = np.copy(bcf) * 0.0
    # Set a threshold for determining that cloud base has been found (e.g. 2 oktas)
    thresh = 2.0 / 8.0
    # Simple search algorithm (done in a noddy way to be clear what is going on).
    for i in np.arange(0, ntotal, 1):
        found = 0
        for k in np.arange(0, nz, 1):
            if found == 0 and bcf[k, i] > thresh:
                cbh[k, i] = 1.0
                found = 1
    # Prepare to Normalise/standardise
    # Comment these back in to print out max and min values to help inform choice of hardwired values.
    # NB. You do NOT want to normalise each file using its own max and mean, because each file will then be normalised slightly differently!
    # print('maxtemp',np.amax(temp))
    # print('mintemp',np.amin(temp))
    # print('maxqv',np.amax(qv))
    # print('maxp',np.amax(pres))
    # Hardwired values found from inspecting one file [0] all files for 1 month period [1]
    # and all files for 3 month period [2] (global max and min)
    # max_temp = [320.0, 317.875, 318.375]
    # min_temp = [140.0, 144.0, 144.0]
    # max_qv   = [0.025, 0.02396667, 0.02521837]
    # max_pres = [106000.0, 105871.75, 106368.5]
    # Normalise/standardise
    # temp = (temp-min_temp[1]) / (max_temp[1]-min_temp[1])
    # qv   = qv / max_qv[1]
    # pres = pres / max_pres[1]
    # Alternatively, load saved values for all 70 levels
    # max_temp = np.load('max_levels_temp.npy')
    # min_temp = np.load('min_levels_temp.npy')
    # max_qv = np.load('max_levels_qv.npy')
    # max_pres = np.load('max_levels_pres.npy')
    # # Normalise/standardise
    # for n in range(70):
    #     temp[n, :] = (temp[n, :] - min_temp[n]) / (max_temp[n] - min_temp[n])
    #     qv[n, :] = qv[n, :] / max_qv[n]
    #     pres[n, :] = pres[n, :] / max_pres[n]
    # If no cloud base has been found, then set cloud base to be in top most layer of model (real cloud base is NEVER that high).
    for i in np.arange(0, ntotal, 1):
        if np.amax(cbh[:, i]) < 0.5:
            cbh[69, i] = 1.0
    # Combine all the variables together into a big array
    data = np.append(np.append(np.append(temp, qv, axis=0), pres, axis=0), cbh, axis=0)
    return {'data': data}


def load_file_cnn_sr(pp_file):
    warnings.filterwarnings("ignore", "HybridHeightFactory")
    warnings.filterwarnings("ignore", "orography")
    # Load in model diagnostics from pp file (for machine learning of cloud base height).
    # Each file contains 3d data (no time index)
    #
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

    # Load in bulk cloud fraction
    # result = make_stash_string(0, 266)
    # data   = iris.load_cube(pp_file, iris.AttributeConstraint(STASH=result['stashstr_iris']))
    # bcf    = data.data
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

    # Data is 3d, but it will be easier to work with if the lat/lon are unfolded
    # and we have a 2d distance-height curtain.
    # nz, ny, nx = temp.shape
    # temp     = np.reshape(temp, (nz, ny*nx))
    # qv       = np.reshape(qv,   (nz, ny*nx))
    # bcf      = np.reshape(bcf,  (nz, ny*nx))
    # pres     = np.reshape(pres, (nz, ny*nx))
    # Create new array to hold cloud base height
    # nz, ntotal = bcf.shape
    # cbh = np.copy(bcf)*0.0
    # Set a threshold for determining that cloud base has been found (e.g. 2 oktas)
    # thresh = 2.0/8.0
    # Simple search algorithm (done in a noddy way to be clear what is going on).
    # for i in np.arange(0,ntotal,1):
    #     found=0
    #     for k in np.arange(0,nz,1):
    #         if found==0 and bcf[k,i]>thresh:
    #             cbh[k,i]=1.0
    #             found=1
    # Prepare to Normalise/standardise
    # Comment these back in to print out max and min values to help inform choice of hardwired values.
    # NB. You do NOT want to normalise each file using its own max and mean, because each file will then be normalised slightly differently!
    # print('maxtemp',np.amax(temp))
    # print('mintemp',np.amin(temp))
    # print('maxqv',np.amax(qv))
    # print('maxp',np.amax(pres))
    # Hardwired values found from inspecting one file [0] all files for 1 month period [1]
    # and all files for 3 month period [2] (global max and min)
    # max_temp = [320.0, 317.875, 318.375]
    # min_temp = [140.0, 144.0, 144.0]
    # max_qv   = [0.025, 0.02396667, 0.02521837]
    # max_pres = [106000.0, 105871.75, 106368.5]
    # Normalise/standardise
    # temp = (temp-min_temp[1]) / (max_temp[1]-min_temp[1])
    # qv   = qv / max_qv[1]
    # pres = pres / max_pres[1]
    # Alternatively, load saved values for all 70 levels
    max_temp = np.load('max_levels_temp_3m.npy')
    min_temp = np.load('min_levels_temp_3m.npy')
    max_qv = np.load('max_levels_qv_3m.npy')
    max_pres = np.load('max_levels_pres_3m.npy')
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
    # If no cloud base has been found, then set cloud base to be in top most layer of model (real cloud base is NEVER that high).
    # for i in np.arange(0, ntotal, 1):
    #     if np.amax(cbh[:, i]) < 0.5:
    #         cbh[69, i] = 1.0
    # Combine all the variables together into a big array
    data_hr = np.append(np.append(temp_hr, qv_hr, axis=0), pres_hr, axis=0)
    data_mr = np.append(np.append(temp_mr, qv_mr, axis=0), pres_mr, axis=0)
    data_lr = np.append(np.append(temp_lr, qv_lr, axis=0), pres_lr, axis=0)
    return {'data_hr': data_hr,
            'data_mr': data_mr,
            'data_lr': data_lr}
