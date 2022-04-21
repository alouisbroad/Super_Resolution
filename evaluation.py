"""
Used to evaluate the models created for super resolution.
"""
import gc
import sys
import iris
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import model_from_json

from data_preprocessing import generate_LR, normalisation, load_file, make_stash_string


def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz

def main():
    pp_file = '/data/nwp1/frme/ML_minichallenge/dev/20170701T0000Z_glm_pa010.pp'
    stash_codes = {"specific humidity": (0, 10),
                   "pressure": (0, 408),
                   "temperature": (16, 4)}
    variable = "temperature"
    result = make_stash_string(*stash_codes[variable])
    data = iris.load_cube(pp_file, iris.AttributeConstraint(STASH=result['stashstr_iris'])).data
    data = tf.reshape(tf.convert_to_tensor(data), (*data.shape, 1))

    data = normalisation(data, "minmax")
    # model_path = "/home/h05/lbroad/PycharmProjects/Super_Resolution/models/"
    model_path = "/home/h05/lbroad/PycharmProjects/Super_Resolution/"

    fig = plt.figure(figsize=(12, 18))
    gs = gridspec.GridSpec(2, 3)
    gs.update(left=0.10, right=0.975, bottom=0.10, top=0.975, wspace=3e-1, hspace=1e-2)

    n = 0
    ax1 = plt.subplot(gs[0])
    cmap1 = ax1.imshow(np.flipud(data[n, :, :, 0]))
    cmap1.set_clim([0, 1])
    plt.title("HR")
    plt.colorbar(cmap1, orientation='horizontal')

    data = generate_LR(data, 16)

    ax2 = plt.subplot(gs[1])
    cmap2 = ax2.imshow(np.flipud(data[n, :, :, 0]))
    cmap2.set_clim([0, 1])
    plt.title("LR")
    plt.colorbar(cmap2, orientation='horizontal')

    model_name = "250"
    json_file = open(model_path + 'model_{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_path + 'checkpoints/checkpoints'.format(model_name))
    # model.load_weights(model_path + 'model_saved_weights_{}.h5'.format(model_name))

    y = model.predict(data)

    ax3 = plt.subplot(gs[2])
    cmap3 = ax3.imshow(np.flipud(y[n, :, :, 0]))
    cmap3.set_clim([0, 1])
    plt.title("100 Epochs")
    plt.colorbar(cmap3, orientation='horizontal')

    model_name = "150"
    json_file = open(model_path + 'model_{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_path + 'model_saved_weights_{}.h5'.format(model_name))

    y = model.predict(data)

    ax3 = plt.subplot(gs[3])
    cmap3 = ax3.imshow(np.flipud(y[n, :, :, 0]))
    cmap3.set_clim([0, 1])
    plt.title("150 Epochs")
    plt.colorbar(cmap3, orientation='horizontal')

    model_name = "200"
    json_file = open(model_path + 'model_{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_path + 'model_saved_weights_{}.h5'.format(model_name))

    y = model.predict(data)

    ax3 = plt.subplot(gs[4])
    cmap3 = ax3.imshow(np.flipud(y[n, :, :, 0]))
    cmap3.set_clim([0, 1])
    plt.title("200 Epochs")
    plt.colorbar(cmap3, orientation='horizontal')

    model_name = "250"
    json_file = open(model_path + 'model_{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_path + 'model_saved_weights_{}.h5'.format(model_name))

    y = model.predict(data)

    ax3 = plt.subplot(gs[5])
    cmap3 = ax3.imshow(np.flipud(y[n, :, :, 0]))
    cmap3.set_clim([0, 1])
    plt.title("250 Epochs")
    plt.colorbar(cmap3, orientation='horizontal')


if __name__ == '__main__':
    main()
