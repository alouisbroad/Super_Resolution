"""
GAN example with custom training loop, learnt from here:
https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
"""
import os
import glob
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.losses import MeanSquaredError

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from data_generators import tf_data_generator
from super_resolution_models import super_resolution_upscaling


save_dir = "/data/users/lbroad/Machine_Learning/gan_outputs/"
epochs = 200
batch_size = 10
steps_per_epoch = 200
upscale_factor = 16

file_list = glob.glob('/data/users/jbowyer/cbh_challenge_data/*')
# file_list = glob.glob('/data/users/lbroad/Machine_Learning/super_resolution/test_data/*')
dataset = tf.data.Dataset.from_generator(tf_data_generator, args=(file_list, batch_size),
                                         output_types=(tf.dtypes.float32, tf.dtypes.float32))

validationset = tf.data.Dataset.from_generator(tf_data_generator, args=(file_list[300:], batch_size),
                                               output_types=(tf.dtypes.float32, tf.dtypes.float32))


def discriminator(channels=1):
    """

    :return: GAN discriminator model.
    """
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), **conv_args)(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), **conv_args)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.GlobalMaxPooling2D()(x)
    outputs = layers.Dense(1)(x)

    return keras.Model(inputs, outputs)


discriminator = discriminator(channels=1)
discriminator.summary()

generator = super_resolution_upscaling(upscale_factor=upscale_factor, channels=1)
generator.summary()

# Instantiate one optimizer for the discriminator and another for the generator.
d_optimizer = keras.optimizers.Adam(learning_rate=0.0003)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0004)

# Instantiate a loss function.
loss_fn = MeanSquaredError()


@tf.function
def train_step(low_res, high_res):
    # Create "fake" high res images.
    generated_images = generator(low_res)
    # Combine them with real images
    combined_images = tf.concat([generated_images, high_res], axis=0)

    # Assemble labels discriminating real from fake images
    labels = tf.concat(
        [tf.ones((batch_size, 1)), tf.zeros((high_res.shape[0], 1))], axis=0
    )
    # Add random noise to the labels - important trick!
    labels += 0.05 * tf.random.uniform(labels.shape)

    # Train the discriminator
    with tf.GradientTape() as tape:
        predictions = discriminator(combined_images)
        d_loss = loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

    # Assemble labels that say "all real images"
    misleading_labels = tf.zeros((batch_size, 1))

    # Train the generator (note that we should *not* update the weights
    # of the discriminator)!
    with tf.GradientTape() as tape:
        predictions = discriminator(generator(low_res))
        g_loss = loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
    return d_loss, g_loss, generated_images


for epoch in range(epochs):
    print("\nStart epoch", epoch)

    for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        print("Batch number %d" % (step,))
        if step+1 == steps_per_epoch:
            break
        # Train the discriminator & generator on one batch of real images.
        d_loss, g_loss, generated_images = train_step(x_batch_train, y_batch_train)

        # Logging.
        if step % 10 == 0:
            # Print metrics
            print("discriminator loss at step %d: %.2f" % (step, d_loss))
            print("adversarial loss at step %d: %.2f" % (step, g_loss))

            # Save one generated image
            fig = plt.figure(figsize=(12, 18))
            gs = gridspec.GridSpec(1, 1)
            gs.update(left=0.10, right=0.975, bottom=0.10, top=0.975, wspace=1e-1, hspace=1e-3)

            ax1 = plt.subplot(gs[0])
            cmap1 = ax1.imshow(np.flipud(generated_images[0, :, :, 0]))
            cmap1.set_clim([0, 1])
            plt.title("Epoch: {}, Batch: {}".format(epoch, step))
            plt.colorbar(cmap1, orientation='horizontal')
            plt.savefig(os.path.join(save_dir, "generated_img" + str(step) + ".png"))
            plt.close()
