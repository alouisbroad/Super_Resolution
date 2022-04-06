"""
GAN example using model subclassing, learnt from here:
https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/
"""
import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_generators import tf_data_generator
from super_resolution_models import super_resolution_upscaling


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


class SuperResolutionGAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(SuperResolutionGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(SuperResolutionGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def call(self, data, training=False):
        # You don't need this method for training.
        # So just pass.
        # https://github.com/keras-team/keras-io/issues/38  - known issue
        pass

    def train_step(self, data):

        low_res, high_res = data
        # Decode them to fake images
        batch_size = tf.shape(high_res)[0]
        generated_images = self.generator(low_res)

        # Combine them with real images
        combined_images = tf.concat([generated_images, high_res], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(low_res))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}


save_dir = "/data/users/lbroad/Machine_Learning/gan_outputs/"
epochs = 2
batch_size = 10
steps_per_epoch = 2
upscale_factor = 16

gan = SuperResolutionGAN(discriminator=discriminator(channels=1),
                         generator=super_resolution_upscaling(upscale_factor=upscale_factor, channels=1))
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

file_list = glob.glob('/data/users/jbowyer/cbh_challenge_data/*')
dataset = tf.data.Dataset.from_generator(tf_data_generator, args=(file_list, batch_size),
                                         output_types=(tf.dtypes.float32, tf.dtypes.float32))

validationset = tf.data.Dataset.from_generator(tf_data_generator, args=(file_list[300:], batch_size),
                                               output_types=(tf.dtypes.float32, tf.dtypes.float32))

gan.fit(x=dataset,
        validation_data=validationset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=50)
