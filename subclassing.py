"""
Super resolution model using model and layer subclassing.
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class BlockLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        outputs = inputs
        return outputs


class SuperResolution(tf.keras.Model):
    def __init__(self, upscale_factor, channels):
        super().__init__()
        conv_args = {
            "kernel_initializer": "Orthogonal",
            "padding": "same",
        }
        self.ip = layers.Input(shape=(None, None, channels))
        self.conv_1 = layers.Conv2D(filters=64, kernel_size=5, **conv_args)
        self.bn1 = layers.BatchNormalization()  # https://keras.io/api/layers/normalization_layers/batch_normalization/
        self.ac1 = layers.ReLU()

        self.conv_2 = layers.Conv2D(filters=64, kernel_size=3, **conv_args)
        self.bn2 = layers.BatchNormalization()
        self.ac2 = layers.ReLU()

        self.conv_3 = layers.Conv2D(filters=32, kernel_size=3, **conv_args)
        self.bn3 = layers.BatchNormalization()
        self.ac3 = layers.ReLU()

        self.conv_4 = layers.Conv2D(filters=channels * (upscale_factor ** 2), kernel_size=3, **conv_args)
        self.bn4 = layers.BatchNormalization()
        self.ac4 = layers.ReLU()

        self.upscale_factor = upscale_factor

    def call(self, inputs):
        x = self.ip(inputs)
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.conv_2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.conv_3(x)
        x = self.bn3(x)
        x = self.ac3(x)
        x = self.conv_4(x)
        x = self.bn4(x)
        x = self.ac4(x)
        return tf.nn.depth_to_space(x, self.upscale_factor)


class SimpleSuperResolution(tf.keras.Model):
    def __init__(self, upscale_factor, channels):
        super().__init__()
        conv_args = {
            "kernel_initializer": "Orthogonal",
            "padding": "same",
        }
        self.ip = layers.Input(shape=(None, None, channels))
        self.conv_1 = layers.Conv2D(filters=64, kernel_size=5, **conv_args)
        self.ac1 = layers.ReLU()
        self.conv_2 = layers.Conv2D(filters=64, kernel_size=3, **conv_args)
        self.ac2 = layers.ReLU()
        self.conv_3 = layers.Conv2D(filters=32, kernel_size=3, **conv_args)
        self.ac3 = layers.ReLU()
        self.conv_4 = layers.Conv2D(filters=channels * (upscale_factor ** 2), kernel_size=3, **conv_args)
        self.ac4 = layers.ReLU()
        self.upscale_factor = upscale_factor

    def call(self, inputs):
        x = self.ip(inputs)
        x = self.conv_1(x)
        x = self.ac1(x)
        x = self.conv_2(x)
        x = self.ac2(x)
        x = self.conv_3(x)
        x = self.ac3(x)
        x = self.conv_4(x)
        x = self.ac4(x)
        return tf.nn.depth_to_space(x, self.upscale_factor)


class SequentialSuperResolution(tf.keras.Model):
    def __init__(self, upscale_factor, channels):
        super().__init__()
        conv_args = {
            "activation": "relu",
            "kernel_initializer": "Orthogonal",
            "padding": "same",
        }
        self.ip = layers.Input(shape=(None, None, channels))
        self.main = Sequential(
            layers.Conv2D(filters=64, kernel_size=5, **conv_args),
            layers.Conv2D(filters=64, kernel_size=3, **conv_args),
            layers.Conv2D(filters=32, kernel_size=3, **conv_args),
            layers.Conv2D(filters=channels * (upscale_factor ** 2), kernel_size=3, **conv_args)
        )
        self.upscale_factor = upscale_factor

    def call(self, inputs):
        x = self.ip(inputs)
        output = self.main(x)
        return tf.nn.depth_to_space(output, self.upscale_factor)

