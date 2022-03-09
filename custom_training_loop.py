"""
Attempt at creating a custom training loop for super resolution.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from functions import SRCallback

epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

