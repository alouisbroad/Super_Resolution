"""
https://www.tensorflow.org/guide/data_performance
https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
"""
import glob
import time
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError

from functions import SRCallback
from data_generators import tf_data_generator
from super_resolution_models import super_resolution_upscaling

epochs = 2
batch_size = 10
steps_per_epoch = 10
upscale_factor = 16

file_list = glob.glob('/data/users/jbowyer/cbh_challenge_data/*')

dataset = tf.data.Dataset.from_generator(tf_data_generator, args=(file_list, batch_size),
                                         output_types=(tf.dtypes.float32, tf.dtypes.float32))
validationset = tf.data.Dataset.from_generator(tf_data_generator, args=(file_list[300:], batch_size),
                                               output_types=(tf.dtypes.float32, tf.dtypes.float32))

model = super_resolution_upscaling(upscale_factor=upscale_factor, channels=1)
opt = Adam(learning_rate=1.0e-4)
loss_fn = MeanSquaredError()
train_acc_metric = RootMeanSquaredError()
val_acc_metric = RootMeanSquaredError()

@tf.function
def root_mean_squared_error(y_true, y_pred):  # Use rmse as a custom loss function.
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value


@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)


# Simple training loop
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        print("Batch number %d" % (step,))
        if step+1 == steps_per_epoch:
            break
        loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 200 batches.
        if step % 2 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))
        # Display metrics at the end of each epoch.

    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    # Run a validation loop at the end of each epoch.
    for step, (x_batch_val, y_batch_val) in enumerate(validationset):
        if step+1 == steps_per_epoch:
            break
        test_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Epoch Time: %.2fs" % (time.time() - start_time))

