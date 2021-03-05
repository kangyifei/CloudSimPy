import numpy as np

np.random.seed(0)

import tensorflow as tf

try:
    tf.get_logger().setLevel('INFO')
except Exception as exc:
    print(exc)
import warnings

warnings.simplefilter("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import ray
from ray import tune
from ray.tune.examples.utils import get_iris_data

import inspect
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
from sklearn.datasets import load_iris

iris = load_iris()
true_data = iris['data']
true_label = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

def create_model(learning_rate, dense_1, dense_2):
    assert learning_rate > 0 and dense_1 > 0 and dense_2 > 0, "Did you set the right configuration?"
    model = Sequential()
    model.add(Dense(int(dense_1), input_shape=(4,), activation='relu', name='fc1'))
    model.add(Dense(int(dense_2), activation='relu', name='fc2'))
    model.add(Dense(3, activation='softmax', name='output'))
    optimizer = SGD(lr=learning_rate)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


import tensorflow.keras as keras
from ray.tune import track


class TuneReporterCallback(keras.callbacks.Callback):
    """Tune Callback for Keras.

    The callback is invoked every epoch.
    """

    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()

    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        track.log(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss"))


def tune_iris(config):  # TODO: Change me.
    train_x, train_y, test_x, test_y = get_iris_data()
    model = create_model(learning_rate=config.get('lr'), dense_1=config.get('dense_1'),
                         dense_2=config.get('dense_2'))  # TODO: Change me.
    checkpoint_callback = ModelCheckpoint(
        "model.h5", monitor='loss', save_best_only=True, save_freq=2)

    # Enable Tune to make intermediate decisions by using a Tune Callback hook. This is Keras specific.
    callbacks = [checkpoint_callback, TuneReporterCallback()]

    # Train the model
    model.fit(
        train_x, train_y,
        validation_data=(test_x, test_y),
        verbose=0,
        batch_size=10,
        epochs=20,
        callbacks=callbacks)


assert len(inspect.getargspec(tune_iris).args) == 1, "The `tune_iris` function needs to take in the arg `config`."

print("Test-running to make sure this function will run correctly.")
tune.track.init()  # For testing purposes only.
print("Success!")
hyperparameter_space = {
    "lr": tune.loguniform(0.001, 0.1),
    "dense_1": tune.uniform(2, 128),
    "dense_2": tune.uniform(2, 128),
}
num_samples = 20
import numpy as np

np.random.seed(5)
####################################################################################################
################ This is just a validation function for tutorial purposes only. ####################
HP_KEYS = ["lr", "dense_1", "dense_2"]
assert all(key in hyperparameter_space for key in HP_KEYS), (
    "The hyperparameter space is not fully designated. It must include all of {}".format(HP_KEYS))
######################################################################################################

ray.shutdown()  # Restart Ray defensively in case the ray connection is lost.
ray.init(address='auto', log_to_driver=False)

analysis = tune.run(
    tune_iris,
    verbose=1,
    config=hyperparameter_space,
    num_samples=num_samples)

assert len(analysis.trials) == 20, "Did you set the correct number of samples?"
