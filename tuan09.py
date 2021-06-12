import tensorflow as tf
assert tf.__version__ >= "2.0" # TensorFlow ≥2.0 is required
from tensorflow import keras
from tensorflow.keras import callbacks, layers, metrics, regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras.backend import softmax
from tensorflow.python.keras.layers.advanced_activations import ELU, LeakyReLU, ReLU
from tensorflow.python.keras.layers.core import Dropout
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn ≥0.20 is required
import numpy as np


model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, padding='valid', activation='relu', input_shape=(227,227,3)),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='valid'),
    keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='valid'),
    keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu'),
    keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu'),
    keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='valid'),

    keras.layers.Flatten(),

    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(1000, activation='softmax')
])

model.summary();
# Total params: 62,378,344