'''
Main reference: github/ageron/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb
Last review: Feb 2021
'''

# In[0]: IMPORTS AND COMMON SETTINGS
#region
import sys
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
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
import os
import joblib
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(42) # to make the output stable across runs
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
#physical_devices = tf.config.list_physical_devices('GPU') # to run multiple instance of VS code
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#endregion


''' WEEK 03 '''


# Lấy dữ liệu CIFAR-10
# CIFAR-10 là một tập dữ liệu hình ảnh nổi tiếng trong Computer Vision
# Bộ dữ liệu này gồm 60000 hình ảnh chia thành 10 classes với
# mỗi class chứa 6000 hình ảnh với độ phân giải 32x32 pixels
# 10 class của tập dữ liệu này là:
# Airplane
# Car
# Bird
# Cat
# Deer
# Dog
# Frog
# Horse
# Ship
# Truck
# Bộ dữ liệu này đã được tích hợp trong keras, ta chỉ việc lấy về bằng đoạn code sau:
# Tập train gồm 50000 sample và test gồm 10000 sample

(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Scale và chia tập validation/train với tỷ lệ 5000/45000
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


# Thử với l2 normalization
# Dùng với Performance scheduling như lần 3 ở bài 1
new_training = 1
if new_training:
    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
    for ii in range(15):
        noNeurons = 100
        model.add(keras.layers.Dense(noNeurons, activation="selu", kernel_initializer="lecun_normal", kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(10, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.01)))
    model.summary()
    tf.keras.utils.plot_model(model,'models/model_A_bai2_lan1.png',show_shapes=True)

    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer="nadam",
                    metrics=["accuracy"])
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    history = model.fit(X_train, y_train, epochs=25,
                        validation_data=(X_valid, y_valid),
                        callbacks=[lr_scheduler])
    history = history.history
    joblib.dump(history,"models/model_A_bai2_lan1_history")
    model.save("models/model_A_bai2_lan1.h5")
else:
    model = keras.models.load_model("models/model_A_bai2_lan1.h5")
    history = joblib.load("models/model_A_bai2_lan1_history")

print("_____Lan 1:_____")
print(history["val_accuracy"][-1])
#Bởi vì chạy có 5 epoches nên val_accuracy quá thấp
#val_accuracy = 0.16519999504089355

# Thử với 5 epochs
# 36s 26ms/step - loss: 2.2187 - accuracy: 0.2232 - val_loss: 2.2324 - val_accuracy: 0.2294
# val_accuracy: 0.22939999401569366

# Thử với 25 epochs
# 51s 36ms/step - loss: 2.1273 - accuracy: 0.2538 - val_loss: 2.1162 - val_accuracy: 0.2694
# val_accuracy: 0.2694000005722046


# Thử với BatchNormalization
new_training = 1
if new_training:
    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
    for ii in range(15):
        noNeurons = 100
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(noNeurons, activation="selu", kernel_initializer="lecun_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()
    tf.keras.utils.plot_model(model,'models/model_A_bai2_lan2.png',show_shapes=True)

    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer="nadam",
                    metrics=["accuracy"])
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    history = model.fit(X_train, y_train, epochs=25,
                        validation_data=(X_valid, y_valid),
                        callbacks=[lr_scheduler])
    history = history.history
    joblib.dump(history,"models/model_A_bai2_lan2_history")
    model.save("models/model_A_bai2_lan2.h5")
else:
    model = keras.models.load_model("models/model_A_bai2_lan2.h5")
    history = joblib.load("models/model_A_bai2_lan2_history")

print("_____Lan 2:_____")
print(history["val_accuracy"][-1])

# Thử với 5 epochs
# 54s 38ms/step - loss: 1.5127 - accuracy: 0.4666 - val_loss: 1.4447 - val_accuracy: 0.4856
# val_accuracy: 0.48559999465942383

# Thử với 25 epochs
# 52s 37ms/step - loss: 1.0312 - accuracy: 0.6365 - val_loss: 1.3234 - val_accuracy: 0.5510
# val_accuracy: 0.5509999990463257