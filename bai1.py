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

# Thử với learning rate cố định
# Training với 25 epochs
# Learning rate = 0.02 (cố định)
new_training = 1
if new_training:
    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
    for ii in range(15):
        noNeurons = 100
        model.add(keras.layers.Dense(noNeurons, activation="elu", kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()
    tf.keras.utils.plot_model(model,'models/model_A_bai1_lan1.png',show_shapes=True)

    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer=keras.optimizers.SGD(lr=0.02, momentum=0.0, decay=0.0, nesterov=False),
                    metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=20,
                        validation_data=(X_valid, y_valid))
    history = history.history
    joblib.dump(history,"models/model_A_bai1_lan1_history")
    model.save("models/model_A_bai1_lan1.h5")
else:
    model = keras.models.load_model("models/model_A_bai1_lan1.h5")
    history = joblib.load("models/model_A_bai1_lan1_history")

print("_____Lan 1:_____")
print(history["val_accuracy"][-1])

# 17s 12ms/step - loss: 1.2025 - accuracy: 0.5676 - val_loss: 1.4779 - val_accuracy: 0.4876
# 0.4875999987125397

#Thử với Exponential Scheduling
# Khởi tạo giá trị learning_rate ban đầu rồi nhân cho d^i/s
# Với s là step và d (decay) là một số nhỏ hơn 1
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

# Activation function: selu
# kernel_initializer: lecun_normal
# Optimizer: nadam
# Training với 25 epochs

new_training = 1
if new_training:
    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
    for ii in range(15):
        noNeurons = 100
        model.add(keras.layers.Dense(noNeurons, activation="selu", kernel_initializer="lecun_normal"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()
    tf.keras.utils.plot_model(model,'models/model_A_bai1_lan2.png',show_shapes=True)

    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer="nadam",
                    metrics=["accuracy"])
    lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
    history = model.fit(X_train, y_train, epochs=25,
                        validation_data=(X_valid, y_valid),
                        callbacks=[lr_scheduler])
    history = history.history
    joblib.dump(history,"models/model_A_bai1_lan2_history")
    model.save("models/model_A_bai1_lan2.h5")
else:
    model = keras.models.load_model("models/model_A_bai1_lan2.h5")
    history = joblib.load("models/model_A_bai1_lan2_history")

print("_____Lan 2:_____")
print(history["val_accuracy"][-1])
# 21s 15ms/step - loss: 2.3249 - accuracy: 0.0982 - val_loss: 2.3407 - val_accuracy: 0.0920
# 0.09200000017881393


# Thử với Performance Scheduling
# Phương pháp này sẽ chọn learning rate theo performance,
# Tức là nếu sau n step mà không có xu hướng hội tụ nó sẽ giảm learning rate
# Training với 25 epochs

new_training = 1
if new_training:
    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
    for ii in range(15):
        noNeurons = 100
        model.add(keras.layers.Dense(noNeurons, activation="selu", kernel_initializer="lecun_normal"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()
    tf.keras.utils.plot_model(model,'models/model_A_bai1_lan3.png',show_shapes=True)

    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer="nadam",
                    metrics=["accuracy"])
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    history = model.fit(X_train, y_train, epochs=25,
                        validation_data=(X_valid, y_valid),
                        callbacks=[lr_scheduler])
    history = history.history
    joblib.dump(history,"models/model_A_bai1_lan3_history")
    model.save("models/model_A_bai1_lan3.h5")
else:
    model = keras.models.load_model("models/model_A_bai1_lan3.h5")
    history = joblib.load("models/model_A_bai1_lan3_history")

print("_____Lan 3:_____")
print(history["val_accuracy"][-1])
# 19s 14ms/step - loss: 1.5691 - accuracy: 0.4297 - val_loss: 1.6009 - val_accuracy: 0.4080
# 0.40799999237060547