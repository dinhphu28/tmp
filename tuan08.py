'''
Main reference: Chapter 14 in (Géron, 2019)
Last review: April 2021
'''

# In[0]: IMPORTS AND SETTINGS
#region
import sys
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.python.keras.layers.core import Flatten
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn ≥0.20 is required
import tensorflow as tf
assert tf.__version__ >= "2.0" # TensorFlow ≥2.0 is required
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(42)
tf.random.set_seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
#endregion


''' WEEK 8 '''

# Load and preprocess CIFAR-10
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
# plt.imshow(X_train[0], cmap='gray')
# Scale images:
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std
# Vì ảnh màu nên ta ko cần thêm newaxis

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten
from functools import partial

from tensorflow.keras.layers import Conv2D, MaxPool2D
myDense = partial(Dense, activation='elu',kernel_initializer='he_normal')
myConv2D = partial(Conv2D, kernel_size=(3,3), strides=(1,1), padding='SAME', activation='elu', kernel_initializer='he_normal') # strides=1 b/c of small images
CNN_model = keras.Sequential([
    Input(shape=X_train.shape[1:]),
    myConv2D(filters=64, kernel_size=(3,3), strides=(1,1)),
    MaxPool2D(pool_size=(2,2)),
    BatchNormalization(),
    Dropout(0.1),

    myConv2D(filters=128),
    myConv2D(filters=128),
    MaxPool2D(pool_size=(2,2)),
    BatchNormalization(),
    Dropout(0.3),

    myConv2D(filters=256),
    myConv2D(filters=256),
    MaxPool2D(pool_size=(2,2)),
    BatchNormalization(),
    Dropout(0.4),

    Flatten(),
    myDense(units=100),
    Dropout(0.5),
    myDense(units=10, activation='softmax') ])
CNN_model.summary()
new_training = 0
if new_training:
    CNN_model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    model_saver = keras.callbacks.ModelCheckpoint('models/best_CNN_model_noBN.h5',monitor='val_accuracy', save_best_only=True)
    early_stopper = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    performance_sched = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2)
    #with tf.device('/gpu:0'):
    CNN_model.fit(X_train, y_train, epochs=5, batch_size=32,
        validation_data=(X_valid, y_valid),
        callbacks=[model_saver, early_stopper, performance_sched])
CNN_model = keras.models.load_model('models/best_CNN_model_noBN.h5')
CNN_model.evaluate(X_test, y_test)
# 1,520,982 params
# loss: 0.6994
# accuracy: 0.7624

print('_______________+++')
myDense = partial(Dense, activation='elu',kernel_initializer='he_normal')
myConv2D = partial(Conv2D, kernel_size=(3,3), strides=(2,2), padding='SAME', activation='elu', kernel_initializer='he_normal') # strides=1 b/c of small images
CNN_model = keras.Sequential([
    Input(shape=X_train.shape[1:]),
    myConv2D(filters=64, kernel_size=(3,3), strides=(2,2)),
    MaxPool2D(pool_size=(2,2)),
    BatchNormalization(),
    Dropout(0.2),

    myConv2D(filters=128),
    myConv2D(filters=128),
    MaxPool2D(pool_size=(2,2)),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    myDense(units=100),
    Dropout(0.5),
    myDense(units=10, activation='softmax') ])
CNN_model.summary()
new_training = 1
if new_training:
    CNN_model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    model_saver = keras.callbacks.ModelCheckpoint('models/best_CNN_model_noBN_p2.h5',monitor='val_accuracy', save_best_only=True)
    early_stopper = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    performance_sched = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2)
    #with tf.device('/gpu:0'):
    CNN_model.fit(X_train, y_train, epochs=5, batch_size=32,
        validation_data=(X_valid, y_valid),
        callbacks=[model_saver, early_stopper, performance_sched])
CNN_model = keras.models.load_model('models/best_CNN_model_noBN_p2.h5')
CNN_model.evaluate(X_test, y_test)
# 237,910 params
# loss: 0.9250
# accuracy: 0.6734