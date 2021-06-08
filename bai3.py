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


new_training = 1
if new_training:
    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
    for n_hidden in (500, 400, 300, 200, 100, 50):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(n_hidden, activation="elu", kernel_initializer="he_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()
    tf.keras.utils.plot_model(model,'models/model_A_bai3.png',show_shapes=True)

    model.compile(optimizer=keras.optimizers.Nadam(lr=2e-3), loss='sparse_categorical_crossentropy', metrics='accuracy')
    history = model.fit(X_train, y_train, epochs=25,
                        validation_data=(X_valid, y_valid))
    history = history.history
    joblib.dump(history,"models/model_A_bai3_history")
    model.save("models/model_A_bai3.h5")
else:
    model = keras.models.load_model("models/model_A_bai3.h5")
    history = joblib.load("models/model_A_bai3_history")

# - Một nửa số epochs đầu tiên, liên tục tăng tuyến tính learning rate.
# - Một nửa của một nửa số epochs còn lại, liên tục giảm tuyến tính learning rate.
# - Một số epochs cuối cùng, giảm learning rate một cách đáng kể để cải thiện performance.


# Tìm good learning rate
def qFindLearningRate(model, X_train, y_train, increase_factor = 1.005, batch_size=32, fig_name='find_lr'):
    # Cài đặt callback để tăng learning rate
    class IncreaseLearningRate_cb(keras.callbacks.Callback):
        def __init__(self, factor):
            self.factor = factor
            self.rates = []
            self.losses = []
        def on_batch_end(self, batch, logs):
            K = keras.backend
            self.rates.append(K.get_value(self.model.optimizer.lr))
            self.losses.append(logs["loss"])
            K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)
    increase_lr = IncreaseLearningRate_cb(factor=increase_factor)

    history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, callbacks=[increase_lr])


    from statistics import median
    plt.plot(increase_lr.rates, increase_lr.losses)
    plt.gca().set_xscale('log')
    plt.axis([min(increase_lr.rates), max(increase_lr.rates), min(increase_lr.losses), median(increase_lr.losses)])
    plt.grid()
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.savefig(fig_name, dpi=300)
    plt.show()
    return increase_lr

batch_size = 128

if 1:
    model = keras.models.load_model('models/model_A_bai3.h5')
    model.compile(optimizer=keras.optimizers.Nadam(lr=2e-3), loss='sparse_categorical_crossentropy', metrics='accuracy')
    increase_lr = qFindLearningRate(model, X_train, y_train, fig_name='find_lr', batch_size=batch_size)
good_lr = 0.00202


# Reference: Book - Géron 2019
class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None, last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    def on_batch_begin(self, batch, logs):
        K = keras.backend
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)
new_training = 1
if new_training:
    batch_size = 128
    init_lr = good_lr/10
    model = keras.models.load_model('models/model_A_bai3.h5')
    model.compile(optimizer=keras.optimizers.Nadam(lr=init_lr), loss='sparse_categorical_crossentropy', metrics='accuracy')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15)
    save_checkpoint = keras.callbacks.ModelCheckpoint('models/best_FashionM_1cycle.h5', save_best_only=True)
    log_dir = "logs/CIFAR10_FashionM_1cycle_" + datetime.datetime.now().strftime("%m%d-%H%M")
    tensor_board = keras.callbacks.TensorBoard(log_dir=log_dir)

    n_epochs = 20
    n_iters = int(len(X_train) / batch_size) * n_epochs
    onecycle = OneCycleScheduler(n_iters, max_rate=good_lr)

    #with tf.device('/gpu:0'): # to use GPU (default of Keras is GPU if any)
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_valid,y_valid),
        callbacks=[early_stop, save_checkpoint, tensor_board, onecycle])
    history = history.history
    #joblib.dump(history, 'models/best_FashionM_1cycle/history')
model = keras.models.load_model('models/best_FashionM_1cycle.h5')
#history = joblib.load('models/best_FashionM_1cycle/history')
print("_____+_____")
print(model.evaluate(X_valid, y_valid))
#endregion

# Thử với 5 epochs
# [1.2210606336593628, 0.5687999725341797]

# Thử với 25 epochs
# [1.544638752937317, 0.5812000036239624]