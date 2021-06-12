'''
Main reference: (Géron, 2019)
Last review: March 2021
'''

# In[0]: IMPORTS AND SETTINGS
#region
import datetime
import sys
from zzz import X_test, X_train_full
from numpy.core import shape_base
from tensorflow._api.v2 import data
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn ≥0.20 is required
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras
assert tf.__version__ >= "2.0" # TensorFlow ≥2.0 is required
import numpy as np
import os
np.random.seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
#endregion


''' WEEK 06 '''

#Load dataset Boston housing
# Bộ dữ liệu Boston housing gồm giá nhà ở những nơi khác nhau ở Boston
# Cùng với giá cả, tập dữ liệu cũng cung cấp những thông tin như tội phạm (CRIM)
# Các khu vực kinh doanh không bán lẻ ở thị trấn (INDUS), tuổi của chủ sở hữu nhà (AGE)
# Và các thuộc tính khác
# Scikit learn có sẵn tập dữ liêu này nên ta có thể lấy về như sau

(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

# StandardScaler dùng để scale data features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

# DÙng train_test_split để chia tập train ra thành train và valid
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(X_train.shape[1],)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(4, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1))
model.summary()
tf.keras.utils.plot_model(model,'models/model_A.png',show_shapes=True)


new_training = 1
if new_training:
    model.compile(optimizer=keras.optimizers.Nadam(lr=2e-3), loss='mean_absolute_error', metrics='mae')
    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_valid, y_valid))
    history = history.history
    joblib.dump(history,"models/model_A_history")
    model.save("models/model_A.h5")
else:
    model = keras.models.load_model("models/model_A.h5")
    history = joblib.load("models/model_A_history")


def qFindLearningRate(model, X_train, y_train, increase_factor = 1.5, batch_size=32, fig_name='find_lr'):
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
batch_size = 150

if 1:
    model = keras.models.load_model('models/model_A.h5')
    #Sửa lại lr = 0.02 sau khi thử lại nhiều lần
    model.compile(optimizer=keras.optimizers.Nadam(lr=0.02), loss='mean_absolute_error', metrics='mae')
    increase_lr = qFindLearningRate(model, X_train, y_train, fig_name='find_lr', batch_size=batch_size)
# Sau khi đã xem xét và thử lại nhiều lần em sẽ chọn good learning rate = 0.15
good_lr = 0.15

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
    batch_size = 150
    init_lr = good_lr/10
    model = keras.models.load_model('models/model_A.h5')
    model.compile(optimizer=keras.optimizers.Nadam(lr=init_lr), loss='mean_absolute_error', metrics='mae')
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=15)
    save_checkpoint = keras.callbacks.ModelCheckpoint('models/best_boston_housing_1cycle.h5', save_best_only=True)
    log_dir = "logs/boston_housing_1cycle_" + datetime.datetime.now().strftime("%m%d-%H%M")
    tensor_board = keras.callbacks.TensorBoard(log_dir=log_dir)
    # Sử dụng 100 epoches
    n_epochs = 100
    n_iters = int(len(X_train) / batch_size) * n_epochs
    onecycle = OneCycleScheduler(n_iters, max_rate=good_lr)

    # with tf.device('/gpu:0'): # to use GPU (default of Keras is GPU if any)
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_valid,y_valid),
        callbacks=[early_stop, save_checkpoint, tensor_board, onecycle])
    history = history.history
model = keras.models.load_model('models/best_boston_housing_1cycle.h5')
print('____valid____')
print(model.evaluate(X_valid, y_valid))
print('____test____')
print(model.evaluate(X_test, y_test))

# ____valid____
# 4/4 [==============================] - 0s 2ms/step - loss: 2.2481 - mae: 2.2481
# [2.2481162548065186, 2.2481162548065186]
# ____test____
# 4/4 [==============================] - 0s 1ms/step - loss: 2.5910 - mae: 2.5910
# [2.5909972190856934, 2.5909972190856934]
