'''
Main reference: Chapter 15 in (Géron, 2019)
Last review: May 2021
'''

# In[0]: IMPORTS AND SETTINGS
#region
import sys
from tensorflow import keras
from tensorflow.core.protobuf.cluster_pb2 import JobDef
from tensorflow.keras import callbacks, layers
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.pooling import AvgPool1D, AvgPool2D
from tensorflow.python.ops.gen_array_ops import shape
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
import joblib

use_GPU = 0
if use_GPU:
	physical_devices = tf.config.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

#endregion

# In[1]: RNNs FOR TIME SERIES: PREDICTING ONE NEXT VALUE
#region
# 1.1. Generate the dataset
# NOTE:
# 	+ Dataset of multivariate time series has shape:
# 	  [no_of_series, no_of_time_steps, no_of_values]
# 	+ Dataset of univariate time series has shape:
# 	  [no_of_series, no_of_time_steps, 1]
def generate_time_series(no_of_series, no_of_time_steps):
	''' Generate univarate time series.
	Returns np array of shape: [no_of_series, no_of_time_steps, 1]
	'''
	freq1, freq2, offsets1, offsets2 = np.random.rand(4, no_of_series, 1)
	time = np.linspace(0, 1, no_of_time_steps)
	series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
	series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
	series += 0.1 * (np.random.rand(no_of_series, no_of_time_steps) - 0.5)   # + noise
	return series[..., np.newaxis].astype(np.float32)

np.random.seed(42)
n_steps = 50
n_future_steps = 1
series = generate_time_series(10000, n_steps + n_future_steps)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

# 1.2. Plot time series
def plot_series(series, y=None, y_pred=None, n_steps=50, n_future_steps=1, x_label="$t$", y_label="$x(t)$", marker='o', color='blue'):
    legends = []
    if y is not None:
    	plt.plot(n_steps+n_future_steps-1, y, "go", markersize=8)
    	legends.append('Future value')
    if y_pred is not None:
    	plt.plot(n_steps+n_future_steps-1, y_pred, "rx", markersize=8, markeredgewidth=3)
    	legends.append('Predicted value')
    plt.grid(True)

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.plot(series, color=color, marker=marker, linestyle='-') # plot series
    plt.legend(legends)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps+n_future_steps+1, -1, 1])

n_series_to_plot = 3
for i in range(n_series_to_plot):
	plot_series(series=X_valid[i, :, 0], y=y_valid[i, 0], n_steps=50)
	plt.title('Series '+str(i))
	plt.show()


# Thử với LSTM
np.random.seed(42)
tf.random.set_seed(42)
model =keras.models.Sequential  ([
   keras.layers.Input(shape=[None,1]),
   keras.layers.LSTM(30,return_sequences=True),
  keras.layers.LSTM(30,return_sequences=True),
  keras.layers.LSTM(30,return_sequences=True),
  keras.layers.LSTM(2,return_sequences=False),
  keras.layers.Dense(1)
])
model.summary()

# return_sequences: Boolean.
# Whether to return the last output.
# in the output sequence, or the full sequence.
# Default: False.

# LSTM will eat the words of your sentence one by one,
# you can chose via "return_sequence" to outuput something (the state)
# at each step (after each word processed) or only output something after
# the last word has been eaten. So with return_sequence=TRUE,
# the output will be a sequence of the same length, with return_sequence=FALSE,
# the output will be just one vector.

new_training = 0
if new_training:
	model.compile(loss="mse", optimizer="nadam")
	# NOTE: deeper net may require longer training.
	history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
	model.save(r'models/deepRNN01.h5')
model = keras.models.load_model(r'models/deepRNN01.h5')

# Evaluate the model:
mse_deepRNN_LSTM = model.evaluate(X_valid, y_valid)
y_pred = model.predict(X_valid)
n_series_to_plot = 3
for i in range(n_series_to_plot):
	plot_series(series=X_valid[i, :, 0], y=y_valid[i, 0], y_pred=y_pred[i], n_steps=50)
	plt.title('Series '+str(i))
	plt.show()
print('\nMSE of LSTM:', np.round(mse_deepRNN_LSTM,4))
# MSE of LSTM: 0.0016
# Total params: 18,747

"""
Part 2
"""
# Thử với GRU
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential  ([
   keras.layers.Input(shape=[None,1]),
   keras.layers.GRU(30,return_sequences=True),
  keras.layers.GRU(30,return_sequences=True),
  keras.layers.GRU(30,return_sequences=True),
  keras.layers.GRU(2,return_sequences=False),
  keras.layers.Dense(1)
])
model.summary()

new_training = 0
if new_training:
	model.compile(loss="mse", optimizer="nadam")
	# NOTE: deeper net may require longer training.
	history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
	model.save(r'models/deepRNN02.h5')
model = keras.models.load_model(r'models/deepRNN02.h5')

mse_deepRNN_GRU = model.evaluate(X_valid, y_valid)
y_pred = model.predict(X_valid)
n_series_to_plot = 3
for i in range(n_series_to_plot):
	plot_series(series=X_valid[i, :, 0], y=y_valid[i, 0], y_pred=y_pred[i], n_steps=50)
	plt.title('Series '+str(i))
	plt.show()
print('\nMSE of GRU:', np.round(mse_deepRNN_GRU,4))
# MSE of GRU: 0.0025
# Total params: 14,337
