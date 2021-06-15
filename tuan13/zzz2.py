'''
Main reference: Chapter 16 in (Géron, 2019)
Last review: May 2021
'''

# In[0]: IMPORTS AND FUNCTIONS
#region
import sys
from tensorflow import keras
from tensorflow.core.protobuf.cluster_pb2 import JobDef
from tensorflow.keras import callbacks, layers
from tensorflow.python.keras.backend import conv1d
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


''' WEEK 13 '''


f = open(r"datasets/TruyenKieu_NguyenDu_vnthuquan.txt", "r", encoding="utf-8")
truyenKieu = f.read()
f.close()
print('\nTRUYEN KIEU:\n',truyenKieu[:334])
all_char = "".join(sorted(set(truyenKieu.lower())))
print('\nText length:', len(truyenKieu), 'characters.')
print('All characters used in the text:', all_char, '(%d char)' % len(all_char))
words_set_truyenKieu = set(truyenKieu.split())
print('Number of words in the text:', len(words_set_truyenKieu), ' words')

#text_data = shakespeare_text
text_data = truyenKieu
words_set = words_set_truyenKieu


#%% 1.1. PREPROCESS DATA
# 1.1.1. Encode each char as an integer
load_tokenizer = 0
if load_tokenizer:
    tokenizer = joblib.load(r'models/tokenizer_truyenKieu.joblib')
else:
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(text_data)
    joblib.dump(tokenizer,r'models/tokenizer_truyenKieu.joblib')
max_id = len(tokenizer.word_index) # number of distinct characters
dataset_size = tokenizer.document_count # total number of characters
print('\nText length:', dataset_size, 'characters.')
print('Total number of characters:', max_id)
samples = ['Một ', 'Trăm năm']
encoded = tokenizer.texts_to_sequences(samples)
print('\nEncoding of', samples, ':\n', encoded)
samples = [[1, 5, 12, 4, 6, 80], [15, 5, 6, 26, 48, 90]]
decoded = tokenizer.sequences_to_texts(samples)
print('\nDecoding of', samples, ':\n', decoded)
# Encode whole data:
[encoded_whole_text] = np.array(tokenizer.texts_to_sequences([text_data])) - 1 # NOTE: Tokenizer assigns id starting from 1 (NOT 0), hence -1 to get id from 0.
print('\nText:',text_data[:20])
print('Encoding:',encoded_whole_text[:20])


# 1.1.2. Convert to tf dataset format
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded_whole_text[:train_size])

# 1.1.3. Chop text into small sequences (using window() function)
# IMPORTANT NOTE: NEED to tune n_steps (#char in a sequence).
# Hint: RNN can ONLY learn patterns SHORTER than n_steps.
n_steps = 100
n_predicts = 1
window_length = n_steps + n_predicts
# Để xây dựng Stateful RNN thì ta cần input dạng sequence và không chồng lấn
# Do đó ta cần đặt shift=n_steps
dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
# Làm phẳng và phân chia batching cho Stateful RNN, giúp các windows liên tiếp nhau
dataset = dataset.flat_map(lambda ds: ds.batch(window_length))
dataset = dataset.batch(1)
dataset = dataset.map(lambda ds: (ds[:, :-1], ds[:, 1:]))
dataset = dataset.map(lambda X_batch, Y_batch: (X_batch[..., np.newaxis] , Y_batch))
dataset = dataset.prefetch(1)

batch_size = 32
#Statefull RNNs
model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=[None, 1], batch_size=1),#Xác định kiểu dữ liệu của input
        keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2, recurrent_dropout=0.2, batch_input_shape=[batch_size, None, max_id]), #SỬ dụng recrrent_dropot=0.2
        keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2, recurrent_dropout=0.2), #stateful=true khi tạo 1 Statefull rnn, cần tạo batch_input_shape=duy trì trạng thái của đầu vào
        keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax")) ])#Tương tự như lớp dense theo em hiểu.
model.summary()

# Dùng Callback để reset lại state
class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()

new_training = 1
if new_training:
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    model.fit(dataset, epochs=5, callbacks=[ResetStatesCallback()])
    model.save(r'models/TruyenKieu_RNN_Stateful.h5')
model=keras.models.load_model(r'models/TruyenKieu_RNN_Stateful.h5')
