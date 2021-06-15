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

#Mở tập tin đọc dữ liệu của truyen kiều
# f = open(r"/content/Week13/MyDrive/Week13/code/datasets/TruyenKieu_NguyenDu_vnthuquan.txt", "r", encoding="utf-8")
# truyenKieu = f.read()
# f.close()
# print('\nTRUYEN KIEU:\n',truyenKieu[:334])
# all_char = "".join(sorted(set(truyenKieu.lower())))
# print('\nText length:', len(truyenKieu), 'characters.')
# print('All characters used in the text:', all_char, '(%d char)' % len(all_char))
# words_set_truyenKieu = set(truyenKieu.split())
# print('Number of words in the text:', len(words_set_truyenKieu), ' words')
# text_data = truyenKieu
# words_set = words_set_truyenKieu

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
dataset = dataset.repeat().window(window_length, shift=1, drop_remainder=True) # drop_remainder=True: remove the last window to ensure all windows have the same length. See https://www.tensorflow.org/api_docs/python/tf/data/Dataset#window
def char_ids_to_text(seq_of_char_ids):
    '''Input: seq of char ids: [[id1, id2, id3...]]'''
    seq_of_char = tokenizer.sequences_to_texts(seq_of_char_ids)
    text = ''.join(seq_of_char) # text='t r ă m   n ă m   t r o n g'
    # NOTE: tokenizer.sequences_to_texts() auto add space when char_level=True

    # Remove added spaces:
    text = text.replace('   ','~') # Convert real spaces to @'s to reserve them
    text = text.replace(' ','') # Remove spaces
    text = text.replace('~',' ') # Restore real spaces
    return text
print("Some windows:"); i=0
for window_ds in dataset.take(3):
    print('\nWINDOW %d:' %i); i+=1
    seq_of_char_ids = [[char_id.numpy()+1 for char_id in window_ds]]
    text = char_ids_to_text(seq_of_char_ids)
    print(text)

# 1.1.4. Flatten dataset
# ds.flat_map(lambda ds: ds.batch(3)) does 2 things:
#   1. Flatten: {{1, 2, 3}, {2, 3, 4}, {5, 6}} (dataset of datasets) => {1, 2, 3, 2, 3, 4, 5, 6} (dataset of tensors)
#   2. Map: {1, 2, 3, 2, 3, 4, 5, 6} => {[1, 2, 3], [2, 3, 4], [5, 6]} (dataset of sequences (vectors))
dataset = dataset.flat_map(lambda ds: ds.batch(window_length))

print("\n\nSome windows:"); i=0
for window_seq in dataset.take(3):
    print('\nWINDOW %d:' %i); i+=1
    seq_of_char_ids = [window_seq.numpy()+1]
    text = char_ids_to_text(seq_of_char_ids)
    print(text)

# 1.1.5. Shuffle and batch
batch_size = 32
np.random.seed(42)
tf.random.set_seed(42)
dataset = dataset.shuffle(10000) #Since Gradient Descent works best when the instances in the training set are independent and identically distributed
dataset = dataset.batch(batch_size)

# 1.1.6. Split input (t=0 => t=n_steps) and label (t=1 => t=n_steps+1)
dataset = dataset.map(lambda ds: (ds[:, :-1], ds[:, 1:]))

# 1.1.7. (MAY SKIP) Encode with one-hot encoding
use_one_hot_encoding = 0
if use_one_hot_encoding:
    dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
else: # add new axis. Recall: data shape (n_series, n_steps, n_values)
    dataset = dataset.map(lambda X_batch, Y_batch: (X_batch[..., np.newaxis] , Y_batch))

# 1.1.8. Prefetch
dataset = dataset.prefetch(1)
for X_batch, Y_batch in dataset.take(1):
    print('\n\nX shape: {}. Y shape: {}'.format(X_batch.shape, Y_batch.shape))


model = keras.models.Sequential()
if use_one_hot_encoding: # different in input shape
    model.add(keras.layers.Input(shape=[None, max_id]))
else:
    model.add(keras.layers.Input(shape=[None, 1]))
keras.layers.Conv1D(filters=8,kernel_size=4,strides=2,padding="valid"),
model.add(keras.layers.GRU(128, return_sequences=True)) # dropout=0.2, recurrent_dropout=0.2),
model.add(keras.layers.GRU(128, return_sequences=True)) # dropout=0.2, recurrent_dropout=0.2),
model.add(keras.layers.Dense(max_id, activation="softmax"))
model.summary()
new_training = 1
if new_training:
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam")
    history = model.fit(dataset, steps_per_epoch=train_size // batch_size, epochs=3)
    model.save(r'models/truyenKieu_GRU_RNN.h5')
model = keras.models.load_model(r'models/truyenKieu_GRU_RNN.h5')
