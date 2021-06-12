'''
Main reference: Chapter 13 in (Géron, 2019)
Last review: March 2021
'''

# In[0]: IMPORTS AND SETTINGS
#region
import sys
from numpy.core import shape_base
from tensorflow._api.v2 import data
from tensorflow.keras import callbacks
from wrapt.wrappers import patch_function_wrapper
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn ≥0.20 is required
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0" # TensorFlow ≥2.0 is required
import numpy as np
import os
np.random.seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import pandas as pd
#endregion


''' WEEK 06 '''

# Tập dữ liệu được sử dụng là Netflix Original Films & IMDB Scores
# Source: https://www.kaggle.com/luiscorter/netflix-original-films-imdb-scores
# Dữ liệu dạng CSV
# Bao gồm các bộ phim gốc của Netflix,
# gồm các điểm IMDB được chọn bởi cách thành viên cộng đồng
# và phần lớn các phim đều có hơn 1000 lượt đánh giá
# Dữ liệu bao gồm
# Title of the film
# Genre of the film
# Original premiere date
# Runtime in minutes
# IMDB scores (tính đến thời điểm hiện tại)
# Languages currently available (tính đến thời điểm hiện tại)
# Trong bài tập này em chỉ lấy Genre, Runtime, IMDB và Language
# Dữ liệu đã được chỉnh sửa mốt ít cho phù hợp.


dataset = pd.read_csv('netflixoriginals.csv', encoding='utf-8')
X_full = dataset.iloc[:,:-1].values
y_full = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train_full, X_test, y_train_full, y_test = train_test_split(X_full, y_full, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2)

#Scale dữ liệu lại
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train[:,1:])
X_train[:,1:] = scaler.transform(X_train[:,1:])# lay tu cot thu 2 den cot thu 3
X_valid[:,1:] = scaler.transform(X_valid[:,1:])
X_test[:,1:] = scaler.transform(X_test[:,1:])


#Đổi chữ thành số
#Lấy ra các loại genre
vocab = np.unique(X_train[:,0])
#Dự phòng thêm 3 cái genre khác
num_oov_buckets = 3

# Đổi genre thành id
table_init = tf.lookup.KeyValueTensorInitializer(
    keys=tf.constant(vocab), values=tf.range(len(vocab), dtype=tf.int64))
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)


#Tạo ra một NN
#Tạo ra một embedding layer
#Tạo ra một vecto = 2, cứ mỗi genre thì đặt là 2 con số
embed_vec_dim = 2
#Gọi embedding layer của keras, input là số chiều đó là độ dài của vocab + số dự phòng (3)
embed_layer = keras.layers.Embedding(
    input_dim=len(vocab) + num_oov_buckets, # imagine the input is an one-hot vec
    output_dim=embed_vec_dim, name='embed_layer')
print('\nInitial embedding vectors:')
init_embeds = []
for i in range(len(vocab)+num_oov_buckets):
    init_embeds.append(embed_layer(i).numpy())
    if i<len(vocab):
        #i là cái id nãy tạo
        print('Init embed of',vocab[i],':',init_embeds[i])
    else:
        print('Init embed',i,'(oov):',init_embeds[i])
#Sau khi khởi tạo xong thì vẽ ra
init_embeds = np.array(init_embeds)
plt.plot(init_embeds[:len(vocab),0],init_embeds[:len(vocab),1], marker='.', color='r', markersize=15, linestyle='None')
for i in range(len(vocab)):
    plt.text(init_embeds[i,0]+.002,init_embeds[i,1]+.002,vocab[i],fontsize=11)
plt.title('Initial embeddings of Genre')
plt.axis([-.05, .05, -.05, .07])
plt.show()

#Sau khi khởi tạo embedding layer thì tạo ra một NN với nó
#Khởi tạo giá trị đầu vào là NA_Sales	EU_Sales	JP_Sales	Other_Sales	Global_Sales
num_inputs = keras.layers.Input(shape=[0], dtype=tf.float32)
#Genre
cat_input = keras.layers.Input(shape=[], dtype=tf.string)
#Đưa vào 1 cái lamda tự động biến chữ thành số
cat_indices = keras.layers.Lambda(lambda cats: table.lookup(cats))(cat_input)
#Đưa vô embedding của mình
cat_embed = embed_layer(cat_indices)
#Ghét 2 lớp lại với nhau (num_inputs+cat_embed)
concaten = keras.layers.concatenate([num_inputs, cat_embed])
#Tạo ra 1 model
#Sử dụng 3 lớp hidden layer mỗi lớp 10 neurons và activation là elu
hidden1 = keras.layers.Dense(10,activation='elu',kernel_initializer='he_normal')(concaten)
hidden2 = keras.layers.Dense(20,activation='elu',kernel_initializer='he_normal')(hidden1)
hidden3 = keras.layers.Dense(30,activation='elu',kernel_initializer='he_normal')(hidden2)
output = keras.layers.Dense(1)(hidden3)
model = keras.models.Model(inputs=[num_inputs, cat_input], outputs=[output])
model.save('models/video_game_report.h5')
model.summary()
#Total params: 341
#Bắt đầu trainning
#Tìm good learning rate
def qFindLearningRate(model, X_train, y_train, increase_factor = 1.005, batch_size=32, fig_name='find_lr'):
    # Create a callback to increase the learning rate after each batch, store losses to plot later
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

    # Train 1 epoch
    history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, callbacks=[increase_lr])

    # Plot losses after training batches.
    # NOTE: a batch has a different learning rate
    from statistics import median
    plt.plot(increase_lr.rates, increase_lr.losses)
    plt.gca().set_xscale('log')
    #plt.hlines(min(increase_lr.losses), min(increase_lr.rates), max(increase_lr.rates))
    plt.axis([min(increase_lr.rates), max(increase_lr.rates), min(increase_lr.losses), median(increase_lr.losses)])
    plt.grid()
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.savefig(fig_name, dpi=300)
    plt.show()
    return increase_lr
batch_size = 128
if 1:
    model = keras.models.load_model('models/video_game_report.h5')
    init_lr = 1e-5
    #Sử dụng Nadam
    model.compile(optimizer=keras.optimizers.Nadam(lr=init_lr), loss='mae', metrics='accuracy')
    increase_lr = qFindLearningRate(model, (X_train[:,1:1].astype('float32'),X_train[:,0]), y_train, fig_name='find_lr', batch_size=batch_size)
# => Good learning rate:
#good_lr = 0.2

#Sau khi test thử thì chọn good_lr = 0.02
class OneCycleScheduler(keras.callbacks.Callback):
    # Source: (Géron, 2019)
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
#Sử dụng 1Cycle là tiến hành trainning
new_training = 1
if new_training:
    data_train = (X_train[:,1:1].astype('float32'),X_train[:,0])
    data_valid = (X_valid[:,1:1].astype('float32'),X_valid[:,0])
    batch_size = 64
    good_lr = 0.03

    n_epochs = 80
    init_lr = good_lr/10
    model = keras.models.load_model('models/video_game_report.h5')
    model.compile(optimizer=keras.optimizers.Nadam(lr=init_lr), loss='mae') # metrics='mae'
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=n_epochs/2)
    save_checkpoint = keras.callbacks.ModelCheckpoint('models/video_game_report_1cycle.h5', save_best_only=True)
    #log_dir = "logs/CIFAR10_qCUSTOM_1cycle_" + datetime.datetime.now().strftime("%m%d-%H%M")
    #tensor_board = keras.callbacks.TensorBoard(log_dir=log_dir)

    n_iters = int(len(X_train) / batch_size) * n_epochs
    onecycle = OneCycleScheduler(n_iters, max_rate=good_lr)

    #with tf.device('/gpu:0'): # to use GPU (default of Keras is GPU if any)
    history = model.fit(data_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(data_valid,y_valid),
        callbacks=[early_stop, save_checkpoint, onecycle])
    history = history.history
model = keras.models.load_model('models/video_game_report_1cycle.h5')
model.evaluate(data_valid, y_valid)

#loss: 0.004723931197077036

#Xem xét embedding
#Lấy tham số của layer
embed_layer = model.get_layer('embed_layer')
embed_vecs = embed_layer.get_weights()
embed_vecs = np.array(embed_vecs[0])
init_embeds = np.array(init_embeds)
#Sau đó vẽ ra để xem xét
plt.plot(init_embeds[:len(vocab),0],init_embeds[:len(vocab),1], marker='.', color='r', markersize=15, linestyle='None')
plt.plot(embed_vecs[:len(vocab),0],embed_vecs[:len(vocab),1], marker='s', color='b', markersize=8, linestyle='None')
for i in range(len(vocab)):
    plt.text(init_embeds[i,0]+.02,init_embeds[i,1]+.02,vocab[i],fontsize=7)
    plt.text(embed_vecs[i,0]+.02,embed_vecs[i,1]+.02,vocab[i],fontsize=7)
plt.text(.3,.48,'unknow',fontsize=15,alpha=.4,fontstyle='italic')
plt.text(.8,1.05,'unknow',fontsize=15,alpha=.4,fontstyle='italic')
plt.text(1.55,1.48,'Unknow',fontsize=15,alpha=.4,fontstyle='italic')
plt.legend(['Init embed','Learned embed'])
plt.title('Embeddings of Vị trí')
plt.show()