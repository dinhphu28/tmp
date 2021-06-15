# In[0]: IMPORTS AND SETTINGS
#region
import sys
from tensorflow import keras
from tensorflow.core.protobuf.cluster_pb2 import JobDef
from tensorflow.keras import callbacks
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.pooling import AvgPool1D, AvgPool2D
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



#================>>thử lần thứ nhất với inceptionV3<<=======
#Load model inceptionV3
model = tf.keras.applications.InceptionV3(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)

#load và resize lại images
#InceptionV3 có input (299, 299, 3)
import glob
file_names = glob.glob ("images/*.jpg")
images = np.empty((0,299,299,3))
for file_name in file_names:
    img_BGR = cv2.imread(file_name)
    img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    #img = tf.image.resize(img, (224,224), antialias=True)
    img = tf.image.resize_with_pad(img, 299,299, antialias=True) # NOTE: To resize but KEEP ASPECT RATIO: resize_with_pad() or crop_and_resize()
    images = np.append(images, img[np.newaxis,:,:,:], axis=0)
    if 10:
        plt.imshow(img/255)
        plt.title(file_name)
        plt.show()
print(images.shape)

#preprocess input đầu vào bằng lệnh preprocess_input
inputs = tf.keras.applications.inception_v3.preprocess_input(images)
#Dự đoán
#ImageNet có tới 1000 classes  nên chỉ  cần lấy 3 classes có dự đoán cao nhất
Y_proba = model.predict(inputs)
k=3
top_k_predictions = tf.keras.applications.inception_v3.decode_predictions(Y_proba, top=k)
for i in range(len(images)):
    if 10:
        plt.imshow(images[i]/80) # processed images do NOT have pixel values of 0-255
        plt.title(file_names[i])
        plt.show()
    print("File {}".format(file_names[i]))
    for class_id, name, y_proba in top_k_predictions[i]:
        #print("  {:12s} (id:{}): {:.1f}%".format(name,class_id,  y_proba * 100))
        print("   {}: {:.1f}%".format(name, y_proba*100))
    print()

#================>>thử lần thứ nhất với inceptionV3<<=======
#Load model inceptionV3
model = tf.keras.applications.InceptionV3(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)

#load và resize lại images
#InceptionV3 có input (299, 299, 3)
import glob
file_names = glob.glob ("images/*.jpg")
images = np.empty((0,299,299,3))
for file_name in file_names:
    img_BGR = cv2.imread(file_name)
    img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    #img = tf.image.resize(img, (224,224), antialias=True)
    img = tf.image.resize_with_pad(img, 299,299, antialias=True) # NOTE: To resize but KEEP ASPECT RATIO: resize_with_pad() or crop_and_resize()
    images = np.append(images, img[np.newaxis,:,:,:], axis=0)
    if 10:
        plt.imshow(img/255)
        plt.title(file_name)
        plt.show()
print(images.shape)

#preprocess input đầu vào bằng lệnh preprocess_input
inputs = tf.keras.applications.inception_v3.preprocess_input(images)
#Dự đoán
#ImageNet có tới 1000 classes  nên chỉ  cần lấy 3 classes có dự đoán cao nhất
Y_proba = model.predict(inputs)
k=3
top_k_predictions = tf.keras.applications.inception_v3.decode_predictions(Y_proba, top=k)
for i in range(len(images)):
    if 10:
        plt.imshow(images[i]/80) # processed images do NOT have pixel values of 0-255
        plt.title(file_names[i])
        plt.show()
    print("File {}".format(file_names[i]))
    for class_id, name, y_proba in top_k_predictions[i]:
        #print("  {:12s} (id:{}): {:.1f}%".format(name,class_id,  y_proba * 100))
        print("   {}: {:.1f}%".format(name, y_proba*100))
    print()

# #=============>>Thử lần 2 với EfficientNet<<===============
# #Load model EfficientNet
# model = tf.keras.applications.EfficientNetB7(
#     include_top=False, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000,
#     classifier_activation='softmax')
# #load và resize lại images
# #EfficientNetB7 có input shape là 600x600x3
# import glob
# file_names = glob.glob ("images/*.jpg")
# images = np.empty((0,600,600,3))
# for file_name in file_names:
#     img_BGR = cv2.imread(file_name)
#     img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
#     #img = tf.image.resize(img, (224,224), antialias=True)
#     img = tf.image.resize_with_pad(img, 600,600, antialias=True) # NOTE: To resize but KEEP ASPECT RATIO: resize_with_pad() or crop_and_resize()
#     images = np.append(images, img[np.newaxis,:,:,:], axis=0)
#     if 10:
#         plt.imshow(img/255)
#         plt.title(file_name)
#         plt.show()
# print(images.shape)
# #preprocess input đầu vào bằng lệnh preprocess_input
# inputs = tf.keras.applications.efficientnet.preprocess_input(images)
# #Dự đoán
# #ImageNet có tới 1000 classes  nên chỉ  cần lấy 3 classes có dự đoán cao nhất
# Y_proba = model.predict(inputs)
# k=3
# top_k_predictions = tf.keras.applications.efficientnet.decode_predictions(Y_proba, top=k)
# for i in range(len(images)):
#     if 10:
#         plt.imshow(images[i]/80) # processed images do NOT have pixel values of 0-255
#         plt.title(file_names[i])
#         plt.show()
#     print("File {}".format(file_names[i]))
#     for class_id, name, y_proba in top_k_predictions[i]:
#         #print("  {:12s} (id:{}): {:.1f}%".format(name,class_id,  y_proba * 100))
#         print("   {}: {:.1f}%".format(name, y_proba*100))
#     print()


# In[6]:  TRANSFER LEARNING WITH PRETRAINED MODELs from tf.keras.efficientnet
#region
# 6.1. Load data and split
import tensorflow_datasets as tfds
dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
# dataset, info = tfds.load("oxford_flowers102", as_supervised=True, with_info=True)
#data = tfds.load('oxford_flowers102') smallnorb
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples
test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
    as_supervised=True)
if 1:
    plt.figure(figsize=(12, 15))
    index = 0
    for image, label in train_set_raw.take(6):
        index += 1
        plt.subplot(3, 2, index)
        plt.imshow(image)
        plt.title("Class: {}".format(class_names[label]), fontsize=25)
        #plt.axis("off")

#%% 6.2. Augment and preprocess images
# NOTE: efficientnet-b3 takes 300x300 images. Source: https://kobiso.github.io/Computer-Vision-Leaderboard/imagenet.html
def central_crop(image):
    #image = cv2.imread('images\Meo_2.jpg')
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]])
    top_crop = (shape[0] - min_dim) // 4
    bottom_crop = shape[0] - top_crop
    left_crop = (shape[1] - min_dim) // 4
    right_crop = shape[1] - left_crop
    return image[top_crop:bottom_crop, left_crop:right_crop]

def random_crop(image, kept_percentage=90): # crop out 10% (default)
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]]) * kept_percentage // 100
    return tf.image.random_crop(image, [min_dim, min_dim, 3])

def preprocess(image, label, x_size=300, y_size=300, rand_crop=False):
    if rand_crop:
        cropped_image = random_crop(image)
        cropped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        cropped_image = central_crop(image)
    resized_image = tf.image.resize_with_pad(cropped_image, x_size, y_size)

    # Add other operations here: eg.,
    # tf.image.random_flip_left_right()  # flip images

    final_image = tf.keras.applications.efficientnet.preprocess_input(resized_image)
    return final_image, label

from functools import partial
batch_size = 32
train_set = train_set_raw.shuffle(buffer_size=1000).repeat()
# NOTE:
#   1. repeat(): infinitely duplicate the data. (repeat(2): duplicate 2 times). To see, try: dem=0; for i in train_set: dem+=1; print(dem)
#   2. Unfortunately, NO way to get shape of tf.dataset except using loop (as NOTE 1.)
train_set = train_set.map(partial(preprocess, rand_crop=True)).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.repeat().map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)
# NOTE: Each batch includes: 1. Images, 2. List of labels.
if 10: # Plot some samples
    for batch in train_set.take(1):
        images, labels = batch[0], batch[1]
        plt.figure(figsize=(15, 30))
        for i in range(8): #len(labels)
            plt.subplot(4, 2, i+1)
            plt.imshow(images[i]/255)
            plt.axis('off')
            plt.title('Label: '+class_names[labels[i]], fontsize=30)
        plt.show()

#%% 6.3. Load EfficientNet B7
# Load model:
base_model = keras.applications.efficientnet.EfficientNetB7(weights="imagenet",
                include_top=False) # NOT include the FC layers
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)
# Display model:
for index, layer in enumerate(model.layers):
    print(index, layer.name)
model.summary()


#%% 6.4. TRANSFER LEARNING
# NOTE: Recall: TRANSFER LEARNING STEPS
#   0. Load pretrained model (usually trained with ImageNet). Add FC layers (to fit your data labels).
#   1. Freeze base-model layers: to train only FC layers. After a few epochs, its validation accuracy stops making much progress => FC layers are now pretty well trained.
#   3. Unfreeze and train all layers (or just the top ones). NOTE: Use a much lower learning rate to avoid damaging the pretrained weights.

# STEP 1: Freeze the base_model layers: to train only the FC layers
# CAUTION: takes a while to finish! ==> Use GPU or try using Google Colab.
new_training = 0
if new_training:
    for layer in base_model.layers:
        layer.trainable = False

    optimizer = keras.optimizers.Nadam(lr=0.2)
    # optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    #with tf.device('/cpu'):
    history = model.fit(train_set,
                        steps_per_epoch=int(0.75*dataset_size / batch_size),
                        validation_data=valid_set,
                        validation_steps=int(0.15*dataset_size / batch_size),
                        epochs=4)
    # NOTE: steps_per_epoch (typically)= dataset_size // batch_size
    #       However, you can change this number to "trick" the trainer, e.g., to update the learning rate using ReduceLROnPlateau() callback, or just to stop training sooner.
    #       Or, in case of infinite training data (.repeat(None)), we MUST specify this for the training to stop.
    model.save('models/effB3_trainFClayers.h5')
    test_loss, test_acc = model.evaluate(test_set)
    import joblib
    joblib.dump(test_acc, 'models/effB3_trainFClayers_testAccuracy')
else:
    model = keras.models.load_model('models/effB3_trainFClayers.h5')
    test_acc = joblib.load('models/effB3_trainFClayers_testAccuracy')
print('Test accuracy:',test_acc)
# Try prediction:
if 10:
    plt.figure(figsize=(12, 80))
    index = 0
    test_set_raw = test_set_raw.shuffle(buffer_size=50)
    for image, label in test_set_raw.take(30):
        index += 1
        plt.subplot(15, 2, index)
        plt.imshow(image)

        test_img, label = preprocess(image, label)
        with tf.device('/cpu'):
            prediction =  model(test_img, training=False) # NOTE: model.predict() is designed for performance in large scale inputs. For small amount of inputs that fit in one batch, directly using __call__ is recommended for faster execution, e.g., model(x), or model(x, training=False)
        prediction_lbl = np.argmax(prediction.numpy())
        prediction_score = np.max(prediction.numpy())

        plt.title("Label: {} \nTop-1 predict: {} ({}%)".format(class_names[label],class_names[prediction_lbl], round(prediction_score*100,1)), fontsize=20)
        plt.axis("off")


#%% STEP 2: Unfreeze and train all layers (or just the top ones)
# CAUTION: takes a while to finish! ==> Use GPU or try using Google Colab.
class OneCycleScheduler(keras.callbacks.Callback):
    # Source: (Géron, 2019)
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
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

new_training = 0
if new_training:
    batch_size = 32
    good_lr = 0.1
    init_lr = good_lr/10

    model = keras.models.load_model('models/effB3_trainFClayers.h5')
    for layer in model.layers[-10:]:
        layer.trainable = True
    optimizer = keras.optimizers.Nadam(learning_rate=init_lr)
    # optimizer = keras.optimizers.SGD(learning_rate=0.03, momentum=0.9, nesterov=True, decay=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    checkpoint_saver = keras.callbacks.ModelCheckpoint('models/effB3_trainMoreLayers.h5', save_best_only=True)
    early_stopper = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
    #performance_sched = keras.callbacks.ReduceLROnPlateau(patience=3)

    n_epochs = 20
    n_iters = int(dataset_size / batch_size) * n_epochs
    onecycle = OneCycleScheduler(n_iters, max_rate=good_lr)

    #with tf.device('/cpu'):
    history = model.fit(train_set,
                        steps_per_epoch=int(0.75 * dataset_size / batch_size),
                        validation_data=valid_set,
                        validation_steps=int(0.15 * dataset_size / batch_size),
                        epochs=n_epochs,
                        callbacks=[checkpoint_saver,early_stopper,onecycle])

    test_loss, test_acc = model.evaluate(test_set)
    joblib.dump(test_acc, 'models/effB3_trainMoreLayers_testAccuracy')
else:
    model = keras.models.load_model('models/effB3_trainMoreLayers.h5')
    test_acc = joblib.load('models/effB3_trainMoreLayers_testAccuracy')
print('Test accuracy:',test_acc)
# Try prediction:
if 10:
    plt.figure(figsize=(12, 80))
    index = 0
    test_set_raw = test_set_raw.shuffle(buffer_size=50)
    for image, label in test_set_raw.take(30):
        index += 1
        plt.subplot(15, 2, index)
        plt.imshow(image)

        test_img, label = preprocess(image, label)
        #with tf.device('/cpu'):
        prediction =  model(test_img, training=False)
        prediction_lbl = np.argmax(prediction.numpy())
        prediction_score = np.max(prediction.numpy())

        plt.title("Label: {} \nTop-1 predict: {} ({}%)".format(class_names[label],class_names[prediction_lbl], round(prediction_score*100,1)), fontsize=20)
        plt.axis("off")
