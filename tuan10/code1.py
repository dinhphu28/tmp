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


#Load model EfficientNet
# model = keras.applications.EfficientNetB7(
#     include_top=False, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000,
#     classifier_activation='softmax')

# Source: https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB7
model = tf.keras.applications.EfficientNetB7(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=max, classes=1000,
    classifier_activation='softmax'
)

#load và resize lại images
#EfficientNetB7 có input shape là 600x600x3
import glob
file_names = glob.glob ("images/*.jpg")
images = np.empty((0,600,600,3))
for file_name in file_names:
    img_BGR = cv2.imread(file_name)
    img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    #img = tf.image.resize(img, (224,224), antialias=True)
    img = tf.image.resize_with_pad(img, 600,600, antialias=True) # NOTE: To resize but KEEP ASPECT RATIO: resize_with_pad() or crop_and_resize()
    images = np.append(images, img[np.newaxis,:,:,:], axis=0)
    if 10:
        plt.imshow(img/255)
        plt.title(file_name)
        plt.show()
print(images.shape)
#preprocess input đầu vào bằng lệnh preprocess_input
inputs = tf.keras.applications.efficientnet.preprocess_input(images)
#Dự đoán
#ImageNet có tới 1000 classes  nên chỉ  cần lấy 3 classes có dự đoán cao nhất
Y_proba = model.predict(inputs)
k=3
top_k_predictions = tf.keras.applications.efficientnet.decode_predictions(Y_proba, top=k)
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
