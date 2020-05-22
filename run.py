# ----------------Required libraries----------------------#

import sys
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

import cv2
import time
import itertools
import random

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras import models
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer, InputSpec
from keras.regularizers import l2
from keras import backend as K
import keras.backend.tensorflow_backend as tfback
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.multi_gpu_utils import multi_gpu_model

from preprocessing import fetch_groups
from tensorflow_details import _get_available_gpus
from model import create_base_network_signet, eucl_dist_output_shape, contrastive_loss, euclidean_distance, load_and_check_model, test_model, predict_score

# ----------------------------------------------------------------------------------------------------#
# ----------------------------------------STEP :1-----------------------------------------------------#
#--------------------------------Splitting into train & test------------------------------------------#

orig_groups, forg_groups = fetch_groups()
orig_train, orig_test, forg_train, forg_test = train_test_split(orig_groups, forg_groups, test_size=0.2, random_state=1)
orig_train, orig_val, forg_train, forg_val = train_test_split(orig_train, forg_train, test_size=0.25, random_state=1)
print("Report: Created train, validation and test data sucessfully. ")

# ----------------------------------------------------------------------------------------------------#
# ----------------------------------------STEP :2-----------------------------------------------------#
#----------------------------Visualize images to see the signatures-----------------------------------#

img_h, img_w = 155, 220

def visualize_sample_signature():
    '''Function to randomly select a signature from train set and
    print two genuine copies and one forged copy'''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 10))
    k = np.random.randint(len(orig_train))
    orig_img_names = random.sample(orig_train[k], 2)
    forg_img_name = random.sample(forg_train[k], 1)
    orig_img1 = cv2.imread(orig_img_names[0], 0)
    orig_img2 = cv2.imread(orig_img_names[1], 0)
    forg_img = plt.imread(forg_img_name[0], 0)
    orig_img1 = cv2.resize(orig_img1, (img_w, img_h))
    orig_img2 = cv2.resize(orig_img2, (img_w, img_h))
    forg_img = cv2.resize(forg_img, (img_w, img_h))

    ax1.imshow(orig_img1, cmap = 'gray')
    ax2.imshow(orig_img2, cmap = 'gray')
    ax3.imshow(forg_img, cmap = 'gray')

    ax1.set_title('Genuine Copy')
    ax1.axis('off')
    ax2.set_title('Genuine Copy')
    ax2.axis('off')
    ax3.set_title('Forged Copy')
    ax3.axis('off')

#---Testing visualize function---------#    
visualize_sample_signature()
visualize_sample_signature()
visualize_sample_signature()

# ----------------------------------------------------------------------------------------------------#
# -------------------------------------------STEP :3--------------------------------------------------#
#-------------------------------------------Run model-------------------------------------------------#

tfback._get_available_gpus = _get_available_gpus
input_shape=(img_h, img_w, 1)

# network definition
base_network = create_base_network_signet(input_shape)

input_a = Input(shape=(input_shape))
input_b = Input(shape=(input_shape))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Compute the Euclidean distance between the two vectors in the latent space
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
p_model = Model(input=[input_a, input_b], output=distance, name = 'head_model')

num_train_samples = 276*120 + 300*120
num_val_samples = num_test_samples = 276*20 + 300*20
num_train_samples, num_val_samples, num_test_samples

# compile model using RMSProp Optimizer and Contrastive loss function defined above
rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
p_model.compile(loss=contrastive_loss, optimizer=rms)
p_model.summary()


callbacks = [
    EarlyStopping(patience=15, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    ModelCheckpoint('Weights/model-signet-{epoch:03d}.h5', verbose=1, save_weights_only=True)
]

batch_sz = 128
results = p_model.fit_generator(generate_batch(orig_train, forg_train, batch_sz),
                              steps_per_epoch = num_train_samples//batch_sz,
                              epochs = 20,
                              validation_data = generate_batch(orig_val, forg_val, batch_sz),
                              validation_steps = num_val_samples//batch_sz,
                              callbacks = callbacks)

# ----------------------------------------------------------------------------------------------------#
# -------------------------------------------STEP :4--------------------------------------------------#
#------------------------------------Predict Accuracy-------------------------------------------------#

acc_thresh = []
for i in range(1,20,1):
    acc_thresh.append(load_and_check_model('Weights/model-signet-'+str(i).zfill(3)+'.h5'))
    print('For model '+str(i)+' Validation Accuracy = ',acc_thresh[i-1][0]*100,'%')
    
acc, threshold = test_model('Weights/model-signet-020.h5')
print("Accuracy: {} & Threshold: {}".format(acc, threshold))

predict_score()
