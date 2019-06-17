import tensorflow as tf
import numpy as np
import imageio as im
import os
import skimage.transform as st

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from os import path

from utilities import SDC_data

def train(data, file_name, params, num_epochs = 50, batch_size = 128, init = None):
    """
    Standard neural network training procedure.
    """
    model = Sequential()

    model.add(BatchNormalization())
    
    model.add(Conv2D(params[0], (5, 5), input_shape = data.attack_data.shape[1:]))
    model.add(Activation('relu'))

    model.add(Conv2D(params[1], (5, 5)))
    model.add(Activation('relu'))

    model.add(Conv2D(params[2], (5, 5)))
    model.add(Activation('relu'))

    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))

    model.add(Conv2D(params[4], (3, 3)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(params[5]))
    model.add(Activation('relu'))

    model.add(Dense(params[6]))
    model.add(Activation('relu'))

    model.add(Dense(params[7]))
    model.add(Activation('relu'))

    model.add(Dense(params[8]))
    model.add(Activation('relu'))

    model.add(Dense(params[9]))
    
    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels = correct,
                                                       logits = predicted)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(data.attack_data, data.attack_labels,
              batch_size=batch_size,
              validation_split = 0.1,
              nb_epoch=num_epochs,
              shuffle=True)

    if file_name != None:
        model.save(file_name)

    return model
    
if not os.path.isdir('models'):
    os.makedirs('models')

IMAGE_FILE = 'straight_right_left.csv'
IMAGE_FOLDER = '/home/alesia/Documents/sdc/'

train(SDC_data(IMAGE_FILE, IMAGE_FOLDER), "models/sdc_nvidia", [24, 36, 48, 64, 64, 1164, 100, 50, 10, 3], num_epochs = 50)

