import numpy as np
import pandas as pd

import tensorflow as tf
import os
import sys

import pickle
import gzip
import urllib.request

from os import path
import random

import time

import imageio as im
import skimage.transform as st

from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.models import load_model

import matplotlib.pyplot as plt

class SDC_model_epoch:
    
    def __init__(self, restore, session = None):

        self.image_size = 128
        self.num_channels = 3 
        self.num_labels = 1

        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), input_shape =(128, 128, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))

        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)
