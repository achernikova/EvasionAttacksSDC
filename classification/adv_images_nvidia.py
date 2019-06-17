import numpy as np

import tensorflow as tf
import os
import sys

import pickle
import gzip
import urllib.request

from os import path
import random
from keras import backend as K
K.set_learning_phase(0)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


import time

import imageio as im
import skimage.transform as st

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.models import load_model

from utilities import SDC_data, generate_data, softmax
from attack import L2ClassificationAttack
from model import SDC_model_epoch

IMAGE_FILE = 'nvidia.csv'
IMAGE_FOLDER = '/home/alesia/Documents/sdc/'
MODEL_FILE = 'models/sdc_nvidia'

NUM_IMAGES = 3

with tf.Session() as sess:


    data = SDC_data(IMAGE_FILE, IMAGE_FOLDER)
    model = SDC_model_epoch(MODEL_FILE)

    for k in range (NUM_IMAGES):
            
        attack = L2ClassificationAttack(sess, model, batch_size = 2, max_iterations=1000, confidence=0)

        inputs, targets = generate_data(data, samples = 1, targeted=True, start=k, inception=False)

        adv = attack.attack(inputs, targets)

        boxmul = 1/2
        boxplus = 1/2

        plt.imshow(inputs[0])
        plt.axis('off')
        plt.savefig('results/input_epoch'+str(k)+'.png', dpi = 250)

        for i in range(len(adv)):
            
            adver = model.model.predict(adv[i:i+1])

            x = adv[i]
            x = np.arctanh((x - 0) /1/2 * 0.999999)

            plt.imshow(x)
            plt.axis('off')

            figname = 'resultst/adv_epoch'+str(k)+str(i)+'.png'
            plt.savefig(figname, dpi = 250)
        























