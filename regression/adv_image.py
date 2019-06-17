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

from utilities import SDC_data, generate_data
from model import SDC_model_epoch
from attack import L2RegressionAttack

IMAGE_FILE = 'test_image.csv'
IMAGE_FOLDER = '/home/alesia/Documents/sdc/'

with tf.Session() as sess:

    data, model =  SDC_data(IMAGE_FILE, IMAGE_FOLDER), SDC_model_epoch("models/sdc", sess)
        
    attack = L2RegressionAttack(sess, model, batch_size=1, max_iterations=1000, confidence=0)

    inputs, targets = generate_data(data, samples=1, targeted=True, start=0)

    adv = attack.attack(inputs, targets)
    
    for i in range(len(adv)):

        inp =  model.model.predict(inputs[i:i+1])
        print("Valid :",inp)
        adver = model.model.predict(adv[i:i+1])
        print("Classification:", adver)
        print()

        mse =  (targets[i] - adver) * (targets[i] - adver)
        mse_init = (targets[i] - inp) * (targets[i] - inp)

        print('MSE',  (targets[i] - adver) * (targets[i] - adver))
        print('MSE_init',  (targets[i] - inp) * (targets[i] - inp))

        plt.imshow(inputs[i], interpolation='none')
        plt.savefig('results/input_image.png',dpi = 250)

        plt.imshow(adv[i], interpolation='none')
        plt.savefig('results/adv_image.png',dpi = 250 )

     

