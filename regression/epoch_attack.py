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


MSE_RATIO = open('results/res_attack_mse_ratio.txt', 'w')
MSE_NO_ATTACK = open('results/res_attack_mse_no_attack.txt', 'w')
MSE_WITH_ATTACK = open('results/res_attack_mse_with_attack.txt', 'w')
DISTANCES = open('results/res_attack_distance.txt', 'w')

IMAGE_FILE = 'straight_right_left.csv'
IMAGE_FOLDER = '/home/alesia/Documents/sdc/'

NUM_ATTACKS = 1

with tf.Session() as sess:

    data, model =  SDC_data(IMAGE_FILE, IMAGE_FOLDER), SDC_model_epoch("models/sdc", sess)
        
    attack = L2RegressionAttack(sess, model, batch_size=1, max_iterations=1000, confidence=0)

    inputs, targets = generate_data(data, samples=1, targeted=True, start=0)

    adv = attack.attack(inputs, targets)
        
    mse_no_attack = []
    mse_with_attack = []
    ratio = []
    ratio_angle = []
    distortion = []
    pred_true = []
    pred_attack = []

    for i in range(NUM_ATTACKS):

        print("Input",targets[i])
        inp =  model.model.predict(inputs[i:i+1])
        print("Valid :",inp)
        adver = model.model.predict(adv[i:i+1])
        print("Classification:", adver)
        print()

        mse =  (targets[i] - adver) * (targets[i] - adver)
        mse_with_attack.append(mse[0][0])
        MSE_WITH_ATTACK.write(str(mse[0][0])+'\n')

        mse_init = (targets[i] - inp) * (targets[i] - inp)
        mse_no_attack.append(mse_init[0][0])
        MSE_NO_ATTACK.write(str(mse_init[0][0])+'\n')

        print('MSE',  (targets[i] - adver) * (targets[i] - adver))
        print('MSE_init',  (targets[i] - inp) * (targets[i] - inp))

        ratio_mse = mse[0][0] / mse_init[0][0]
        ratio.append(ratio_mse)
        MSE_RATIO.write(str(ratio_mse)+'\n')

        ratio_steering = adver[0][0]/inp[0][0]
        ratio_angle.append(ratio_steering)

        pred_true.append(inp[0][0])
        pred_attack.append(adver[0][0])

        dist =  np.sum((adv[i]-inputs[i])**2)**.5
        distortion.append(dist)
        DISTANCES.write(str(dist)+'\n')

