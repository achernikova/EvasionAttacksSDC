import numpy as np

import tensorflow as tf
import os
import sys

import pickle
import gzip
from keras import backend as K
K.set_learning_phase(1)
from os import path
import random

import time

import imageio as im
import skimage.transform as st

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.models import load_model

from utilities import SDC_data, generate_data
from attack import L2ClassificationAttack
from model import SDC_model_nvidia

IMAGE_FILE = 'straight_right_left.csv'
IMAGE_FOLDER = '/home/alesia/Documents/sdc/'
MODEL_FILE = 'models/sdc_nvidia'
RESULTS_FILE = 'results/res_attack_nvidia.txt'
RESULTS_SUCCESS_FILE = 'results/res_attack_success_nvidia.txt'
RESULTS_PROBAS_FILE = 'results/res_attack_probas_nvidia.txt'
RESULTS_LABELS_FILE = 'results/res_attack_labels_nvidia.txt'
NUM_ATTACKS = 300

with tf.Session() as sess:

    data, model =  SDC_data(IMAGE_FILE, IMAGE_FOLDER), SDC_model_nvidia(MODEL_FILE, sess)
    distortions = []  
    success = 0
    results = open(RESULTS_FILE, 'w')
    results_success = open(RESULTS_SUCCESS_FILE, 'w')
    results_probas = open(RESULTS_PROBAS_FILE,'w')
    results_labels = open(RESULTS_LABELS_FILE,'w')


    for k in range (NUM_ATTACKS):

        attack = L2ClassificationAttack(sess, model, batch_size = 2, max_iterations = 1000, confidence = 0)

        inputs, targets = generate_data(data, samples = 1, targeted=True, start = k, inception=False)
        adv = attack.attack(inputs, targets)

        for i in range(len(adv)):

            inp =  model.model.predict(inputs[i:i+1])
            results_labels.write(str(inp))  

            adver = model.model.predict(adv[i:i+1])
            results_probas.write(str(adver))

            if (np.argmax(inp) != np.argmax (adver)):
                success +=1

            distortions.append(np.sum((adver - inp)**2)**.5)

    results.write(str(distortions))
    results_success.write(str(success))
