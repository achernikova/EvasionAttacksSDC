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

def read_images_steering_directions(steering_image_log, image_folder):
    """
        Read the images with corresponding direction
        :param steering_image_log: the file that contains the path to image and all necessary information about it including steering angle and path to image
        :param image_folder: the path to the folder that contains all images
        :return: imgs, steerings: images, labels
    """

    STEERING_ANGEL_THRESHOLD = 0.15
    NUM_CLASSES = 3
    
    steerings = []
    imgs = []
    
    with open(steering_image_log) as f:
        for line in f.readlines()[1:]:
            
            fields = line.split(",")
            
            if 'center' not in line:
                continue
            
            #getting the value of steering angle
            steering = fields[6]
            steering_label = np.zeros(NUM_CLASSES)
            #checking if the direction is 'right'
            if float(steering) > STEERING_ANGEL_THRESHOLD:
                steering_label[2] = 1

            elif float(steering) < -1 * STEERING_ANGEL_THRESHOLD:
                steering_label[0] = 1

            else:
                steering_label[1] = 1

            #path to the image
            url = fields[5]
            #reading the image
            img = im.imread(path.join(image_folder, url))
            
            #image preprocessing
            crop_img = img[200:,:]
            img = st.resize(crop_img, (128, 128))
            
            steerings.append(steering_label)
            imgs.append(img)
            
    return imgs, steerings

def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class SDC_data:
    def __init__(self, image_file, image_folder):
             
        data, labels = read_images_steering_directions(image_file,image_folder)
        
        self.attack_data = np.asarray(data)
        self.attack_labels = np.asarray(labels)


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.
    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.attack_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.attack_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.attack_data[start+i])
                targets.append(np.eye(data.attack_labels.shape[1])[j])
        else:
            inputs.append(data.attack_data[start+i])
            targets.append(data.attack_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets
