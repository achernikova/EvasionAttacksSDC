import tensorflow as tf
import numpy as np
import imageio as im
import os
import skimage.transform as st

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model
from os import path


def read_images_steering_angles(steering_image_log,image_folder):
    
    steerings = []
    imgs = []
    
    with open(steering_image_log) as f:
        for line in f.readlines()[1:]:
            
            fields = line.split(",")
            
            if 'center' not in line:
                continue
            
            steering = float(fields[6])

            url = fields[5]
            
            img = im.imread(path.join(image_folder, url))
            
            crop_img = img[200:,:]
            img = st.resize(crop_img, (128, 128))
            
            steerings.append(steering)
            imgs.append(img)
            
    return imgs, steerings

class SDC_data:
    def __init__(self, image_file, image_folder):
             
        data, labels = read_images_steering_angles(image_file,image_folder)
        
        self.attack_data = np.asarray(data)
        self.attack_labels = np.asarray(labels)


def generate_data(data, samples, targeted=True, start=0):
    """
    Generate the input data to the attack algorithm.
    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    """
    inputs = []
    targets = []
    for i in range(samples):


            inputs.append(data.attack_data[start+i])
            targets.append(data.attack_labels[start+i].flatten())

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

