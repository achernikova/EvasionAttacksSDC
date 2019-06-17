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

IMAGE_FILE = 'straight_right_left.csv'
IMAGE_FOLDER = '/home/alesia/Documents/sdc/'
MODEL_FILE = 'models/sdc_epoch'
RESULTS_FILE = 'res_attack_epoch.txt'
RESULTS_SUCCESS_FILE = 'res_attack_success_epoch.txt'
RESULTS_PROBAS_FILE = 'res_attack_probas_epoch.txt'
RESULTS_LABELS_FILE = 'res_attack_probas_labels_epoch.txt'
NUM_ATTACKS = 300

with tf.Session() as sess:

    data, model =  SDC_data(IMAGE_FILE, IMAGE_FOLDER), SDC_model_epoch(MODEL_FILE, sess)
    
    distortions = []  
    success = 0

    results = open(RESULTS_FILE, 'w')
    results_success = open(RESULTS_SUCCESS_FILE, 'w')
    results_probas = open(RESULTS_PROBAS_FILE,'w')
    results_labels_probas = open(RESULTS_LABELS_FILE,'w')

    tmp = 0

    for k in range (2):
        
        attack = L2ClassificationAttack(sess, model, batch_size = 2, max_iterations = 1000, confidence = 0)
        inputs, targets = generate_data(data, samples = 1, targeted = True, start = k, inception = False)

        adv = attack.attack(inputs, targets)

        for i in range(len(adv)):

            inp =  model.model.predict(inputs[i:i+1])
            inp = softmax(inp)
            inp_str = str(inp[:,0])+','+str(inp[:,1])+','+str(inp[:,2])+'\n'
            results_labels_probas.write(inp_str)

            adver = model.model.predict(adv[i:i+1])
            adver = softmax(adver)
            adver_str = str(adver[:,0])+','+str(adver[:,1])+','+str(adver[:,2])+'\n'
            results_probas.write(adver_str)

            if (np.argmax(inp) != np.argmax (adver)):
                success +=1

            distortions.append(np.sum((adver - inp)**2)**.5)
        
    results.write(str(distortions))
    results_success.write(str(success))
       




    # for k in range (3):
        


    #     attack = CarliniL2(sess, model, batch_size = 2, max_iterations=1000, confidence=0)

    #     inputs, targets = generate_data(data, samples = 1, targeted=True, start=k, inception=False)

    #     tmp += np.sum((inputs[0])**2)**.5

    #     adv = attack.attack(inputs, targets)

    #     boxmul = 1 / 2
    #     boxplus = 1/2

        




    #     plt.imshow(inputs[0])
    #     plt.axis('off')
    #     plt.savefig('input_epoch'+str(k)+'.png', dpi = 1000)

    #     # x = adv[0]

    #     # x = np.arctanh((x - boxplus) /boxmul * 0.999999)
    #     # print (x)
    
    #     # plt.imshow(x)
    #     #plt.show()

    #     for i in range(len(adv)):
    #         inp =  model.model.predict(inputs[i:i+1])
    #         # print("Valid:",inp)
            
    #         # inp = softmax(inp)

    #         # inp_str = str(inp[:,0])+','+str(inp[:,1])+','+str(inp[:,2])+'\n'

    #         # results_labels_probas.write(inp_str)

    #         adver = model.model.predict(adv[i:i+1])
    #         # print("Classification:", adver)
    #         # print()

    #         # adver = softmax(adver)
    #         # adver_str = str(adver[:,0])+','+str(adver[:,1])+','+str(adver[:,2])+'\n'
    #         # results_probas.write(adver_str)


    #         x = adv[i]
    #         x = np.arctanh((x - 0) /1/2 * 0.999999)

    #         plt.imshow(x)
    #         plt.axis('off')

    #         figname = 'adv_epoch'+str(k)+str(i)+'.png'
    #         plt.savefig(figname, dpi = 1000)
        


    #         #results.write(str(adver))
    #         #results.write("---------------------")

    #         # if (np.argmax(inp) != np.argmax (adver)):
    #         #     success +=1

    #         # print(success)

    #         # distortions.append(np.sum((adver - inp)**2)**.5)
    #         # print(str(distortions))
            

        






































