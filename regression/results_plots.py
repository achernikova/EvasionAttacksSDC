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


mse_no_attack = []
mse_with_attack = []

mse_ratio = []

distortion = []


with open('results/res_attack_mse_no_attack.txt', 'r') as f:

    for mse in f:
        mse_no_attack.append(float(mse))

with open('results/res_attack_mse_with_attack.txt', 'r') as f:

    for mse in f:
        mse_with_attack.append(float(mse))

with open('results/res_attack_mse_ratio.txt', 'r') as f:

    for ratio in f:
        mse_ratio.append(float(ratio))

with open('results/res_attack_distance.txt', 'r') as f:

    for dist in f:
        distortion.append(float(dist))

for q in [10, 25, 50, 75, 90]:
  print ("{}%% ratio: {}".format (q, np.percentile(mse_ratio, q)))

for q in [10, 25, 50, 75, 90]:
  print ("{}%% distance: {}".format (q, np.percentile(distortion, q)))

num_bins = 30

counts, bin_edges = np.histogram (mse_no_attack, bins = num_bins)
cdf = np.cumsum (counts)
cdf = list(cdf)
bin_edges_list = list(bin_edges[1:])
cdf = cdf/cdf[-1]
cdf_list = list(cdf)

bin_edges_list.append(0.014)
cdf_list.append(1)

bin_edges_list = [0] + bin_edges_list
cdf_list = [0] + cdf_list

plt.plot (bin_edges_list, cdf_list, label = 'MSE, no attack', linewidth = '2')

counts_attack, bin_edges_attack = np.histogram (mse_with_attack, bins = num_bins)
cdf_attack = np.cumsum (counts_attack)
cdf_attack = list(cdf_attack)
bin_edges_attack_list = list(bin_edges_attack[1:])
cdf_attack = cdf_attack/cdf_attack[-1]
cdf_attack_list = list(cdf_attack)

bin_edges_attack_list = [0] + bin_edges_attack_list
cdf_attack_list = [0] + cdf_attack_list

plt.plot (bin_edges_attack_list, cdf_attack_list, label = 'MSE, attack', linewidth = '2')

plt.xlabel("MSE value",fontsize=12)
plt.ylabel("CDF",fontsize=12)
  
plt.legend(loc = 'lower right')

plt.savefig('results/cdf_mse.png', dpi = 250)


