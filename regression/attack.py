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


BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2   # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess

class L2RegressionAttack:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 boxmin = -0.5, boxmax = 0.5):
        """
        The L_2 optimized attack. 
        Returns adversarial examples for the supplied model.
        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels

        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        shape = (batch_size, image_size, image_size, num_channels)
        
        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape,dtype = np.float32))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size, num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        #self.boxmul = (boxmax - boxmin) / 2.
        self.boxmul = 0.5

        #self.boxplus = (boxmin + boxmax) / 2.
        self.boxplus = 0.5

        self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus

        #self.newimg = self.timg + modifier
        # prediction BEFORE-SOFTMAX of the model
        self.output = model.predict(self.newimg)
        
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)),[1,2,3])
        
        #MSE error
        self.mseloss = 0.5 * tf.reduce_sum((self.output - self.tlab) * (self.output - self.tlab), 1)


        # compute the probability of the label class versus the maximum other
        #real = tf.reduce_sum((self.tlab)*self.output,1)
        #other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)

        #if self.TARGETED:
            # if targetted, optimize for making the other class most likely
         #   loss1 = tf.maximum(0.0, other-real+self.CONFIDENCE)
        #else:
            # if untargeted, optimize for making this class least likely.
         #   loss1 = tf.maximum(0.0, real-other+self.CONFIDENCE)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.mseloss)
        self.loss =  -100 * self.loss1 +  self.loss2
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        
        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        batch_size = self.batch_size

        # convert to tanh-space
        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestl1 = [0]*batch_size

        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        
        # completely reset adam's internal state.
        self.sess.run(self.init)
        batch = imgs[:batch_size]
        batchlab = labs[:batch_size]
    
        bestl2 = [1e10]*batch_size
        bestl1 = [0]*batch_size

        # set the variables so that we don't have to send them over again
        self.sess.run(self.setup, {self.assign_timg: batch,
                                   self.assign_tlab: batchlab, self.assign_const: CONST})
            
        prev = 1e6
        for iteration in range(self.MAX_ITERATIONS):
            # perform the attack 
            _, l, mses, l2s, scores, nimg = self.sess.run([self.train, self.loss, self.mseloss, self.l2dist, self.output, 
                                                         self.newimg])

            #    if np.all(scores>=-.0001) and np.all(scores <= 1.0001):
            #        if np.allclose(np.sum(scores,axis=1), 1.0, atol=1e-3):
            #            if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
            #                raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")
                
            # print out the losses every 10%
            if iteration%(self.MAX_ITERATIONS//20) == 0:
                print(iteration,self.sess.run((self.loss,self.loss1,self.loss2)))

            # check if we should abort search if we're getting nowhere.
            if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                if l > prev*.9999:
                    break
                prev = l

            # adjust the best result found so far
            for e,(l2, mse, ii) in enumerate(zip(l2s,mses, nimg)):
                if l2 < bestl2[e] and mse > bestl1[e]:
                    bestl1[e] = mse
                    bestl2[e] = l2

                if l2 < o_bestl2[e] and mse > bestl1[e]:
                    o_bestl1[e] = mse
                    o_bestl2[e] = l2

                    o_bestattack[e] = ii

        return o_bestattack