import numpy as np
from itertools import product
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DISTANCES_FILE = 'results/res_attack_epoch.txt'

with open(DISTANCES_FILE, 'r') as f:
    dist = f.read()

dist_str = dist[1:len(dist)-2]
dist_str_list = dist_str.split(',')
dist_floats = [float(i) for i in dist_str_list]

adv_0_dist = dist_floats[::2]
adv_2_dist = dist_floats[1::2]

adv_aver_dist = (np.asarray(adv_0_dist) + np.asarray(adv_2_dist))/2

num_adversaries = 600

more_0 = 0
more_9 = sum(i < 0.71 for i in dist_floats)/num_adversaries
more_11 = sum(i < 0.72 for i in dist_floats)/num_adversaries
more_13 = sum(i < 0.73 for i in dist_floats)/num_adversaries
more_15 = sum(i < 0.74 for i in dist_floats)/num_adversaries
more_17 = sum(i < 0.75 for i in dist_floats)/num_adversaries
more_19 = sum(i < 0.76 for i in dist_floats)/num_adversaries
more_21 = sum(i < 0.77 for i in dist_floats)/num_adversaries
more_23 = sum(i < 0.78 for i in dist_floats)/num_adversaries
more_25 = sum(i < 0.79 for i in dist_floats)/num_adversaries
more_27 = sum(i < 0.80 for i in dist_floats)/num_adversaries

srates = [0.7 + i * 0.01 for i in range(0, 11)]

results = []

results.append(more_0)
results.append(more_9)
results.append(more_11)
results.append(more_13)
results.append(more_15)
results.append(more_17)
results.append(more_19)
results.append(more_21)
results.append(more_23)
results.append(more_25)
results.append(more_27)


plt.xlabel("Maximum L2-norm perturbation", fontsize=14)
plt.ylabel("Attack success rate",fontsize=14)
  
plt.plot(srates,results,linewidth=3,markersize=8,label='L2 Carlini attack',marker='D')

plt.legend(loc=4,fontsize=14)
ax = plt.gca()
plt.setp(ax.get_xticklabels())
plt.setp(ax.get_yticklabels())
plt.tight_layout()
plt.ylim(-0.03,1.1)
plt.savefig("results/epoch_success_rate.png", dpi = 250)
