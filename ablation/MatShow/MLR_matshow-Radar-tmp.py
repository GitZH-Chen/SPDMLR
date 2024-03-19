import sys
sys.path.insert(0,'/data/Disk_A/ziheng/Proposed/CholeskyNet/')
from pathlib import Path
import os
import random
import argparse
import time

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

import spd_ChoMetric.nn as nn_spd
import spd_ChoMetric.functional as functional_spd
from spd_ChoMetric.optimizers import MixOptimizer
from spd_ChoMetric.utils import resuming_writer

import matplotlib.pyplot as plt
import matplotlib

plt.close()
base_dir='/home/zchen/Proposed_Methods/RMLR/SPDNet-BWM-MLR/Radar'
name = '1e-2-SPDNet-BWMRMLR-Ly_PT-BWM-div4-[20,16,8]-epoch26.npy'
model_path = os.path.join(base_dir,name)
titles = name
c=3;n=8
fig,axs = plt.subplots(1,c)
axs = axs.flatten()

weight = np.load(model_path)


max,min= weight.max(), weight.min()
norm = matplotlib.colors.Normalize(vmin=min,vmax=max)
for ith in range(c):
    im = axs[ith].matshow(weight[ith,:,:],cmap=plt.cm.coolwarm,norm=norm)

axs[0].set_title(titles)



plt.tight_layout()
# fig.colorbar(im, ax=[axs[0], axs[1], axs[2], axs[3]])
fig.colorbar(im, ax=[axs[0], axs[1], axs[2]])
plt.show()
print('max: ' + str(max)+' min: ' + str(min))