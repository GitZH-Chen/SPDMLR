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
model_path=['/home/zchen/Proposed_Methods/RMLR/SPDNet-PowerLogEM-MLR/Radar-AlphaEM-MLR/0.025-SPDNet-AlphaPM-RMLR-BWM-0.5-[20,16,8]-11_28.tar-199',
            '/home/zchen/Proposed_Methods/RMLR/SPDNet-AIM-MLR/Radar/0.025-SPDNet-AIMRMLR-BWM-[20,16,8]-11_29.tar-299',
            '/home/zchen/Proposed_Methods/RMLR/SPDNet-OILEM-BW/Radar/0.025-SPDNet-OILEM-RMLR-BW-a1_b1-div-fixed-[20,16,8]-11_33.tar-199']
titles = ['0.025-SPDNet-AlphaPM-RMLR-BWM-0.5-[20,16,8]','0.025-SPDNet-AIMRMLR-BWM-[20,16,8]','0.025-SPDNet-OILEM-RMLR-BWM-[20,16,8]']
c=3;n=8
num = len(model_path)
fig,axs = plt.subplots(num,c)
axs = axs.flatten()
weight = np.empty([num,c,n,n])

for ith in range(num):
    checkpoint = th.load(model_path[ith])
    param = checkpoint['model_state_dict']
    weight[ith,:,:,:]  = param['RMLR.P'].data.numpy()


max,min= weight.max(), weight.min()
norm = matplotlib.colors.Normalize(vmin=min,vmax=max)
for ith in range(num):
    for jth in range(c):
        im = axs[ith * c +jth].matshow(weight[ith,jth,:,:],cmap=plt.cm.coolwarm,norm=norm)
    axs[ith * c +jth-1].set_title(titles[ith])



plt.tight_layout()
# fig.colorbar(im, ax=[axs[0], axs[1], axs[2], axs[3]])
fig.colorbar(im, ax=[axs[0], axs[1], axs[2], axs[3],axs[4], axs[5],axs[6],axs[7], axs[8]])
plt.show()
print('max: ' + str(max)+' min: ' + str(min))