import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import csv
# import pandas as pd
import numpy as np
from plot_utili import  read_cvs_data,plot_data

fontsize = 14
data_path_SPDNet = '5e-2_SPDNet-[93,30].csv'
data_path_SPDNetMul = '5e-2_SPDNet-RMLR-uniform6A-ID_P-AIM-[93,30].csv'
legend = ['SPDNet','SPDNet-RMLR']

epochs=200
y_min = 40
y_max= 66
step=5
dpi=400

SPDNet=read_cvs_data(data_path_SPDNet)
SPDNetMUL=read_cvs_data(data_path_SPDNetMul)
epoch = (range(1,epochs+1))

chart_num = len(data_path_SPDNet)
plt.figure(figsize=(7, 5), dpi=dpi)
tmp_SPDNet = SPDNet[0:epochs]
tmp_SPDNetMUL= SPDNetMUL[0:epochs]
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(which='major',ls='--',alpha=.8,lw=.8)
my_y_ticks = np.arange(y_min, y_max, step)
# my_x_ticks = np.arange(0, epochs+step_epcoh, step_epcoh)
plot_data(epoch,tmp_SPDNet,tmp_SPDNetMUL,legend,my_y_ticks,y_min=y_min,y_max=y_max,fontsize=fontsize)
plt.show()
plt.savefig('ACC_RMLR_HDM05.pdf',dpi=dpi)

a=1

