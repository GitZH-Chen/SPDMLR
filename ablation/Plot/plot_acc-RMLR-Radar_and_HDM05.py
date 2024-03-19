import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import csv
# import pandas as pd
import numpy as np
from plot_utili import  read_cvs_data,plot_data_g

fontsize = 10
dpi=400
plt.figure(figsize=(14, 8.6), dpi=dpi)

legend = ['SPDNet','SPDNet-RMLR']
#HDM05
data_path_SPDNet = '5e-2_SPDNet-[93,30].csv'
data_path_SPDNetRMLR = '5e-2_SPDNet-RMLR-uniform6A-ID_P-AIM-[93,30].csv'
legend = ['SPDNet','SPDNet-RMLR']

epcohs=200
y_min = 40
y_max= 66
step=5

plt.subplot(121)
plot_data_g(data_path_SPDNet,data_path_SPDNetRMLR,epcohs,y_min,y_max,step,legend)

## Radar
data_path_SPDNet = '1e-2-SPDNet-[20,16,8].csv'
data_path_SPDNetRMLR = '1e-2-SPDNet-OILEM-RMLR-a1_b1-div-fixed-[20,16,8]-23_16.csv'
legend = ['SPDNet','SPDNet-RMLR']

epochs=300
y_min = 70
y_max= 100
step=5
plt.subplot(122)
plot_data_g(data_path_SPDNet,data_path_SPDNetRMLR,epcohs,y_min,y_max,step,legend)

plt.show()
plt.savefig('ACC_RMLR_Rdar_and_HDM05.pdf',dpi=dpi)
a=1

