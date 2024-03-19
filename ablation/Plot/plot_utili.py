import csv
import numpy as np
import matplotlib.pyplot as plt

def read_cvs_data(data_path):
    data = []
    with open(data_path,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        header_row=next(plots)
        for row in plots:
            data.append(float(row[2])*100)
    return data

def plot_data(epoch,SPDNet,SPDNet_ACM_MLOG,legend,my_y_ticks, y_min, y_max, fontsize=12):
    plt.plot(epoch,SPDNet,label=legend[0])
    plt.plot(epoch,SPDNet_ACM_MLOG,label=legend[1])
    plt.yticks(my_y_ticks,fontsize=fontsize)
    # plt.xticks(my_x_ticks,fontsize=fontsize)
    plt.legend(fontsize = fontsize,loc='lower right')
    plt.xlabel('Training epoch', fontsize = fontsize)
    plt.ylabel('Acc', fontsize = fontsize)
    plt.ylim(y_min,y_max)

def plot_data_g(data_path_SPDNet,data_path_SPDNetMLR,epochs,y_min,y_max,step,legend,dpi=400):
    SPDNet = read_cvs_data(data_path_SPDNet)
    SPDNetRMLR = read_cvs_data(data_path_SPDNetMLR)
    epoch = (range(1, epochs + 1))

    tmp_SPDNet = SPDNet[0:epochs]
    tmp_SPDNetMUL = SPDNetRMLR[0:epochs]
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(which='major', ls='--', alpha=.8, lw=.8)
    my_y_ticks = np.arange(y_min, y_max, step)
    plot_data(epoch, tmp_SPDNet, tmp_SPDNetMUL, legend, my_y_ticks, y_min=y_min, y_max=y_max)