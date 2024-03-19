import os

import numpy as np
import torch as th

import matplotlib.pyplot as plt

drange = 10
dmode = 'max'
is_plot=0
fold=99
details=True

def get_acc(model_path,legend_name,average_range = drange, mode = dmode,details=False,normalizer=1):
    checkpoint = th.load(model_path)
    results = checkpoint['acc_val_all']
    acc_means = []
    maximum = min(fold,len(results))
    for ith in range(maximum):
        tmp = results[ith]
        if len(tmp)==2:
            acc = np.array(tmp[0])
        else:
            acc = np.array(tmp)
        acc=acc*normalizer
        if mode == 'mean':
            ith_acc = acc[-average_range:].mean()
        elif mode == 'max':
            ith_acc = acc[-average_range:].max()
        if details:
            print("{:d}-ith fold acc is {:.2f}".format(ith + 1, ith_acc))
        acc_means.append(ith_acc)
        acc_means.append(ith_acc)
        if is_plot:
            plt.plot(acc)
            # plt.legend(legend_name+'{:d}th fold'.format(ith))
            plt.legend(legend_name)
    if is_plot:
        plt.ylim(15, 40)
        plt.grid()
        plt.show()
    array = np.array(acc_means)
    print("Final average acc is {:.2f}±{:.2f}, {:.2f}".format(array.mean(), array.std(),array.max()))


def get_file_names(directory_path):
    # Get a list of all file names in the directory, excluding subdirectories
    file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    return file_names

def print_results(model_name,base_dir,details=False,normalizer=1):
    for name in model_name:
        print(name)
        get_acc(os.path.join(base_dir,name),legend_name='SPDNet',details=details,normalizer=normalizer)

def list_files(directory):
    """
    Lists all files in the given directory and returns them as a list
    """
    files = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

plt.close('all')

# base_dir_baslines = '/home/zchen/Proposed_Methods/LieBN/outputs/20Fold/FPHA/torch_resutls'
# file_names = ['0.005-wd_0-SPDNet-AMSGRAD-[63, 33]',
#               '0.005-wd_0-SPDNetBN-AMSGRAD-[63, 33]']
# print_results(file_names,base_dir_baslines,details=details)

base_dir_LieBN = '/home/zchen/Proposed_Methods/RMLR/outputs/10Folds/Radar/torch_resutls'

file_names = list_files(base_dir_LieBN)

# file_names = ['LieNet-euler_axis_angle-hm--SGD-clip5-bs30-0.01-wd_0.0-1024',
#                 '1024-0.005-wd_0-m_0.1-SPDNetLieBN-AMSGRAD-[63, 33]-var-no_grad-w_dfm-LEM-(1.5,1.0,0.0)',
#                 '1024-0.005-wd_0-m_0.1-SPDNetLieBN-AMSGRAD-[63, 33]-var-no_grad-w_dfm-LEM-(1.0,1.0,0.0)',
#                 '1024-0.005-wd_0-m_0.1-SPDNetLieBN-AMSGRAD-[63, 33]-var-no_grad-w_dfm-LEM-(0.5,1.0,0.0)',
#               '1024-0.005-wd_0-m_0.1-SPDNetLieBN-AMSGRAD-[63, 33]-var-no_grad-w_dfm-LCM-(1.0)',
#               '1024-0.005-wd_0-m_0.1-SPDNetLieBN-AMSGRAD-[63, 33]-var-no_grad-w_dfm-LCM-(-0.5)',]

print_results(file_names,base_dir_LieBN,details=details,normalizer=1)
