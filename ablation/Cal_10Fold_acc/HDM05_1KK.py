from ablation.Cal_10Fold_acc.untilies_gen_results import list_files,process_and_generate_table
import pandas as pd

drange = 10
dmode = 'max'
is_plot=0
fold=10
details=True
normalizer=1

import os
import numpy as np
import torch as th
import matplotlib.pyplot as plt

# Use this function to process all files and then generate the results table using the previously provided function

# Example usage:
base_dir = "/data/zchen/Released_Code/SPDMLR/outputs/10Folds/uniform/HDM05/torch_resutls"
# base_dir = "/home/zchen/Proposed_Methods/RMLR/outputs/HDM05_1KK/torch_resutls"

model_names = list_files(base_dir)  # Assuming list_files is defined as before and correctly lists all model filenames
model_names.sort()

result_table = process_and_generate_table(base_dir, model_names, average_range = drange, mode = dmode, details = details, normalizer = normalizer, fold = fold, is_plot = is_plot)
pd.set_option('display.max_rows', None)      # or set a specific large number if None is too much
pd.set_option('display.max_columns', None)   # to ensure all columns are displayed
pd.set_option('display.width', None)         # to ensure the total width of the table fits the screen
pd.set_option('display.max_colwidth', None)  # to ensure full content of each cell is displayed

# Now when you print, it should show the entire DataFrame

print(result_table.sort_values(by='name',ascending=False))
# plt.close('all')

