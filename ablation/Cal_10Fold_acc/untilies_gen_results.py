import pandas as pd
import os
import numpy as np
import torch as th
import matplotlib.pyplot as plt

def get_acc(model_path,legend_name='SPDNet',average_range = 10, mode = 'max',details=False,normalizer=1,fold=10,is_plot=False):
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
        elif mode == 'min':
            ith_acc = acc[-average_range:].min()
        if details:
            print(f"{ith + 1:d}-ith fold acc is {ith_acc:.2f}")
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

    mean_acc = array.mean()
    std_acc = array.std()
    max_acc = array.max()
    # Instead of printing, return a dictionary with the necessary data
    base_name = os.path.basename(model_path).split('/')[-1]  # Get the base name of the file
    parts = base_name.split('-')  # Split by '-'
    # time_part_index = next(i for i, part in enumerate(parts) if '[' in part and ']' in part and ',' in part) + 1
    time_part_index=len(parts)
    model_name = '-'.join(parts[:time_part_index - 2])  # Join parts excluding the architecture and time
    architecture = parts[-2]  # The architecture part

    # model_name = os.path.basename(tmp).split('.')[0]# Assuming model_path includes the file extension
    return {
        'model_name': model_name,
        'architecture': architecture,
        'acc_mean': f"{mean_acc:.2f}",
        'acc_std': f"{std_acc:.2f}",
        'max_acc': f"{max_acc:.2f}"
    }
def generate_results_table(data):
    # Parsing model_name to separate columns and combining metrics
    for d in data:
        name = d['model_name']
        architecture = d['architecture']
        d.update({'name': name, 'architecture': architecture})

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Combine 'acc (mean±std)' and 'max_acc' into one column
    df['results'] = df.apply(lambda row: f"{row['acc_mean']}±{row['acc_std']}, max acc: {row['max_acc']}", axis=1)

    # Select relevant columns for the pivot table
    df = df[['name', 'architecture', 'results']]

    # Pivot table to get desired format
    result_table = df.pivot(index='name', columns='architecture', values='results').fillna('')

    # Convert the pivot table to a more visually appealing format if needed
    result_table = result_table.reset_index()
    result_table.columns.name = None  # Remove the hierarchy name

    return result_table


def process_and_generate_table(base_dir, model_names, legend_name='SPDNet', average_range = 10, mode = 'max', details = False, normalizer = 1, fold = 10, is_plot = False):
    data = []
    for name in model_names:
        model_path = os.path.join(base_dir, name)
        base_name = os.path.basename(model_path).split('/')[-1]  # Get the base name of the file
        print(base_name)
        model_data = get_acc(model_path, legend_name=legend_name, average_range=average_range, mode=mode,
                             details=details,normalizer=normalizer,fold=fold,is_plot=is_plot)
        data.append(model_data)

    # Assuming you have defined generate_results_table function as provided earlier
    result_table = generate_results_table(data)
    return result_table

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