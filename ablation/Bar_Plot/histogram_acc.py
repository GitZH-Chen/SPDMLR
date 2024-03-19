import matplotlib.pyplot as plt
import numpy as np

# Enable LaTeX rendering in Matplotlib
# plt.rcParams['text.usetex'] = True
dpi = 400
fontsize = 20
num=2
# Sample data (replace with your actual data)
datasets = ['Radar', 'HDM05', 'Hinss21 Inter-session', 'Hinss21 Inter-subject']
model_names = ['SPDNet', 'MLR-$(\\theta,\\alpha,\\beta)$-AIM', 'MLR-$(\\theta,\\alpha,\\beta)$-EM', 'MLR-$(\\theta,\\alpha,\\beta)$-LEM',
               'MLR-$(2\\theta)$-BWM','MLR-$(\\theta)$-LCM']
means=[[93.47, 94.32, 95.11, 95.87, 94.84, 95.16],
 [60.76, 61.14, 70.22, 60.28, 70.2, 65.71],
 [53.83, 55.27, 54.48, 53.51, 55.54, 56.43],
 [49.68, 51.15, 51.38, 51.41, 51.67, 54.14]]
stds=[[0.45, 0.94, 0.82, 0.58, 0.68, 0.67],
 [0.8, 0.94, 0.81, 0.91, 0.91, 0.75],
 [9.77, 8.68, 9.21, 10.02, 7.45, 8.79],
 [7.88, 7.83, 5.77, 7.98, 8.73, 8.36]]

y_lims=[[90, 97],[55,72],[30,70],[30,70]]
step=[2,3]

# Create a figure and axes with 1 row and 4 columns
fig, axes = plt.subplots(1, num, figsize=(10, 6), dpi=dpi)

# Define colors for best parameters with stronger contrast
bar_colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928']

for i, dataset in enumerate(datasets[0:num]):
    ax = axes[i]

    # Sample data for each dataset (replace with your actual data)
    ith_means = means[i];ith_stds = stds[i]

    # Plot bar plots for best parameters with different colors
    for j, model in enumerate(model_names):
        ax.bar(j, ith_means[j], yerr=ith_stds[j], width=0.4, color=bar_colors[j], capsize=5, label=model)
        # Show the model name below each bar vertically
        # ax.text(j, -1, model, rotation='vertical', ha='center', va='top', fontsize=fontsize, color='black')
        ax.set_ylim(y_lims[i])
        ax.set_yticks(np.arange(y_lims[i][0], y_lims[i][1] + 2, step=step[i]))

    # Remove x-axis ticks
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xticks([])
    ax.set_title(dataset, fontsize=fontsize)

axes[0].set_ylabel('Acc', fontsize=fontsize)

# Show a shared legend for all subplots outside the subplots
handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', fancybox=True, shadow=True, ncol=3, fontsize=fontsize, bbox_to_anchor=(0.5, 1.15))
fig.legend(handles, labels, loc='lower center', fancybox=True, shadow=True, ncol=3, fontsize=14)

plt.subplots_adjust(bottom=0.2)
# Adjust layout to prevent clipping of labels
# plt.tight_layout()

# Show the plot
plt.show()
plt.savefig('histogram_radar_hdm05.pdf',dpi=dpi)

