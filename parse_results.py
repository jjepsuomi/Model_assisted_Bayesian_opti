from joblib import Parallel, delayed, dump, load
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import time




data_files = glob.glob('./analysis_results/*/results/*.joblib')
#data_files = data_files[0:100]
file_count = len(data_files)
method_list = ['srs', 'pu', 'ilcb', 'ei', 'sei']
result_holder = {}
for method in method_list:
    result_holder[method] ={}
    result_holder[method]['total_de_difference'] = np.zeros((file_count, 1))
    result_holder[method]['y_mean_difference'] = np.zeros((file_count, 1))
    result_holder[method]['KLD'] = np.zeros((file_count, 1))



example_data = load(data_files[0])
print(f'The data main keys: {example_data.keys()}')
print(f'Single method keys are: {example_data["srs"].keys()}')
print(f'There in total {file_count} sampling results to be processed.')
y_total = np.sum(example_data['y'])

for idx, data_file in enumerate(data_files):
    print(f'Processing data file: {idx}/{file_count}')
    data = load(data_file)
    for method in result_holder.keys():
        total_diff_val = np.abs(y_total - data[method]['difference_estimator'][0])
        if total_diff_val < 1e15:
            result_holder[method]['total_de_difference'][idx,0] = total_diff_val
        else:
            result_holder[method]['total_de_difference'][idx,0] = np.nan
        mean_diff_val = np.abs(data[method]['mean_true_y'][0] - data[method]['mean_estimated_y'][0])
        result_holder[method]['y_mean_difference'][idx,0] = mean_diff_val
        result_holder[method]['KLD'][idx,0] = data[method]['KLD'][0]


plot_data = {}
plot_data['total_de_difference'] = np.empty((file_count, 0))
plot_data['y_mean_difference'] = np.empty((file_count, 0))
plot_data['KLD'] = np.empty((file_count, 0))
# Concatenating the arrays using a loop
for metric in plot_data.keys():
    for method in result_holder.keys():
        metric_array = result_holder[method][metric]
        """
        contains_nan = np.isnan(metric_array).any()
        if contains_nan:
            print(f"The method {method} array contains NaN values.")
        else:
            print(f"The method {method} array does not contain NaN values.")
        """
        plot_data[metric] = np.concatenate((plot_data[metric], metric_array), axis=1)
    # Count number of rows with NaN values
    rows_with_nan = np.isnan(plot_data[metric]).any(axis=1)
    num_rows_with_nan = np.sum(rows_with_nan)
    print(f'Number of rows with NaN values in metric {metric} is {num_rows_with_nan}')
    # Remove rows with NaN values
    plot_data[metric] = plot_data[metric][~rows_with_nan]
    #plot_data[metric] = plot_data[metric] / float(plot_data[metric].shape[0])


# Create boxplot
for metric in plot_data.keys():
    title_label = ''
    y_label = ''
    if metric == 'total_de_difference':
        title_label = f'Absolute difference between\npopulation total and difference estimator\n{plot_data[metric].shape[0]} sampling simulations'
        y_label = 'Absolute total difference'
    elif metric == 'y_mean_difference':
        title_label = f'Absolute difference between\npopulation mean and estimated mean\n{plot_data[metric].shape[0]} sampling simulations'
        y_label = 'Absolute pop. mean difference'
    elif metric == 'KLD':
        title_label = f'Kullback-Leibler divergence between\ntrue and estimated population distribution\n{plot_data[metric].shape[0]} sampling simulations'
        y_label = 'KL-divergence'
    plt.figure(figsize=(8, 6))
    plt.boxplot(plot_data[metric], meanline=True, showmeans=False, showfliers=False)
    plt.xlabel('Sampling design')
    plt.ylabel(y_label)
    plt.title(title_label)
    plt.grid(True)
    plt.xticks(np.arange(1, 6), [word.upper() for word in method_list])  # Adjust labels accordingly
    plt.savefig(f'{metric}.png', dpi=600)
    #plt.show()
