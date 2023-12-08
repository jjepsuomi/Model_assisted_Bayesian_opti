from joblib import Parallel, delayed, dump, load
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

data_set = pd.read_csv('data_set.csv', sep=';')
data_set_numpy = data_set.values
x = data_set_numpy[:,:-1]
y = data_set_numpy[:,-1]
data_files = glob.glob('./results/*.joblib')
#data_files1 = glob.glob('./da/results1/*.joblib')
#data_files = data_files + data_files1
print(len(data_files))
srs, pu, ilcb, ei, sei = [],[], [], [], []
total = np.sum(y)
for data_file in data_files:
    data = load(data_file)
    #print(data.keys())
    #print(data['srs'].keys())
    #srs.append(data['srs']['KLD'][0])
    #pu.append(data['pu']['KLD'][0])
    #ilcb.append(data['ilcb']['KLD'][0])
    #ei.append(data['ei']['KLD'][0])
    #sei.append(data['sei']['KLD'][0])
    #srs.append(np.abs(data['srs']['mean_true_y'][0]-data['srs']['mean_estimated_y'][0]))
    #pu.append(np.abs(data['pu']['mean_true_y'][0]-data['pu']['mean_estimated_y'][0]))
    #ilcb.append(np.abs(data['ilcb']['mean_true_y'][0]-data['ilcb']['mean_estimated_y'][0]))
    #ei.append(np.abs(data['ei']['mean_true_y'][0]-data['ei']['mean_estimated_y'][0]))
    #sei.append(np.abs(data['sei']['mean_true_y'][0]-data['sei']['mean_estimated_y'][0]))
    srs.append(np.abs(total-data['srs']['difference_estimator'][0]))
    pu.append(np.abs(total-data['pu']['difference_estimator'][0]))
    ilcb.append(np.abs(total-data['ilcb']['difference_estimator'][0]))
    ei.append(np.abs(total-data['ei']['difference_estimator'][0]))
    sei.append(np.abs(total-data['sei']['difference_estimator'][0]))
    #print(total-data['sei']['difference_estimator'][0])
    #print(total)


#print(srs)
data = [np.array(srs), np.array(pu), np.array(ilcb), np.array(ei), np.array(sei)]

print(data)
# Generating some sample data
#np.random.seed(10)
#data = [np.random.normal(0, std, 100) for std in range(1, 4)]
#print(data[0].shape)
# Creating the boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(data, meanline=True, showmeans=True)

# Calculating and plotting the mean line for each boxplot
#for i, dataset in enumerate(data, start=1):
#    mean = np.mean(dataset)
#    plt.axhline(mean, color='r', linestyle='dashed', linewidth=1)
#    plt.text(i + 0.1, mean, f'Mean: {mean:.2f}', ha='left', va='center', color='r', fontsize=8)

# Calculating and plotting the mean line for each boxplot
#for i, dataset in enumerate(data, start=1):
#    mean = np.mean(dataset)
#    plt.axhline(mean, color='r', linestyle='dashed', linewidth=1)
#    plt.text(i, mean, f'{mean:.2f}', ha='right', va='center', color='r')
# Custom x-ticks
labels = ['SRS', 'BO-PU', 'BO-ILCB', 'BO-EI', 'BO-SEI']
plt.xticks(np.arange(1, len(labels) + 1), labels)
plt.xlabel('Sampling method')
plt.ylabel('Kullback-Leibler divergence, true dist. vs. estimated dist. (vkph_ka)')
plt.title(f'Sampling simulation ({len(data_files)} repeats), prior data size: 100, sample size: 100')
plt.grid(True)
#plt.legend()
plt.show()
