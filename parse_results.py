from joblib import Parallel, delayed, dump, load
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import time
#data_files = glob.glob('./results/*.joblib')
#data_files1 = glob.glob('./results/results1/*.joblib')
#data_files = data_files + data_files1
#print(len(data_files))
data_files = glob.glob('./analysis_results/*/results/*.joblib')
srs, pu, ilcb, ei, sei = [],[], [], [], []
print(load(data_files[0])['srs'].keys())

for idx, data_file in enumerate(data_files):
    print(f'{idx}/{len(data_files)}')
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
    #pu.append(data['pu']['KLD'][0])
    #ilcb.append(data['ilcb']['KLD'][0])
    #ei.append(data['ei']['KLD'][0])
    #sei.append(data['sei']['KLD'][0])
    srs_val = data['srs']['difference_estimator'][0]
    pu_val = data['pu']['difference_estimator'][0]
    ilcb_val = data['ilcb']['difference_estimator'][0]
    ei_val = data['ei']['difference_estimator'][0]
    sei_val = data['sei']['difference_estimator'][0]
    print(srs_val, pu_val, ilcb_val, ei_val, sei_val, "\n")
    time.sleep(3)


print(srs)
data = [np.array(srs), np.array(pu), np.array(ilcb), np.array(ei), np.array(sei)]


# Generating some sample data
#np.random.seed(10)
#data = [np.random.normal(0, std, 100) for std in range(1, 4)]
#print(data[0].shape)
# Creating the boxplot
plt.boxplot(data)

# Calculating and plotting the mean line for each boxplot
#for i, dataset in enumerate(data, start=1):
#    mean = np.mean(dataset)
#    plt.axhline(mean, color='r', linestyle='dashed', linewidth=1)
#    plt.text(i, mean, f'{mean:.2f}', ha='right', va='center', color='r')

plt.xlabel('Data')
plt.ylabel('Value')
plt.title(f'Boxplot with Mean {len(data_files)}')

plt.show()
