from joblib import Parallel, delayed, dump, load
import numpy as np
import pandas as pd
import glob

data_files = glob.glob('./results/*.joblib')

true, srs, bo = [],[], []
for data_file in data_files:
    data = load(data_file)
    #print(data)
    srs.append(data['srs']['mean_estimated_y'])
    bo.append(data['sei']['mean_estimated_y'])
    true.append(data['srs']['mean_true_y'])
print(np.mean(true), np.mean(srs), np.mean(bo))