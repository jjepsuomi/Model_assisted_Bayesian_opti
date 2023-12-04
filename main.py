import numpy as np
import pandas as pd
import bosampler
import importlib
importlib.reload(bosampler)
import matplotlib.pyplot as plt
from bosampler import BOsampler
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
#from utilities import plot_density_histogram
from utilities import true_function, check_2d_format
import sys, getopt
from joblib import Parallel, delayed, dump, load
import time

"""
# STEP 1 : Generate a synthetic data set and true function.
print(f'***************************\nStarting analysis\n***************************')
input_x_interval_min = -20
input_x_interval_max = 20
number_y_points_from_true_function = 200
print(f'Generating true function {number_y_points_from_true_function} points from x-interval: [{input_x_interval_min}, {input_x_interval_max}]')
x = np.linspace(input_x_interval_min, input_x_interval_max, number_y_points_from_true_function)
y = true_function(x)
print(f'Shape of x is: {x.shape}')
print(f'Shape of y is: {y.shape}')
x, y = check_2d_format(x), check_2d_format(y)
print(f'New shape of x is: {x.shape}')
print(f'New shape of y is: {y.shape}')
"""

print(f'Starting sampling analysis...')

sample_methods = None
sampling_repeats = None
prior_sample_count = 100
sample_count = 50
sampling_iterations = 1
job_id = 1
results_path = './'

opts, args = getopt.getopt(sys.argv[1:], "i", ["sample_methods =",
                                               "sampling_repeats =",
                                               "prior_sample_count =",
                                               "sample_count =",
                                               "sampling_iterations =",
                                               "job_id =",
                                               "results_path ="])
print("\nReading input arguments...\n")
for argument_type, argument_value in opts:
    argument_type = argument_type.strip()
    argument_value = argument_value.strip()
    print(argument_type, argument_value)
    if argument_type in ("--results_path"):
        results_path = str(argument_value)
    elif argument_type in ("--job_id"):
        job_id = int(argument_value)
    elif argument_type in ("--sampling_iterations"):
        sampling_iterations = int(argument_value)
    elif argument_type in ("--sample_count"):
        sample_count = int(argument_value)
    elif argument_type in ("--prior_sample_count"):
        prior_sample_count = int(argument_value)
    elif argument_type in ("--sampling_repeats"):
        sampling_repeats = int(argument_value)
    elif argument_type in ("--sample_methods"):
        sample_methods = argument_value.split(':')


#combined_data.to_csv('a.csv', sep=';')
input_features = ['h0f', 'h5f', 'h10f', 'h20f', 'h30f', 'h40f','h50f', 'h60f', 'h70f', 'h80f', 'h85f', 'h90f',	'h95f',	'h100f', 'vegf', 'h_mean']
output_feature = ['vkph_ka']
data_set = pd.read_csv('./data_set.csv', sep=';')
data_set_numpy = data_set.values
x = data_set_numpy[:,:-1]
y = data_set_numpy[:,-1]

# Define the hyperparameter grid
param_grid = {
    'kernel': [RBF(length_scale=l) for l in [0.1, 0.5, 1.0, 2.0, 5.0]],
    'alpha': [np.power(10.0, -x) for x in np.arange(1, 3, 1)],
    'n_restarts_optimizer': [25],
}
#param_grid = {
#    'kernel': [Matern(length_scale=l, nu=1.5) for l in [0.1, 0.5, 1.0, 2.0, 5.0]],
#    'alpha': [np.power(10.0, -x) for x in np.arange(1, 3, 1)],
#    'n_restarts_optimizer': [50],
#}
noise_params = {'mean' : 0, 'std' : 1e-5}
noise_params = None

#analysis_container = {}
for sampling_analysis_id in range(0, sampling_repeats):
    print(f'Performing sampling {sampling_analysis_id+1}/{sampling_repeats}, ID: {job_id}')
    t = time.time()
    bo_sampler = BOsampler(hyperparam_grid=param_grid,
                        X=x,
                        y=y,
                        y_noise_params=noise_params,
                        normalize_data=True,
                        cv_folds=5)
    sampling_data_container = bo_sampler.perform_sampling_comparison(sample_count=sample_count, 
                                                                    sampling_iterations=sampling_iterations, 
                                                                    prior_sample_count=prior_sample_count, 
                                                                    sampling_method_list=sample_methods)
    #analysis_container[sampling_analysis_id] = sampling_data_container
    print(f'Sampling analysis {sampling_analysis_id+1}/{sampling_repeats} took: {time.time()-t} seconds')
    dump(sampling_data_container, f'{results_path}analysis_data_samplingid_{sampling_analysis_id+1}_jobid_{job_id}.joblib')
print(f'Sampling analysis finished! =) Results done.')

    


