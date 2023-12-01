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




# STEP 1 : Generate the data set and true function.
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


field_data = pd.read_csv('./data/field_data.csv', sep=';')
RS_data = pd.read_csv('./data/RS_features.csv', sep=';')
combined_data = pd.merge(field_data, RS_data, on='id', how='left')
#combined_data.to_csv('a.csv', sep=';')
data_features = ['h0f', 'h5f', 'h10f', 'h20f', 'h30f', 'h40f','h50f', 'h60f', 'h70f', 'h80f', 'h85f', 'h90f',	'h95f',	'h100f', 'vegf', 'h_mean', 'vkph_ka']
data_set = combined_data[data_features].dropna()
#data_set.to_csv('data_set.csv', sep=';')
data_set_numpy = data_set.values
x = data_set_numpy[:,:-1]
y = data_set_numpy[:,-1]

# Define the hyperparameter grid
param_grid = {
    'kernel': [RBF(length_scale=l) for l in np.arange(1, 4, 1)],
    'alpha': [np.power(10.0, -x) for x in np.arange(1, 3, 1)],
    'n_restarts_optimizer': [25],
}
#param_grid = {
#    'kernel': [Matern(length_scale=l, nu=1.5) for l in np.arange(1, 4, 1)],
#    'alpha': [np.power(10.0, -x) for x in np.arange(1, 3, 1)],
#    'n_restarts_optimizer': [50],
#}

bo_sampler = BOsampler(hyperparam_grid=param_grid,
                       X=x,
                       y=y,
                       y_noise_params={'mean' : 0, 'std' : 1e-5},
                       normalize_data=True,
                       cv_folds=5)
#bo_sampler.plot_target_function()
sampling_data_container = bo_sampler.perform_sampling_comparison(sample_count=500, 
                                                                 sampling_iterations=1, 
                                                                 prior_sample_count=100, 
                                                                 sampling_method_list=['srs', 'pu'])
print(sampling_data_container)
# Plotting
plt.figure()  # This line creates a new figure
plt.plot(sampling_data_container['srs']['KLD'], color='blue', label='SRS')
plt.plot(sampling_data_container['pu']['KLD'], color='red', label='BO')

# Adding legends
plt.legend()
plt.grid(True)
# Show the plot
plt.show()

# STEPS TO DO
"""
1. Get the data
2. Normalize the features
3. Produce model based on the data given to the procedure.
4. Estimate the utility function
4. Produce GPR model for the utility function
5. Given input X, estimate the acquisition values --> transform to inclusion probabilties
6. Illustrate inclusion probabilities on plot. 6 figures:
- GPR fit to data
- Utility function
- EI SEI, LCB and PU figures.
"""