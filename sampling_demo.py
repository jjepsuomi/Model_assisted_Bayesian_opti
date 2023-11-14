import numpy as np
import bosampler
import importlib
importlib.reload(bosampler)
import matplotlib.pyplot as plt
from bosampler import BOsampler
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
#from utilities import plot_density_histogram


"""
Container for making sure that we deal with 2D data. 
 """
def check_2d_format(arr):
    arr = np.array(arr)
    # Check if the input is already 2D (n,m)
    if len(arr.shape) == 2:
        return arr  # Do nothing if it's already 2D
    # If it's 1D (n,), convert it to (n,1)
    elif len(arr.shape) == 1:
        return arr.reshape(-1, 1)

"""
Gaussian function.
"""
def gaussian(mean=0, std=1, input_x=0):
    gaussian_value = np.exp(-(input_x - mean)**2 / (2 * std**2))
    return gaussian_value

"""
The true function f(x)=y.
"""
def true_function(input_x=0):
    true_function_values = 0.1*input_x + input_x * np.sin(input_x) \
    + gaussian(5.7, 1, input_x) \
    - 2*gaussian(-8, 2, input_x) \
    - gaussian(8, 0.8, input_x) \
    - 5*gaussian(-7.5, 1, input_x) \
    - 3.7*gaussian(-15,2,input_x) \
    - 2*gaussian(-11,2,input_x) \
    - 10*gaussian(-14,1,input_x) \
    + 5*gaussian(-17.5, 3, input_x) \
    - 15*gaussian(-20,5,input_x) \
    + 15*gaussian(-15,2,input_x) \
    + 2*gaussian(-18,1,input_x) \
    - 0.5*gaussian(-15,1,input_x) \
    + np.cos(input_x*2)
    return true_function_values


# STEP 1 : Generate the data set and true function.
print(f'***************************\nStarting analysis\n***************************')
input_x_interval_min = -20
input_x_interval_max = 20
number_y_points_from_true_function = 300
print(f'Generating true function {number_y_points_from_true_function} points from x-interval: [{input_x_interval_min}, {input_x_interval_max}]')
x = np.linspace(input_x_interval_min, input_x_interval_max, number_y_points_from_true_function)
y = true_function(x)
print(f'Shape of x is: {x.shape}')
print(f'Shape of y is: {y.shape}')
x, y = check_2d_format(x), check_2d_format(y)
print(f'New shape of x is: {x.shape}')
print(f'New shape of y is: {y.shape}')

# Define a list of length scale values to search over for RBF and Matern kernels
length_scales = [0.5, 1.0, 1.5]

#plot_density_histogram(y)
# Define the hyperparameter grid
param_grid = {
    'kernel': [RBF(length_scale=l) for l in np.arange(1, 8, 1)],
    'alpha': [np.power(10.0, -x) for x in np.arange(1, 3, 1)],
    'n_restarts_optimizer': [n_restarts for n_restarts in np.arange(1, 10, 1)],
}
print(param_grid['alpha'])
"""
hyperparam_grid=None,
                 X=None,
                 y=None,
                 y_noise_params=None,
                 normalize_data=True,
                 cv_folds=5,
                 sample_size=5):
"""
noise_parameters = {'mean' : 0, 'std' : 1e-5}
bo_sampler = BOsampler(hyperparam_grid=param_grid,
                       X=x,
                       y=y,
                       y_noise_params=noise_parameters,
                       normalize_data=True,
                       cv_folds=4,
                       sample_size=50)
#bo_sampler.fit_response_gpr_model()
#bo_sampler.estimate_utility_function()

#x = np.linspace(input_x_interval_min, input_x_interval_max, 10)
#x = x.reshape(-1, 1)
#inc_probs = bo_sampler.get_inclusion_probabilities(X=x, method='ei')
KL_list = bo_sampler.perform_sampling_comparison(sample_count=30, sampling_iterations=5, sampling_method_list=['srs', 'pu'])
print(KL_list)
# Plotting
plt.figure()  # This line creates a new figure
plt.plot(KL_list[:,0], color='blue', label='SRS')
plt.plot(KL_list[:,1], color='red', label='BO')

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