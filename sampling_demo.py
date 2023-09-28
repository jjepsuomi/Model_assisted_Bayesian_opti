import numpy as np
import bosampler
import importlib
importlib.reload(bosampler)
from bosampler import BOsampler
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
number_y_points_from_true_function = 100
print(f'Generating true function {number_y_points_from_true_function} points from x-interval: [{input_x_interval_min}, {input_x_interval_max}]')
x = np.linspace(input_x_interval_min, input_x_interval_max, number_y_points_from_true_function)
y = true_function(x)
print(f'Shape of x is: {x.shape}')
print(f'Shape of y is: {y.shape}')
x, y = x.reshape(-1, 1), y.reshape(-1, 1)
print(f'New shape of x is: {x.shape}')
print(f'New shape of y is: {y.shape}')

y_noise = {
    'mean' : 0,
    'std'  : 1e-5    
}
bo_sampler = BOsampler(hyperparam_grid=None,
                       X=x,
                       y=y,
                       y_noise_params=y_noise)


