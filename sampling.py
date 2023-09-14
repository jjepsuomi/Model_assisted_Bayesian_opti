import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import chi2, norm
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut, KFold
import warnings
from sklearn.exceptions import ConvergenceWarning
import copy
# Disable ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

"""
Generate a sample from the true function with added Gaussian noise.
"""
def get_random_sample_from_true_function(xmin=-10, xmax=10, num_samples=1, noise_std=0.00001):
    random_x = np.random.uniform(xmin, xmax, num_samples)
    sample_sort_indices = np.argsort(random_x, axis=0) # Sort the x into ascending order (makes plotting nicer)
    random_x = random_x[sample_sort_indices]
    random_y = true_function(random_x)
    random_y_with_noise = random_y + np.random.normal(0, noise_std, size=(num_samples,))
    return random_x.reshape(-1, 1), random_y_with_noise.reshape(-1, 1) # Return and make sure they are 2D-matrices with one column.

"""
Initiate an unfitted Gaussian Process Regressor model.
"""
def initiate_gpr_model():
    kernel = RBF(length_scale=1) + C(0.05, (1e-3, 1e3))
    # Create the Gaussian Process Regressor model with the defined kernel
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10) 
    return model

"""
Fit a new GPR model into given data.
"""
def fit_gaussian_process(X_train=None, y_train=None):
    model = initiate_gpr_model()
    model.fit(X_train, y_train)
    return model


"""
Evaluate sample point utilities.
"""
def evaluate_single_point_impact(x_data=None, y_data=None, nfolds=5):
    loo = LeaveOneOut()
    loo_ind = 1
    point_utility_list = np.zeros(shape=(len(y_data), 1))
    for inner_data_index, single_point_index in loo.split(x_data):
        sample_data_x, single_point_x = x_data[inner_data_index], x_data[single_point_index]
        sample_data_y, single_point_y = y_data[inner_data_index], y_data[single_point_index]
        kfcv = KFold(n_splits=nfolds, shuffle=True)
        print(f'Evaluating sample point {loo_ind}/{len(y_data)}')
        without_point_residual, with_point_residual = 0, 0
        for inner_train_index, inner_test_index in kfcv.split(sample_data_x):
            train_x, test_x = sample_data_x[inner_train_index], sample_data_x[inner_test_index]
            train_y, test_y = sample_data_y[inner_train_index], sample_data_y[inner_test_index]
            # Vertically stack the arrays
            train_x_with_new_point = np.vstack((train_x, single_point_x))
            train_y_with_new_point = np.vstack((train_y, single_point_y))
            model_without_point = initiate_gpr_model()
            model_without_point.fit(X=train_x, y=train_y)
            model_with_point = initiate_gpr_model()
            model_with_point.fit(X=train_x_with_new_point, y=train_y_with_new_point)
            predicted_y_without_point = model_without_point.predict(test_x)
            predicted_y_with_point = model_with_point.predict(test_x)
            without_point_residual += np.sum(np.abs(predicted_y_without_point - test_y)) / len(test_y)
            with_point_residual += np.sum(np.abs(predicted_y_with_point - test_y)) / len(test_y)
        without_point_residual = without_point_residual / nfolds
        with_point_residual = with_point_residual / nfolds
        point_utility_list[single_point_index] = with_point_residual - without_point_residual
        if point_utility_list[single_point_index] < 0:
            print(f'Point addition improved test error by: {point_utility_list[single_point_index]}')
        else:
            print(f'Point addition did not improve test error, but increased: {point_utility_list[single_point_index]}')
        loo_ind += 1
    return point_utility_list.reshape(-1, 1) # Return the utility values as 2D-matrix.
            

def expected_improvement(x_data, surrogate_model, f_min, var_epsilon=1e-10):
    mu, sigma = surrogate_model.predict(X=x_data, return_std=True)  # Get mean and standard deviation from the surrogate model.
    # Calculate the standard score Z with a small epsilon to avoid division by zero.
    sigma[sigma == 0] = var_epsilon # In places where sigma is 0, set sigma to small value to prevent division by zero.
    Z = (f_min - mu) / sigma
    # Calculate the Expected Improvement.
    phi = norm.pdf(Z)
    Phi = norm.cdf(Z)
    ei = (f_min - mu) * Phi + sigma * phi
    return ei

def scaled_expected_improvement(x_data, surrogate_model, f_min, var_epsilon=1e-10):
    mu, sigma = surrogate_model.predict(X=x_data, return_std=True)  # Get mean and standard deviation from the surrogate model.
    # Calculate the standard score Z with a small epsilon to avoid division by zero.
    sigma[sigma == 0] = var_epsilon # In places where sigma is 0, set sigma to small value to prevent division by zero.
    Z = (f_min - mu) / sigma
    # Calculate the Expected Improvement.
    phi = norm.pdf(Z)
    Phi = norm.cdf(Z)
    ei = (f_min - mu) * Phi + sigma * phi
    # Next EI variance
    ei2 = np.power(ei, 2)
    Z2 = np.power(Z, 2)
    sigma2 = np.power(sigma, 2)
    ei_std = np.sqrt(sigma2 * ((Z2 + 1)*Phi + Z*phi) - ei2)
    ei_std[ei_std == 0] = var_epsilon
    sei = ei / ei_std
    return sei

def predictive_uncertainty(x_data, y_mean, y_std):
    max_std_ind = np.where(y_std == np.max(y_std))[0][0]
    return x_data[max_std_ind], y_mean[max_std_ind]

def inverted_lower_confidence_bound(y_mean=0, y_std=0, l=0.2):
    return l*y_std - y_mean

"""
Minmax-normalization.
"""
def acquisition_to_inclusion_probs(acquisition_values=None):
    max_acquisition_value = np.max(acquisition_values)
    min_acquisition_value = np.min(acquisition_values)
    divisor = max_acquisition_value - min_acquisition_value
    normalized_acquisition_values = None
    if divisor == 0: # If this happens, all acquisition values are same. In this case use random sampling with same prob. 1/N
        print(f'The maximum and minimum  acquisition values are equal, setting inc.probs to: {1.0/acquisition_values.size}')
        normalized_acquisition_values = np.full(shape=acquisition_values.shape, fill_value=1.0/acquisition_values.size)
    else: # divisor > 0
        normalized_acquisition_values = (acquisition_values - min_acquisition_value) / divisor # p € [0, 1]
        unique_acquisition_values = np.unique(normalized_acquisition_values)
        if unique_acquisition_values.size > 2:
            min_prob = unique_acquisition_values[0] + (unique_acquisition_values[1] - unique_acquisition_values[0]) / 2
            max_prob = unique_acquisition_values[-2] + (unique_acquisition_values[-1] - unique_acquisition_values[-2]) / 2
            normalized_acquisition_values[normalized_acquisition_values == 1] = max_prob
            normalized_acquisition_values[normalized_acquisition_values == 0] = min_prob # p € (0, 1)
        else: # This means there must be two unique values
            print(f'There are only two unique acquisition values, setting to 0.2 and 0.8.')
            normalized_acquisition_values[normalized_acquisition_values == 1] = 0.8
            normalized_acquisition_values[normalized_acquisition_values == 0] = 0.2 # p € (0, 1)
    return normalized_acquisition_values


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


# STEP 2: Create a sample from the true functrion
sample_count = 8
random_sample_y_noise_std = 0.0000001
print(f'Sampling {sample_count} points from true function with noise std. level: {random_sample_y_noise_std}')
sample_x, sample_y = get_random_sample_from_true_function(xmin=input_x_interval_min,
                                                          xmax=input_x_interval_max, 
                                                          num_samples=sample_count,
                                                          noise_std=random_sample_y_noise_std)
print(f'Random (x-sorted) sample generated with data shapes x,y: {sample_x.shape}, {sample_y.shape}')


# STEP 3: Evaluate the impact of all data points to model test performance.
number_of_Monte_Carlo_evaluations = 5
print(f'Evaluating sample point utilities using {number_of_Monte_Carlo_evaluations}-fold cross-validation.')
sample_y_utility_values = evaluate_single_point_impact(x_data=sample_x,
                                                       y_data=sample_y,
                                                       nfolds=number_of_Monte_Carlo_evaluations)

# STEP 4: Fit GPR model to sample data and make the data to plot the GPR function fit.
gpr_model = fit_gaussian_process(X_train=sample_x,
                                 y_train=sample_y)
# Next, predict the function
predicted_y_mean, predicted_y_std = gpr_model.predict(X=x, return_std=True)

print(f'Fitting GPR model to y-utility value.')
utility_gpr_model = fit_gaussian_process(X_train=sample_x,
                                         y_train=sample_y_utility_values)


# Get the predictive uncertainty point
xopt, yopt = predictive_uncertainty(x, predicted_y_mean, predicted_y_std)
pu_ip = acquisition_to_inclusion_probs(predicted_y_std)

# Expected improvement
min_utility_y = np.min(sample_y_utility_values)
print(f'Minimum utility value is: {min_utility_y}')
eis = expected_improvement(x, utility_gpr_model, min_utility_y, var_epsilon=1e-10)
neis = acquisition_to_inclusion_probs(eis)

seis = scaled_expected_improvement(x, utility_gpr_model, min_utility_y, var_epsilon=1e-10)
nseis = acquisition_to_inclusion_probs(seis)

# Lower confidence bound
ilcb01 = inverted_lower_confidence_bound(y_mean=predicted_y_mean, y_std=predicted_y_std, l=0.1)
ilcb01p = acquisition_to_inclusion_probs(ilcb01)
ilcb05 = inverted_lower_confidence_bound(y_mean=predicted_y_mean, y_std=predicted_y_std, l=0.5)
ilcb05p = acquisition_to_inclusion_probs(ilcb05)
ilcb09 = inverted_lower_confidence_bound(y_mean=predicted_y_mean, y_std=predicted_y_std, l=0.9)
ilcb09p = acquisition_to_inclusion_probs(ilcb09)

"""
# Create subplots with two columns
fig, axs = plt.subplots(1, 3, figsize=(12, 6))

# Left subplot (existing code)
axs[0].plot(x.ravel(), y, label='target function', color='blue', linestyle='--')
axs[0].plot(x.ravel(), predicted_y_mean, label='GPR', color='green')
axs[0].fill_between(x.ravel(), predicted_y_mean - 1.96*predicted_y_std, predicted_y_mean + 1.96*predicted_y_std, color='orange', alpha=0.2)
axs[0].set_xlabel('Explanatory variable value')
axs[0].set_ylabel('Function value')
axs[0].scatter(sample_x, sample_y, color="red", label="sample point")
axs[0].set_aspect('equal')
axs[0].set_ylim(-22, 20)  # Replace these values with your desired y-range
axs[0].grid(True)
axs[0].legend()

# Right subplot (simple sinusoid function)
#axs[1].plot(sample_x, y_utility, label='utility', color='blue')
axs[1].plot(x.ravel(), y, label='target function', color='blue', linestyle='--')
axs[1].plot(sample_x, sample_y_utility_values, label='function', color='red', linestyle='--')
print(sample_x)
axs[1].scatter(sample_x, sample_y_utility_values, color="violet", label="sample point")
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_title('Simple Sinusoid Function')
axs[1].grid(True)
axs[1].set_aspect('equal')
axs[1].legend()

# Left subplot (existing code)
axs[2].plot(x.ravel(), eis, label='target function', color='blue', linestyle='-')
axs[2].set_xlabel('Explanatory variable value')
axs[2].set_ylabel('Function value')
#axs[2].set_aspect('equal')
#axs[2].set_ylim(-1, 1)  # Replace these values with your desired y-range
axs[2].grid(True)
axs[2].legend()

# Adjust the layout to prevent overlapping labels
#plt.tight_layout()

# Show the subplots
plt.show()
"""
# Create a figure for the first plot
fig1 = plt.figure(figsize=(6, 6))
plt.plot(x.ravel(), y, label='target function', color='blue', linestyle='--')
plt.plot(x.ravel(), predicted_y_mean, label='GPR fit', color='green')
plt.fill_between(x.ravel(), predicted_y_mean - 1.96*predicted_y_std, predicted_y_mean + 1.96*predicted_y_std, color='orange', alpha=0.2)
plt.xlabel('Explanatory variable value')
plt.ylabel('Function value')
plt.scatter(sample_x, sample_y, color="red", label="sample point")
plt.gca().set_aspect('equal')
plt.ylim(-22, 20)  # Replace these values with your desired y-range
plt.grid(True)
plt.title('Target function and GPR fit')
plt.legend()

# Create a figure for the second plot
fig2 = plt.figure(figsize=(6, 6))
plt.plot(x.ravel(), y, label='target function', color='blue', linestyle='--')
plt.plot(sample_x, sample_y_utility_values, label='utility function', color='violet', linestyle='-')
plt.scatter(sample_x, sample_y_utility_values, color="red", label="utility sample point")
plt.xlabel('Explanatory variable value')
plt.ylabel('Function value')
plt.title('Target and sample utility function.')
plt.grid(True)
plt.gca().set_aspect('equal')
plt.legend()

# Create a figure for the third plot
# Create a figure for PU
fig_pu = plt.figure(figsize=(6, 6))
plt.plot(x.ravel(), pu_ip, label='PU', color='blue', linestyle='--')
plt.plot(x.ravel(), ilcb05p, label='ILCB', color='red', linestyle='-', linewidth=1)
plt.xlabel('Explanatory variable value')
plt.ylabel('Function value')
plt.grid(True)
plt.legend()

# Create a figure for SEI
fig_sei = plt.figure(figsize=(6, 6))

plt.plot(x.ravel(), neis, label='EI', color='orange', linestyle='-')
plt.plot(x.ravel(), nseis, label='SEI', color='green', linestyle='--')
plt.xlabel('Explanatory variable value')
plt.ylabel('Function value')
plt.grid(True)
plt.legend()



# Show the figures
plt.show()