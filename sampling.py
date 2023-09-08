import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def gaussian(mean=0, std=1, input_x=0):
    gaussian_value = np.exp(-(input_x - mean)**2 / (2 * std**2))
    return gaussian_value

def true_function(input_x=0):
    #true_function_values = input_x * np.sin(input_x) + gaussian(5.7, 1, 5, input_x) - gaussian(-8, 2, 40, input_x) - gaussian(8, 0.8, 3, input_x) - gaussian(-4, 1.4, 10, input_x)
    return input_x * np.sin(input_x) + gaussian(5.7, 1, input_x) - 2*gaussian(-8, 2, input_x) - gaussian(8, 0.8, input_x) - 5*gaussian(-7.5, 1, input_x)


def get_random_sample_from_true_function(xmin=-10, xmax=10, num_samples=1):
    random_x = np.random.uniform(xmin, xmax, num_samples)
    random_y = true_function(random_x)
    random_y_with_noise = random_y + np.random.normal(0, 0.00001, size=(num_samples,))
    print(random_y[0:3], random_y_with_noise[0:3])
    
    #np.random.normal(0, 0.001, size=(num_samples,))
    #random_y = true_function(random_x) 
    return random_x, random_y_with_noise

def fit_gaussian_process(X_train, y_train):
    # Define the kernel (Radial Basis Function kernel)
    #X_train = X_train.reshape(-1, 1) # Only one feature
    kernel = RBF(length_scale=2) + C(0.45, (1e-3, 1e3))
    # Create the Gaussian Process Regressor model with the defined kernel
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10) 
    # Fit the model to the training data
    model.fit(X_train, y_train)
    return model

def predictive_uncertainty(input_x, y_mean, y_std):
    max_std_ind = np.where(y_std == np.max(y_std))[0][0]
    return input_x[max_std_ind], y_mean[max_std_ind]

# Generate the data set
input_x_interval_min = -10
input_x_interval_max = 10
number_y_points_from_true_function = 100
print(f'Generating true function {number_y_points_from_true_function} points from x-interval: [{input_x_interval_min}, {input_x_interval_max}]')
x = np.linspace(input_x_interval_min, input_x_interval_max, number_y_points_from_true_function)
y = true_function(x)
print(f'Shape of x is: {x.shape}')
print(f'Shape of y is: {y.shape}')

# Make sample data
sample_x, sample_y = get_random_sample_from_true_function(-10, 10, 12)
sample_x = sample_x.reshape(-1, 1)
gpr_model = fit_gaussian_process(sample_x.reshape(-1,1), sample_y)
# Next, predict the function
predicted_y_mean, predicted_y_std = gpr_model.predict(X=x.reshape(-1,1), return_std=True)
# Get the predictive uncertainty point
xopt, yopt = predictive_uncertainty(x, y, predicted_y_std)
# Create the plot of true function
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
plt.plot(x, y, label='target function', color='blue', linestyle='--')  # Plot the sine function
plt.plot(x, predicted_y_mean, label='GPR', color='green')  # Plot the sine function
plt.fill_between(x, predicted_y_mean - 1.96*predicted_y_std, predicted_y_mean + 1.96*predicted_y_std, color='orange', alpha=0.2)
#plt.title('Plot of the Sine Function')
plt.xlabel('Explanatory variable value')
plt.ylabel('Function value')

# Plot each sample as a circle
for xi, yi in zip(sample_x, sample_y):
    circle = plt.Circle((xi, yi), 0.2, color='red', alpha=0.5)
    plt.gca().add_patch(circle)


circle = plt.Circle((xopt, yopt), 0.2, color='violet', alpha=0.8)
plt.gca().add_patch(circle)


# Set the aspect ratio to be equal
plt.axis('equal')
# Set the y-range (limits)
plt.ylim(-10, 10)  # Replace these values with your desired y-range
#plt.xlim(-10, 10)  # Replace these values with your desired y-range
plt.grid(True)
plt.legend()
plt.show()
