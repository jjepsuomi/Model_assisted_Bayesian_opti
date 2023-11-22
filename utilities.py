import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


def sort_by_x_values(X=None, y=None):
    sorted_indexes = np.argsort(X, axis=0)
    sorted_data_X = X[sorted_indexes, :].reshape(X.shape)
    sorted_data_y = y[sorted_indexes, :].reshape(y.shape)
    return sorted_data_X, sorted_data_y


def kl_divergence(true_distribution=None, estimated_distribution=None):
    return entropy(true_distribution, estimated_distribution)

def calculate_histogram_distances(data_sources=None, num_of_bins=30):
    # Flatten the input data
    flattened_data = [data.flatten() for data in data_sources]

    # Set up bins
    bins = np.linspace(min(min(data) for data in flattened_data), max(max(data) for data in flattened_data), num_of_bins)

    # Calculate densities
    densities = []
    for data in flattened_data:
        #print("NOW", data)
        #print(bins)
        density, _ = np.histogram(data, bins=bins, density=True)
        density[density == 0] = 1e-20
        densities.append(density)

    KL_divergence = []
    label, kl_divergence_vs_ref = None, None

    # Plot densities using bar plot
    for i, density in enumerate(densities):
        #if i == 0:  # Reference data
        #    label = f'{legend_labels[i]}'
        #    plt.bar(bins[:-1], density, width=bins[1] - bins[0], alpha=0.5, label=label)
        #else:
        kl_divergence_vs_ref = round(kl_divergence(densities[0], density), 2)
        KL_divergence.append(kl_divergence_vs_ref)
        #label = f'{legend_labels[i]}, KL: {kl_divergence_vs_ref}'
        #plt.bar(bins[:-1], density, width=bins[1] - bins[0], alpha=0.5, label=label)

    #plt.legend()
    #plt.xlabel('Value')
    #plt.ylabel('Density')
    #plt.title('Density Histograms and KL Divergence for Data Sets')
    #plt.grid(True)
    #plt.show()

    return bins, densities, KL_divergence


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


if __name__ == "__main__":
    
    np.random.seed(42)
    data1 = np.random.randn(100, 1)
    data2 = np.random.randn(100, 1)
    data_sources = [data1, data2]

    #_, _, _ = calculate_histogram_distances(data_sources=data_sources, num_of_bins=30, legend_labels=['Ref', 'Est'])
