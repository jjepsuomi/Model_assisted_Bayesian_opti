import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def kl_divergence(true_distribution=None, estimated_distribution=None):
    return entropy(true_distribution, estimated_distribution)

def calculate_histogram_distances(data_sources=None, num_of_bins=30, legend_labels=None):
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

    KL_divergence = [0]
    label, kl_divergence_vs_ref = None, None

    # Plot densities using bar plot
    for i, density in enumerate(densities):
        if i == 0:  # Reference data
            label = f'{legend_labels[i]}'
            plt.bar(bins[:-1], density, width=bins[1] - bins[0], alpha=0.5, label=label)
        else:
            kl_divergence_vs_ref = round(kl_divergence(densities[0], density), 2)
            KL_divergence.append(kl_divergence_vs_ref)
            label = f'{legend_labels[i]}, KL: {kl_divergence_vs_ref}'
            plt.bar(bins[:-1], density, width=bins[1] - bins[0], alpha=0.5, label=label)

    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Density Histograms and KL Divergence for Data Sets')
    plt.grid(True)
    #plt.show()

    return bins, densities, KL_divergence

if __name__ == "__main__":
    
    np.random.seed(42)
    data1 = np.random.randn(100, 1)
    data2 = np.random.randn(100, 1)
    data_sources = [data1, data2]

    _, _, _ = calculate_histogram_distances(data_sources=data_sources, num_of_bins=30, legend_labels=['Ref', 'Est'])
