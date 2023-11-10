import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def kl_divergence(true_distribution, estimated_distribution):
    return entropy(true_distribution, estimated_distribution)

def plot_density_histograms(data_sources, bins=30, legend_labels=None):
    # Set up bins
    bins = np.linspace(min(min(data) for data in data_sources), max(max(data) for data in data_sources), bins)

    # Calculate densities
    densities = []
    for data in data_sources:
        density, _ = np.histogram(data, bins=bins, density=True)
        density[density == 0] = 1e-20 # 
        densities.append(density)

    label, kl_divergence_vs_ref = None, None
    # Plot densities using bar plot
    for i, density in enumerate(densities):
        if i == 0: # Reference data
            label = f'Data {i + 1}' if legend_labels is None else f'{legend_labels[i]}'
            plt.bar(bins[:-1], density, width=bins[1] - bins[0], alpha=0.5, label=label)
        else:
            kl_divergence_vs_ref = round(kl_divergence(densities[0], density), 2)
            label = f'Data {i + 1}' if legend_labels is None else f'{legend_labels[i]}, KL: {kl_divergence_vs_ref}'
            plt.bar(bins[:-1], density, width=bins[1] - bins[0], alpha=0.5, label=label)

    plt.legend()

    # Add labels
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Density Histograms and KL Divergence for Data Sets')
    plt.grid(True)
    # Show the plot
    plt.show()

# Example usage with custom legend labels
np.random.seed(42)


# Incorporate the reference data into the data sources
reference_data = np.random.normal(0, 1, 1000)
data2 = np.random.normal(1, 1.5, 1000)
data3 = np.random.normal(-5, 0.5, 1000)
data_sources = [reference_data, data2, data3]

# Calculate densities
#densities, bins = calculate_densities(data_sources)

legend_labels = ['Reference', 'BO', 'SRS']
plot_density_histograms(data_sources, 30, legend_labels=legend_labels)
