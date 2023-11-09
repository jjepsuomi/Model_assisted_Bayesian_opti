import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def kl_divergence(p, q):
    return entropy(p, q)

def plot_density_histograms(data_sources, bins=30, legend_labels=None):
    # Set up bins
    bins = np.linspace(min(min(data) for data in data_sources), max(max(data) for data in data_sources), bins)

    # Plot histograms and calculate densities
    densities = []
    for i, data in enumerate(data_sources):
        label = f'Data {i + 1}' if legend_labels is None else legend_labels[i]
        density, _, _ = plt.hist(data, bins=bins, density=True, alpha=0.5, label=label)
        density[density==0]=1e-10
        densities.append(density)

    # Calculate and display KL divergence for data2 and data3
    if len(data_sources) >= 3:
        kl_divergence_2_vs_1 = kl_divergence(densities[1], densities[0])
        kl_divergence_3_vs_1 = kl_divergence(densities[2], densities[0])

        label_2 = f'Data 2 (KL Div: {kl_divergence_2_vs_1:.4f})'
        label_3 = f'Data 3 (KL Div: {kl_divergence_3_vs_1:.4f})'

        plt.legend([label_2, label_3] + (legend_labels[2:] if legend_labels else []))

    else:
        plt.legend()

    # Add labels
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Density Histograms and KL Divergence for Data Sets')

    # Show the plot
    plt.show()

# Example usage with custom legend labels
np.random.seed(42)
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(1, 1.5, 1000)
data3 = np.random.normal(-5, 0.5, 1000)

legend_labels = ['Source A', 'Source Ba', 'Source C']
plot_density_histograms([data1, data2, data3], legend_labels=legend_labels)

print("hello there")
