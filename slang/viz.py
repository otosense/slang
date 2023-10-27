"""Tools to make visualizations of slang stuffs"""


# -----------------------------------------------------------------------------
# Comparing two sets of points (like bayes-factors, or llm log probs, etc.)

import matplotlib.pyplot as plt


def plot_distributions(arrays):
    """Plots the distribution of multiple arrays using matplotlib.

    Args:
        arrays (dict): A dictionary of {name: array} pairs.

    Usage:
        plot_distributions({'set1': array1, 'set2': array2, 'set3': array3})
    """
    for name, array in arrays.items():
        plt.hist(array, alpha=0.5, label=name)

    plt.legend(loc='upper right')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Values')
    plt.show()


import seaborn as sns


def seaborn_distributions(arrays):
    """Plots the distribution of multiple arrays using seaborn.

    Args:
        arrays (dict): A dictionary of {name: array} pairs.

    Usage:
        seaborn_distributions({'set1': array1, 'set2': array2, 'set3': array3})
    """
    for name, array in arrays.items():
        sns.kdeplot(array, label=name)

    plt.legend(loc='upper right')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of Values')
    plt.show()


from scipy.stats import ks_2samp


def compare_distributions(arrays):
    """Performs the Kolmogorov-Smirnov test to compare distributions.

    Args:
        arrays (dict): A dictionary of {name: array} pairs.

    Returns:
        dict: A dictionary of p-values for each pair of distributions.

    Usage:
        p_values = compare_distributions({'set1': array1, 'set2': array2, 'set3': array3})
        print(p_values)
    """
    names = list(arrays.keys())
    results = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_i, name_j = names[i], names[j]
            array_i, array_j = arrays[name_i], arrays[name_j]
            _, p_value = ks_2samp(array_i, array_j)
            results[f'{name_i} vs {name_j}'] = p_value

    return results


import matplotlib.pyplot as plt


def plot_whisker(arrays):
    """Plots a box-and-whisker plot for each array in arrays.

    Args:
        arrays (dict): A dictionary of {name: array} pairs.

    Usage:
        plot_whisker({'set1': array1, 'set2': array2, 'set3': array3})
    """
    # Convert the dictionary to a list of arrays and a list of labels
    data = list(arrays.values())
    labels = list(arrays.keys())

    # Create the box-and-whisker plot
    plt.boxplot(data, labels=labels)

    plt.xlabel('Array')
    plt.ylabel('Value')
    plt.title('Box-and-Whisker Plots')
    plt.grid(True)
    plt.show()


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_violins(arrays):
    """Plots a violin plot for each array in arrays using seaborn.

    Args:
        arrays (dict): A dictionary of {name: array} pairs.

    Usage:
        plot_violins({'set1': array1, 'set2': array2, 'set3': array3})

    """
    # Convert the dictionary to a DataFrame for easier plotting with Seaborn
    df = pd.DataFrame(arrays)

    # Melt the DataFrame to have a long-form DataFrame, which works well with Seaborn
    df_melted = df.melt(var_name='Array', value_name='Value')

    # Create the violin plot
    sns.violinplot(x='Array', y='Value', data=df_melted)

    plt.title('Violin Plots')
    plt.grid(True)
    plt.show()
