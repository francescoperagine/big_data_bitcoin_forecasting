import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix

from src.utils.constants import *

def plot_correlation_matrix(data_type: str, exchange: str, correlation_matrix: pd.DataFrame):
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title(f'{exchange}-{data_type} correlation matrix of Crypto Market Features')
    plt.savefig(os.path.join(FIGURE_PATH, f'{exchange}_{data_type}_correlation_matrix.png'))
    plt.show()

def plot_feature_importance(data_type: str, exchange: str, explained_variance: np.ndarray, cumulative_variance: np.ndarray):
    components = np.arange(1, len(explained_variance) + 1)

    if data_type == ORDERBOOK:
        plt.figure(figsize=(20, 6))
    else:
        plt.figure(figsize=(6,4))

    bars = plt.bar(components, explained_variance, alpha=0.6, color='g', label='Explained variance')
    plt.plot(components, cumulative_variance, marker='o', linestyle='-', color='r', label='Cumulative variance')

    # Individual explained variance - skip the first bar
    for idx, bar in enumerate(bars):
        yval = bar.get_height()
        if idx != 0:  # Skip the first bar
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', fontsize=8, va='bottom', ha='center')

    # Cumulative explained variance on the line plot
    for i, txt in enumerate(cumulative_variance):
        plt.text(components[i], cumulative_variance[i], f'{txt:.3f}', fontsize=8, ha='right', va='bottom')

    plt.title(f'Variance Explained by feature for {exchange} {data_type}')
    plt.xlabel('Feature')
    plt.ylabel('Variance Explained')
    plt.xticks(components)
    plt.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, f'{exchange}_{data_type}_feature_variance.png'))
    plt.show()

# Plot the PCA loadings
def plot_loadings(data_type: str, exchange: str, loadings: pd.DataFrame):
    if data_type == ORDERBOOK:
        plt.figure(figsize=(15, 15))
    else:
        plt.figure(figsize=(13, 5))
    loadings.plot(kind='bar', width=3)
    plt.title(f'PCA Loadings for {exchange}')
    plt.savefig(os.path.join(FIGURE_PATH, f'{exchange}_{data_type}_pca_loadings.png'))
    plt.show()

# Plot the PCA loadings as a heatmap   
def plot_loadings_heatmap(data_type: str, exchange: str, loadings: pd.DataFrame, component_index: int = 0):
    font_size = 9
    # loadings = loadings.sort_values(by=loadings.columns[0], ascending=False)
    sorted_loadings = loadings.sort_values(by=loadings.columns[component_index], ascending=False)

    # Adjust the figure size dynamically based on the number of features
    height = max(6, min(loadings.shape[0] * 0.5, 20))  
    width = max(10, min(loadings.shape[1] * 1.2, 20))
    annot = False if data_type == ORDERBOOK else True
    
    if data_type == ORDERBOOK:
        plt.figure(figsize=(width, height))
        ax = sns.heatmap(loadings, cmap='coolwarm', annot=annot)
    else:
        plt.figure(figsize=(width, height))
        ax = sns.heatmap(sorted_loadings, cmap='coolwarm', annot=annot, annot_kws={'size': font_size - 2})

    plt.title(f'PCA Loadings for {exchange}', fontsize=font_size + 2)
    plt.xlabel('Principal Components', fontsize=font_size)
    plt.ylabel('Features', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, f'{exchange}_{data_type}_pca_loadings_heatmap.png'))
    plt.show()

# Plot the combined scores as a histogram and density plot
def plot_histogram_density(data_type: str, exchange: str, df: pd.DataFrame, labels: list, colors: list):
    plt.figure(figsize=(12, 8))
    for label, color in zip(labels, colors):
        # Plotting the histogram
        sns.histplot(df[label], color=color, label=f'{label} Histogram', alpha=0.5, edgecolor='k', bins=30)
        # Plotting the density
        sns.kdeplot(df[label], color=color, label=f'{label} Density', lw=3)
    
    plt.title('Histogram and Density Plots of Feature Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend(title='Legend')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_PATH, f'{exchange}_{data_type}_combined_scores_histogram_density.png'))
    plt.show()

# Plot the elbow plot to find the optimal number of features
def plot_elbow_curve(data_type: str, exchange: str, combined_scores, elbow_index):
    """Plot the sorted scores and highlight the elbow point."""
    
    sorted_scores = combined_scores.iloc[:, 0].sort_values(ascending=False).values
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_scores, 'o-', label='Combined Scores')
    if elbow_index > 0 and elbow_index < len(sorted_scores):  # Check if elbow_index is valid
        plt.plot(elbow_index, sorted_scores[elbow_index], 'ro', label='Elbow Point')
        plt.annotate('Elbow', xy=(elbow_index, sorted_scores[elbow_index]),
                     xytext=(elbow_index + 10, sorted_scores[elbow_index] - 0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    plt.title('Elbow Method to Determine Optimal Number of Features')
    plt.xlabel('Feature Index')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_PATH, f'{exchange}_{data_type}_combined_scores_elbow_plot.png'))
    plt.show()

def plot_confusion_matrix(exchange: str, data_type: str, folder: str, cm, labels):
    path = os.path.join(FIGURE_PATH, folder)
    if not os.path.exists(path):
        os.makedirs(path)
    # cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(path, f'{exchange}_{data_type}_confusion_matrix.png'))
    plt.show()