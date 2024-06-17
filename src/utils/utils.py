from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
from src.visualization.visualize import plot_correlation_matrix
from scipy.stats import ttest_ind
from sklearn.utils import resample

def get_features(df: pd.DataFrame) -> list:
    """Return a list of features in the DataFrame."""	
    return [x for x in df.columns]

def get_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return the rows with null values."""	
    return df[df['null'] == True]

def add_time_features(df):
    updated_df = df.copy()
    updated_df['hour'] = updated_df['origin_time'].dt.hour
    updated_df['day_of_week'] = updated_df['origin_time'].dt.dayofweek
    updated_df['day_of_month'] = updated_df['origin_time'].dt.day
    updated_df['month'] = updated_df['origin_time'].dt.month

    # Cyclic representation of time features
    updated_df['hour_sin'] = np.sin(2 * np.pi * updated_df['hour'] / 24)
    updated_df['hour_cos'] = np.cos(2 * np.pi * updated_df['hour'] / 24)
    updated_df['day_of_week_sin'] = np.sin(2 * np.pi * updated_df['day_of_week'] / 7)
    updated_df['day_of_week_cos'] = np.cos(2 * np.pi * updated_df['day_of_week'] / 7)
    return updated_df

def add_lag_features(df, lags):
    updated_df = df.copy()
    for lag in lags:
        updated_df[f'prev_change_lag_{lag}'] = updated_df['close'] - updated_df['close'].shift(lag)
        updated_df.dropna(inplace=True)
    return updated_df

def get_dataframe_null_summary(df: pd.DataFrame, name: str) -> dict:
    """Return a summary of the DataFrame including the total entries, null entries, null percentage, and zero count."""	
    null_df = get_null_values(df)
    total_entries = len(df)
    null_entries = len(null_df)
    null_percentage = null_entries / total_entries if total_entries > 0 else 0

    # Calculate zero count
    zero_count = (df == 0).sum()
    
    return {
        'Exchange': name,
        'Total Entries': total_entries,
        'Null Entries': null_entries,
        'Null Percentage (%)': round(null_percentage, 5),
        **zero_count.to_dict()  # Expand the zero_count Series into the dictionary
    }

def evaluate_correlation(df, exchange, data_type, threshold=0.9):
    scaled_no_origin_time = df.drop(columns=['origin_time'])
    plot_correlation_matrix(data_type, exchange, scaled_no_origin_time.corr())

    highly_correlated_pairs = df.corr().unstack().sort_values(kind="quicksort", ascending=False)
    highly_correlated_pairs = highly_correlated_pairs[(highly_correlated_pairs != 1) & (highly_correlated_pairs > threshold)]

    print("Highly correlated pairs:")
    (highly_correlated_pairs)

def standard_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Scale the DataFrame using StandardScaler."""	
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

def compute_pca(df, variance_threshold = 0.95, n_components = None):

    # Perform PCA to get the explained variance and cumulative variance
    pca_fit = PCA().fit(df)
    explained_variance = pca_fit.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    if n_components:
        num_components = n_components
    else:
        # Get the number of components that explain the variance threshold
        num_components = np.where(cumulative_variance >= variance_threshold)[0][0] + 1

    # Perform PCA with the number of components
    pca_reduced = PCA(n_components=num_components).fit(df)

    # Compute the loadings
    loadings = compute_loadings(pca_reduced, df)

    # Transform the data
    pca_transformed = pca_reduced.transform(df)

    pca_df = pd.DataFrame(pca_transformed)
    pca_df.columns = [f"PC{i+1}" for i in range(pca_df.shape[1])]

    return pca_df, explained_variance[:num_components], cumulative_variance[:num_components], loadings

def compute_loadings(pca: PCA, df: pd.DataFrame) -> pd.DataFrame:
    """Compute the PCA loadings and return them as a DataFrame."""
    return pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=df.columns)

def merge_datasets(df1, df2, on='origin_time'):
    return pd.merge(df1, df2, on=on, how='inner')

def add_orderbook_value_feature(df, orders=20):
    """Add price * size product features to orderbook DataFrame."""
    new_df = df.copy()
    for i in range(orders):
        new_df.insert(new_df.columns.get_loc(f'bid_{i}_size') + 1, f'bid_{i}_value', 0) 
        new_df[f'bid_{i}_value'] = new_df[f'bid_{i}_price'] * new_df[f'bid_{i}_size']
        new_df[f'ask_{i}_value'] = new_df[f'ask_{i}_price'] * new_df[f'ask_{i}_size']
    return new_df

def perform_ttest(df, metric):
    results_list = []
    for data_type in df['data_type'].unique():
        data_type_df = df[df['data_type'] == data_type]
        
        exchanges = data_type_df['exchange'].unique()
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                exchange1 = exchanges[i]
                exchange2 = exchanges[j]
                
                scores1 = np.array(data_type_df[data_type_df['exchange'] == exchange1][metric].values[0])
                scores2 = np.array(data_type_df[data_type_df['exchange'] == exchange2][metric].values[0])
                
                t_stat, p_value = ttest_ind(scores1, scores2, equal_var=False)
                
                results_list.append({
                    'data_type': data_type,
                    'exchange1': exchange1,
                    'exchange2': exchange2,
                    'metric': metric,
                    't_stat': t_stat,
                    'p_value': p_value
                })
    
    return pd.DataFrame(results_list)

def bootstrap(data, n_iterations=1000, sample_size=None):
    if sample_size is None:
        sample_size = len(data)
    means = []
    for _ in range(n_iterations):
        sample = resample(data, n_samples=sample_size)
        means.append(np.mean(sample))
    return means

# Compare mean test scores between different exchanges for the same data type using bootstrapping
def compute_comparison(df, data_type):
    results_list = []
    
    df_filtered = df[df['data_type'] == data_type]
    
    exchanges = df_filtered['exchange'].unique()
    for i in range(len(exchanges)):
        for j in range(i + 1, len(exchanges)):
            exchange1 = exchanges[i]
            exchange2 = exchanges[j]
            
            scores1 = df_filtered[df_filtered['exchange'] == exchange1]['mean_test_score'].explode().dropna().astype(float)
            scores2 = df_filtered[df_filtered['exchange'] == exchange2]['mean_test_score'].explode().dropna().astype(float)
            
            bootstrap_means1 = bootstrap(scores1)
            bootstrap_means2 = bootstrap(scores2)
            
            lower1, upper1 = np.percentile(bootstrap_means1, [2.5, 97.5])
            lower2, upper2 = np.percentile(bootstrap_means2, [2.5, 97.5])
            
            diff_means = np.array(bootstrap_means1) - np.array(bootstrap_means2)
            lower_diff, upper_diff = np.percentile(diff_means, [2.5, 97.5])
            
            results_list.append({
                'data_type': data_type,
                'exchange1': exchange1,
                'exchange2': exchange2,
                'exchange1_mean_lower': lower1,
                'exchange1_mean_upper': upper1,
                'exchange2_mean_lower': lower2,
                'exchange2_mean_upper': upper2,
                'mean_diff_lower': lower_diff,
                'mean_diff_upper': upper_diff,
                'exchange1_ci_percentage': (upper1 - lower1) / abs(lower1) * 100,
                'exchange2_ci_percentage': (upper2 - lower2) / abs(lower2) * 100,
                'mean_diff_ci_percentage': (upper_diff - lower_diff) / abs(lower_diff) * 100
            })
    
    return pd.DataFrame(results_list)