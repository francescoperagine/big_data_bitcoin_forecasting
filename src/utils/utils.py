from sklearn.preprocessing import StandardScaler
from technical_analysis import candles as ta_candles
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_features(df: pd.DataFrame) -> list:
    """Return a list of features in the DataFrame."""	
    return [x for x in df.columns]

def get_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return the rows with null values."""	
    return df[df['null'] == True]

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

def reduce_cumulative_variance(array: np.array, tolerance: float =1e-3) -> np.array:
    array = np.array(array)
    condition = array >= 1.0 - tolerance
    index = np.argmax(condition) if np.any(condition) else len(array)
    return array[:index + 1]

def standard_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Scale the DataFrame using StandardScaler."""	
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

def perform_pca(df: pd.DataFrame, variance_threshold: float) -> tuple[PCA, np.array, np.array]:
    """Perform PCA and return the PCA object."""

    # Perform PCA
    pca = PCA()
    pca = pca.fit(df)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    # Get the number of components that explain the variance threshold
    num_components = np.where(cumulative_variance >= variance_threshold)[0][0] + 1
    pca_reduced = PCA(n_components=num_components).fit(df)
    
    return pca_reduced, explained_variance[:num_components], cumulative_variance[:num_components]

def compute_loadings(pca: PCA, df: pd.DataFrame) -> pd.DataFrame:
    """Compute the PCA loadings and return them as a DataFrame."""
    return pd.DataFrame(pca.components_.T, columns=[f'PC_{i+1}' for i in range(pca.n_components_)], index=df.columns)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the DataFrame."""	
    new_df = df.copy()
    new_df['doji'] = ta_candles.is_doji(new_df['open'], new_df['high'], new_df['low'], new_df['close']).astype(int)
    new_df['bullish_engulfing'] = ta_candles.bullish_engulfing(new_df['open'], new_df['high'], new_df['low'], new_df['close']).astype(int)
    return new_df

def add_orderbook_value_feature(df, orders=20):
    """Add price * size product features to orderbook DataFrame."""
    new_df = df.copy()
    for i in range(orders):
        new_df.insert(new_df.columns.get_loc(f'bid_{i}_size') + 1, f'bid_{i}_value', 0) 
        new_df[f'bid_{i}_value'] = new_df[f'bid_{i}_price'] * new_df[f'bid_{i}_size']
        new_df[f'ask_{i}_value'] = new_df[f'ask_{i}_price'] * new_df[f'ask_{i}_size']
    return new_df

def get_information_gain(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Compute the information gain for each feature and return it as a DataFrame."""	
    ig_scores = mutual_info_classif(X, y)
    return pd.DataFrame(ig_scores, index=X.columns, columns=[f'Information_Gain'])

def find_elbow_point(combined_scores: pd.DataFrame) -> int:
    """Find the elbow point in the scores DataFrame and return the index."""	

    # Calculate the angle between points (naive elbow detection)
    sorted_scores = combined_scores.iloc[:, 0].sort_values(ascending=False).values
    if len(sorted_scores) > 2:  # Ensure there are enough points to calculate second derivative
        angles = np.arctan(np.diff(sorted_scores, n=2))
        if len(angles) > 0:
            elbow = np.argmin(angles) + 1  # +1 as diff reduces the original index by 1
            return elbow
    return 0

def compare_features_scores(pca_loadings: pd.DataFrame, info_scores: pd.DataFrame):
    # Normalize PCA loadings and MI scores
    
    # Euclidean norm of PCA loadings for quantifying the overall importance of each feature across all components
    loadings_norm = np.linalg.norm(pca_loadings, axis=1)

    # Min-max normalization
    loadings_norm = pd.DataFrame(loadings_norm, index=pca_loadings.index, columns=['Loadings_Norm'])
    loadings_norm_scaled = (loadings_norm - loadings_norm.min()) / (loadings_norm.max() - loadings_norm.min())

    info_gain_scaled = (info_scores - info_scores.min()) / (info_scores.max() - info_scores.min())
    
    combined_scores = pd.DataFrame({
        'Loadings_Norm': loadings_norm_scaled.squeeze(),
        'Information_Gain': info_gain_scaled.squeeze()
    })

    combined_scores['Combined_Scores'] = pd.DataFrame((combined_scores['Loadings_Norm'] + combined_scores['Information_Gain']) / 2, columns=['Combined_Scores'])
    return combined_scores

def get_evaluation(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    classification_report = pd.DataFrame(classification_report(y_test, y_pred, target_names=['positive', 'neutral', 'negative'], digits=2, output_dict=True)).transpose()
    confusion_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred), index=['true:positive', 'true:neutral', 'true:negative'], columns=['pred:positive', 'pred:neutral', 'pred:negative'])
    return accuracy, classification_report, confusion_matrix