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

def merge_datasets(df1, df2, on='origin_time'):
    return pd.merge(df1, df2, on=on, how='inner')

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