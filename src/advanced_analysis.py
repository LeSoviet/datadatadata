"""
Advanced GDP Analysis Module
Includes: volatility, clustering, forecasting, anomaly detection, regional analysis
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def calculate_volatility(df: pd.DataFrame, metric_col: str = 'gdp_obs', window: int = 10) -> pd.DataFrame:
    """
    Calculate volatility (rolling standard deviation) for each entity.
    
    Args:
        df: DataFrame with GDP data
        metric_col: Column name for GDP metric
        window: Rolling window size for volatility calculation
    
    Returns:
        DataFrame with volatility metrics
    """
    volatility_data = []
    
    for entity in df['Entity'].unique():
        entity_data = df[df['Entity'] == entity].sort_values('Year')
        values = entity_data[metric_col].dropna()
        
        if len(values) >= window:
            # Overall statistics
            mean_growth = values.mean()
            std_growth = values.std()
            cv = std_growth / abs(mean_growth) if mean_growth != 0 else np.nan
            
            # Rolling volatility
            rolling_std = entity_data[metric_col].rolling(window=window, min_periods=window//2).std()
            avg_volatility = rolling_std.mean()
            
            # Recent volatility (last 10 years)
            recent_data = entity_data[entity_data['Year'] >= entity_data['Year'].max() - 9]
            recent_volatility = recent_data[metric_col].std()
            
            volatility_data.append({
                'Entity': entity,
                'mean_growth': mean_growth,
                'std_growth': std_growth,
                'coefficient_of_variation': cv,
                'avg_rolling_volatility': avg_volatility,
                'recent_volatility_10y': recent_volatility,
                'n_observations': len(values)
            })
    
    return pd.DataFrame(volatility_data).sort_values('coefficient_of_variation', ascending=False)


def cluster_countries(df: pd.DataFrame, metric_col: str = 'gdp_obs', n_clusters: int = 4, min_years: int = 20) -> tuple:
    """
    Cluster countries based on GDP growth patterns.
    
    Args:
        df: DataFrame with GDP data
        metric_col: Column name for GDP metric
        n_clusters: Number of clusters
        min_years: Minimum years of data required
    
    Returns:
        Tuple of (clustered_df, cluster_summary, pca_df)
    """
    # Prepare feature matrix
    features_list = []
    entities = []
    
    for entity in df['Entity'].unique():
        entity_data = df[df['Entity'] == entity].sort_values('Year')
        values = entity_data[metric_col].dropna()
        
        if len(values) >= min_years:
            # Extract features
            features = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median(),
                'skew': values.skew(),
                'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0,
                'recent_mean_5y': values.tail(5).mean() if len(values) >= 5 else values.mean(),
                'cv': values.std() / abs(values.mean()) if values.mean() != 0 else 0
            }
            features_list.append(features)
            entities.append(entity)
    
    features_df = pd.DataFrame(features_list, index=entities)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Create result DataFrame
    result_df = features_df.copy()
    result_df['Entity'] = entities
    result_df['Cluster'] = clusters
    
    # Cluster summary
    cluster_summary = result_df.groupby('Cluster').agg({
        'mean': 'mean',
        'std': 'mean',
        'trend': 'mean',
        'Entity': 'count'
    }).rename(columns={'Entity': 'n_countries'})
    cluster_summary['profile'] = cluster_summary.apply(lambda x: _cluster_profile(x), axis=1)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(features_scaled)
    pca_df = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'], index=entities)
    pca_df['Cluster'] = clusters
    pca_df['Entity'] = entities
    
    return result_df, cluster_summary, pca_df


def _cluster_profile(row):
    """Helper to determine cluster profile"""
    if row['mean'] > 4 and row['std'] < 3:
        return "High Stable Growth"
    elif row['mean'] > 4 and row['std'] >= 3:
        return "High Volatile Growth"
    elif row['mean'] < 2 and row['std'] < 2:
        return "Low Stable Growth"
    elif row['std'] > 4:
        return "Very Volatile"
    elif row['trend'] > 0.1:
        return "Accelerating"
    elif row['trend'] < -0.1:
        return "Decelerating"
    else:
        return "Moderate Growth"


def detect_anomalies(df: pd.DataFrame, entity: str, metric_col: str = 'gdp_obs', threshold: float = 2.5) -> pd.DataFrame:
    """
    Detect anomalous years using z-score method.
    
    Args:
        df: DataFrame with GDP data
        entity: Entity name to analyze
        metric_col: Column name for GDP metric
        threshold: Z-score threshold for anomaly detection
    
    Returns:
        DataFrame with anomalies marked
    """
    entity_data = df[df['Entity'] == entity].copy().sort_values('Year')
    values = entity_data[metric_col].dropna()
    
    if len(values) < 5:
        entity_data['is_anomaly'] = False
        entity_data['z_score'] = np.nan
        return entity_data
    
    # Calculate z-scores
    mean = values.mean()
    std = values.std()
    entity_data['z_score'] = (entity_data[metric_col] - mean) / std
    entity_data['is_anomaly'] = entity_data['z_score'].abs() > threshold
    
    return entity_data


def detect_crisis_years(df: pd.DataFrame, metric_col: str = 'gdp_obs', threshold: float = -2.0) -> pd.DataFrame:
    """
    Identify global crisis years based on widespread negative growth.
    
    Args:
        df: DataFrame with GDP data
        metric_col: Column name for GDP metric
        threshold: Growth threshold to consider crisis
    
    Returns:
        DataFrame with crisis years
    """
    yearly_stats = df.groupby('Year').agg({
        metric_col: ['mean', 'median', 'min', 'count'],
        'Entity': 'count'
    })
    
    yearly_stats.columns = ['mean_growth', 'median_growth', 'min_growth', 'n_with_data', 'n_entities']
    yearly_stats = yearly_stats.reset_index()
    
    # Identify crisis years
    yearly_stats['pct_negative'] = df[df[metric_col] < 0].groupby('Year').size() / yearly_stats['n_with_data'] * 100
    yearly_stats['is_crisis'] = (yearly_stats['mean_growth'] < threshold) | (yearly_stats['pct_negative'] > 50)
    
    return yearly_stats.sort_values('Year')


def regional_comparison(df: pd.DataFrame, metric_col: str = 'gdp_obs') -> pd.DataFrame:
    """
    Compare growth by predefined regional groups.
    
    Args:
        df: DataFrame with GDP data
        metric_col: Column name for GDP metric
    
    Returns:
        DataFrame with regional statistics
    """
    # Define regional mappings (based on common entity names)
    regional_keywords = {
        'Advanced economies': ['Advanced', 'OECD', 'G7', 'G20 advanced'],
        'Emerging markets': ['Emerging', 'BRICS', 'developing'],
        'Asia': ['Asia', 'ASEAN', 'East Asia', 'South Asia'],
        'Europe': ['Europe', 'Euro', 'European Union'],
        'Americas': ['America', 'Latin America', 'North America'],
        'Africa': ['Africa', 'Sub-Saharan'],
        'Middle East': ['Middle East', 'MENA']
    }
    
    regional_data = []
    
    for region, keywords in regional_keywords.items():
        # Find entities matching region keywords
        mask = df['Entity'].str.contains('|'.join(keywords), case=False, na=False)
        region_df = df[mask]
        
        if len(region_df) > 0:
            stats = {
                'Region': region,
                'n_entities': region_df['Entity'].nunique(),
                'mean_growth': region_df[metric_col].mean(),
                'median_growth': region_df[metric_col].median(),
                'std_growth': region_df[metric_col].std(),
                'recent_5y_mean': region_df[region_df['Year'] >= region_df['Year'].max() - 4][metric_col].mean(),
                'n_observations': len(region_df)
            }
            regional_data.append(stats)
    
    return pd.DataFrame(regional_data).sort_values('mean_growth', ascending=False)


def forecast_entity(df: pd.DataFrame, entity: str, metric_col: str = 'gdp_obs', 
                   periods: int = 5, method: str = 'prophet') -> pd.DataFrame:
    """
    Forecast GDP growth for a specific entity.
    
    Args:
        df: DataFrame with GDP data
        entity: Entity to forecast
        metric_col: Column name for GDP metric
        periods: Number of periods to forecast
        method: 'prophet' or 'lightgbm'
    
    Returns:
        DataFrame with historical data and forecasts
    """
    entity_data = df[df['Entity'] == entity].sort_values('Year').copy()
    entity_data = entity_data[['Year', metric_col]].dropna()
    
    if len(entity_data) < 10:
        return entity_data
    
    if method == 'prophet':
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            prophet_df = entity_data.rename(columns={'Year': 'ds', metric_col: 'y'})
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')
            
            # Fit model
            model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            model.fit(prophet_df)
            
            # Make forecast
            future = model.make_future_dataframe(periods=periods, freq='YE')
            forecast = model.predict(future)
            
            # Combine with original data
            result = entity_data.copy()
            forecast_years = range(int(entity_data['Year'].max()) + 1, int(entity_data['Year'].max()) + periods + 1)
            forecast_values = forecast['yhat'].tail(periods).values
            
            forecast_df = pd.DataFrame({
                'Year': forecast_years,
                metric_col: np.nan,
                'forecast': forecast_values
            })
            
            result['forecast'] = result[metric_col]
            result = pd.concat([result, forecast_df], ignore_index=True)
            
            return result
            
        except Exception as e:
            print(f"Prophet forecast failed: {e}")
            return entity_data
    
    else:  # Simple moving average fallback
        last_5_mean = entity_data[metric_col].tail(5).mean()
        forecast_years = range(int(entity_data['Year'].max()) + 1, int(entity_data['Year'].max()) + periods + 1)
        forecast_df = pd.DataFrame({
            'Year': forecast_years,
            metric_col: np.nan,
            'forecast': [last_5_mean] * periods
        })
        
        entity_data['forecast'] = entity_data[metric_col]
        result = pd.concat([entity_data, forecast_df], ignore_index=True)
        return result
