"""
Correlation Analysis Module
Analyzes relationships between GDP growth and various economic indicators
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_correlation_matrix(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Calculate correlation matrix between multiple metrics.
    
    Args:
        df: DataFrame containing the metrics
        metrics: List of column names to correlate
    
    Returns:
        Correlation matrix DataFrame
    """
    correlation_data = df[metrics].corr(method='pearson')
    return correlation_data


def analyze_lagged_correlation(
    df: pd.DataFrame, 
    target_col: str,
    feature_col: str,
    entity_col: str = 'Entity',
    max_lag: int = 5
) -> pd.DataFrame:
    """
    Analyze lagged correlations between two variables.
    
    Args:
        df: DataFrame with time series data
        target_col: Target variable column name
        feature_col: Feature variable column name
        entity_col: Entity/country column name
        max_lag: Maximum lag periods to test
    
    Returns:
        DataFrame with lag correlations by entity
    """
    results = []
    
    for entity in df[entity_col].unique():
        entity_data = df[df[entity_col] == entity].sort_values('Year').copy()
        
        if len(entity_data) < max_lag + 5:
            continue
            
        for lag in range(max_lag + 1):
            lagged_feature = entity_data[feature_col].shift(lag)
            
            valid_mask = entity_data[target_col].notna() & lagged_feature.notna()
            
            if valid_mask.sum() > 10:
                corr, pval = stats.pearsonr(
                    entity_data.loc[valid_mask, target_col],
                    lagged_feature[valid_mask]
                )
                
                results.append({
                    'Entity': entity,
                    'lag': lag,
                    'correlation': corr,
                    'p_value': pval,
                    'n_observations': valid_mask.sum()
                })
    
    return pd.DataFrame(results)


def granger_causality_test(
    df: pd.DataFrame,
    target_col: str,
    cause_col: str,
    entity: str,
    max_lag: int = 4
) -> Dict:
    """
    Perform Granger causality test to determine if one variable predicts another.
    
    Args:
        df: DataFrame with time series data
        target_col: Target variable column
        cause_col: Potential cause variable column
        entity: Entity name to test
        max_lag: Maximum lag to test
    
    Returns:
        Dictionary with test results
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        
        entity_data = df[df['Entity'] == entity].sort_values('Year')
        data = entity_data[[target_col, cause_col]].dropna()
        
        if len(data) < max_lag + 10:
            return {'entity': entity, 'error': 'Insufficient data'}
        
        test_results = grangercausalitytests(data, max_lag, verbose=False)
        
        results = {
            'entity': entity,
            'lags': {}
        }
        
        for lag in range(1, max_lag + 1):
            f_test = test_results[lag][0]['ssr_ftest']
            results['lags'][lag] = {
                'f_statistic': f_test[0],
                'p_value': f_test[1],
                'significant': f_test[1] < 0.05
            }
        
        return results
        
    except Exception as e:
        return {'entity': entity, 'error': str(e)}


def rolling_correlation(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    entity_col: str = 'Entity',
    window: int = 10
) -> pd.DataFrame:
    """
    Calculate rolling correlation between two variables.
    
    Args:
        df: DataFrame with time series data
        col1: First variable column name
        col2: Second variable column name
        entity_col: Entity/country column name
        window: Rolling window size
    
    Returns:
        DataFrame with rolling correlations
    """
    results = []
    
    for entity in df[entity_col].unique():
        entity_data = df[df[entity_col] == entity].sort_values('Year').copy()
        
        if len(entity_data) < window:
            continue
        
        entity_data['rolling_corr'] = entity_data[col1].rolling(
            window=window,
            min_periods=window//2
        ).corr(entity_data[col2].rolling(window=window, min_periods=window//2))
        
        entity_results = entity_data[['Year', entity_col, 'rolling_corr']].copy()
        entity_results.columns = ['Year', 'Entity', 'correlation']
        
        results.append(entity_results)
    
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def calculate_economic_indicators_correlation(
    gdp_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
    merge_on: List[str] = ['Entity', 'Year']
) -> pd.DataFrame:
    """
    Calculate correlation between GDP growth and multiple economic indicators.
    
    Args:
        gdp_df: DataFrame with GDP data
        indicators_df: DataFrame with economic indicators
        merge_on: Columns to merge on
    
    Returns:
        DataFrame with correlation analysis results
    """
    merged_data = pd.merge(gdp_df, indicators_df, on=merge_on, how='inner')
    
    gdp_col = 'gdp_obs' if 'gdp_obs' in merged_data.columns else 'GDP_growth'
    
    indicator_cols = [col for col in indicators_df.columns 
                     if col not in merge_on and col in merged_data.columns]
    
    results = []
    
    for indicator in indicator_cols:
        valid_data = merged_data[[gdp_col, indicator]].dropna()
        
        if len(valid_data) > 10:
            corr, pval = stats.pearsonr(valid_data[gdp_col], valid_data[indicator])
            
            results.append({
                'indicator': indicator,
                'correlation': corr,
                'p_value': pval,
                'n_observations': len(valid_data),
                'significant': pval < 0.05
            })
    
    return pd.DataFrame(results).sort_values('correlation', ascending=False, key=abs)


def analyze_growth_momentum(
    df: pd.DataFrame,
    gdp_col: str = 'gdp_obs',
    entity_col: str = 'Entity'
) -> pd.DataFrame:
    """
    Calculate growth momentum indicators.
    
    Args:
        df: DataFrame with GDP data
        gdp_col: GDP column name
        entity_col: Entity column name
    
    Returns:
        DataFrame with momentum indicators
    """
    results = []
    
    for entity in df[entity_col].unique():
        entity_data = df[df[entity_col] == entity].sort_values('Year').copy()
        
        if len(entity_data) < 3:
            continue
        
        entity_data['growth_rate'] = entity_data[gdp_col].pct_change()
        entity_data['acceleration'] = entity_data['growth_rate'].diff()
        entity_data['momentum_3y'] = entity_data[gdp_col].rolling(3).mean()
        entity_data['momentum_5y'] = entity_data[gdp_col].rolling(5).mean()
        
        latest = entity_data.iloc[-1]
        
        results.append({
            'Entity': entity,
            'current_growth': latest[gdp_col],
            'growth_rate': latest['growth_rate'],
            'acceleration': latest['acceleration'],
            'momentum_3y': latest['momentum_3y'],
            'momentum_5y': latest['momentum_5y'],
            'avg_growth_10y': entity_data.tail(10)[gdp_col].mean()
        })
    
    return pd.DataFrame(results)


def detect_structural_breaks(
    df: pd.DataFrame,
    gdp_col: str = 'gdp_obs',
    entity: str = None
) -> Dict:
    """
    Detect structural breaks in GDP time series.
    
    Args:
        df: DataFrame with GDP data
        gdp_col: GDP column name
        entity: Entity name to analyze
    
    Returns:
        Dictionary with break point detection results
    """
    entity_data = df[df['Entity'] == entity].sort_values('Year').copy()
    
    if len(entity_data) < 20:
        return {'entity': entity, 'error': 'Insufficient data'}
    
    values = entity_data[gdp_col].values
    years = entity_data['Year'].values
    
    break_candidates = []
    window = 5
    
    for i in range(window, len(values) - window):
        before_mean = np.mean(values[i-window:i])
        after_mean = np.mean(values[i:i+window])
        
        before_std = np.std(values[i-window:i])
        after_std = np.std(values[i:i+window])
        
        if before_std > 0 and after_std > 0:
            t_stat = abs(before_mean - after_mean) / np.sqrt((before_std**2 + after_std**2) / window)
            
            if t_stat > 2:
                break_candidates.append({
                    'year': years[i],
                    't_statistic': t_stat,
                    'before_mean': before_mean,
                    'after_mean': after_mean,
                    'magnitude': abs(after_mean - before_mean)
                })
    
    return {
        'entity': entity,
        'break_points': sorted(break_candidates, key=lambda x: x['t_statistic'], reverse=True)[:3]
    }
