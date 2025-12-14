"""
Event Impact Analysis Module
Analyzes the impact of major geopolitical and economic events on GDP growth
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


MAJOR_EVENTS = {
    '2008 Financial Crisis': {'year': 2008, 'type': 'financial', 'global': True},
    'COVID-19 Pandemic': {'year': 2020, 'type': 'health', 'global': True},
    'Dot-com Bubble': {'year': 2000, 'type': 'financial', 'global': True},
    'Asian Financial Crisis': {'year': 1997, 'type': 'financial', 'regions': ['Asia']},
    'European Debt Crisis': {'year': 2011, 'type': 'financial', 'regions': ['Europe']},
    'Oil Price Shock': {'year': 1990, 'type': 'commodity', 'global': True},
    'September 11': {'year': 2001, 'type': 'geopolitical', 'global': True},
    'Brexit Referendum': {'year': 2016, 'type': 'political', 'regions': ['Europe']},
    'Russian Financial Crisis': {'year': 1998, 'type': 'financial', 'regions': ['Europe']},
    'Great Recession': {'year': 2009, 'type': 'financial', 'global': True}
}


def detect_crisis_impact(
    df: pd.DataFrame,
    event_year: int,
    gdp_col: str = 'gdp_obs',
    entity_col: str = 'Entity',
    before_window: int = 3,
    after_window: int = 3
) -> pd.DataFrame:
    """
    Analyze the impact of a crisis on different countries.
    
    Args:
        df: DataFrame with GDP data
        event_year: Year of the event
        gdp_col: GDP column name
        entity_col: Entity column name
        before_window: Years before event to compare
        after_window: Years after event to analyze
    
    Returns:
        DataFrame with impact analysis by entity
    """
    results = []
    
    for entity in df[entity_col].unique():
        entity_data = df[df[entity_col] == entity].sort_values('Year')
        
        before_data = entity_data[
            (entity_data['Year'] >= event_year - before_window) & 
            (entity_data['Year'] < event_year)
        ][gdp_col]
        
        during_data = entity_data[entity_data['Year'] == event_year][gdp_col]
        
        after_data = entity_data[
            (entity_data['Year'] > event_year) & 
            (entity_data['Year'] <= event_year + after_window)
        ][gdp_col]
        
        if len(before_data) >= 2 and len(during_data) > 0 and len(after_data) >= 2:
            before_avg = before_data.mean()
            during_value = during_data.iloc[0]
            after_avg = after_data.mean()
            
            immediate_impact = during_value - before_avg
            sustained_impact = after_avg - before_avg
            recovery_speed = (after_avg - during_value) / after_window if during_value < before_avg else 0
            
            results.append({
                'Entity': entity,
                'event_year': event_year,
                'before_avg_growth': before_avg,
                'during_growth': during_value,
                'after_avg_growth': after_avg,
                'immediate_impact': immediate_impact,
                'sustained_impact': sustained_impact,
                'recovery_speed': recovery_speed,
                'severity': abs(immediate_impact),
                'recovered': after_avg >= before_avg
            })
    
    return pd.DataFrame(results).sort_values('severity', ascending=False)


def compare_recovery_trajectories(
    df: pd.DataFrame,
    event_years: List[int],
    gdp_col: str = 'gdp_obs',
    entity_col: str = 'Entity',
    recovery_window: int = 5
) -> pd.DataFrame:
    """
    Compare recovery patterns across different crises.
    
    Args:
        df: DataFrame with GDP data
        event_years: List of crisis years to compare
        gdp_col: GDP column name
        entity_col: Entity column name
        recovery_window: Years after crisis to track
    
    Returns:
        DataFrame with recovery comparison
    """
    results = []
    
    for event_year in event_years:
        for entity in df[entity_col].unique():
            entity_data = df[df[entity_col] == entity].sort_values('Year')
            
            baseline = entity_data[entity_data['Year'] == event_year - 1][gdp_col]
            
            if len(baseline) == 0:
                continue
                
            baseline_value = baseline.iloc[0]
            
            for year_offset in range(recovery_window + 1):
                year = event_year + year_offset
                current = entity_data[entity_data['Year'] == year][gdp_col]
                
                if len(current) > 0:
                    results.append({
                        'Entity': entity,
                        'event_year': event_year,
                        'years_after_crisis': year_offset,
                        'year': year,
                        'growth': current.iloc[0],
                        'growth_vs_baseline': current.iloc[0] - baseline_value,
                        'recovered': current.iloc[0] >= baseline_value
                    })
    
    return pd.DataFrame(results)


def measure_contagion_effect(
    df: pd.DataFrame,
    epicenter_countries: List[str],
    event_year: int,
    gdp_col: str = 'gdp_obs',
    entity_col: str = 'Entity'
) -> pd.DataFrame:
    """
    Measure contagion effects from crisis epicenter to other countries.
    
    Args:
        df: DataFrame with GDP data
        epicenter_countries: Countries where crisis originated
        event_year: Year of crisis
        gdp_col: GDP column name
        entity_col: Entity column name
    
    Returns:
        DataFrame with contagion analysis
    """
    epicenter_impact = []
    other_impact = []
    
    for entity in df[entity_col].unique():
        entity_data = df[df[entity_col] == entity].sort_values('Year')
        
        before = entity_data[entity_data['Year'] == event_year - 1][gdp_col]
        during = entity_data[entity_data['Year'] == event_year][gdp_col]
        
        if len(before) > 0 and len(during) > 0:
            impact = during.iloc[0] - before.iloc[0]
            
            if entity in epicenter_countries:
                epicenter_impact.append(impact)
            else:
                other_impact.append(impact)
    
    return pd.DataFrame({
        'group': ['Epicenter', 'Others'],
        'mean_impact': [np.mean(epicenter_impact), np.mean(other_impact)],
        'median_impact': [np.median(epicenter_impact), np.median(other_impact)],
        'std_impact': [np.std(epicenter_impact), np.std(other_impact)],
        'n_countries': [len(epicenter_impact), len(other_impact)]
    })


def analyze_event_timeline(
    df: pd.DataFrame,
    events: Dict[str, Dict],
    gdp_col: str = 'gdp_obs'
) -> pd.DataFrame:
    """
    Create timeline analysis of major events and their global impact.
    
    Args:
        df: DataFrame with GDP data
        events: Dictionary of events with metadata
        gdp_col: GDP column name
    
    Returns:
        DataFrame with event timeline and impacts
    """
    results = []
    
    for event_name, event_info in events.items():
        year = event_info['year']
        
        year_data = df[df['Year'] == year][gdp_col]
        
        if len(year_data) > 10:
            global_avg = year_data.mean()
            global_median = year_data.median()
            global_std = year_data.std()
            negative_count = (year_data < 0).sum()
            total_count = len(year_data)
            
            results.append({
                'event': event_name,
                'year': year,
                'type': event_info['type'],
                'global': event_info.get('global', False),
                'global_avg_growth': global_avg,
                'global_median_growth': global_median,
                'volatility': global_std,
                'pct_negative_growth': (negative_count / total_count) * 100,
                'severity_score': abs(global_avg) * global_std
            })
    
    return pd.DataFrame(results).sort_values('year')


def calculate_crisis_propagation_speed(
    df: pd.DataFrame,
    event_year: int,
    epicenter_region: str,
    gdp_col: str = 'gdp_obs',
    entity_col: str = 'Entity',
    region_col: str = 'Region'
) -> pd.DataFrame:
    """
    Calculate how quickly a crisis spreads from epicenter to other regions.
    
    Args:
        df: DataFrame with GDP data and regional information
        event_year: Year of crisis
        epicenter_region: Region where crisis started
        gdp_col: GDP column name
        entity_col: Entity column name
        region_col: Region column name
    
    Returns:
        DataFrame with propagation speed by region
    """
    results = []
    
    if region_col not in df.columns:
        return pd.DataFrame()
    
    regions = df[region_col].unique()
    
    for region in regions:
        if pd.isna(region):
            continue
            
        region_data = df[df[region_col] == region]
        
        for year_offset in range(-1, 4):
            year = event_year + year_offset
            year_growth = region_data[region_data['Year'] == year][gdp_col]
            
            if len(year_growth) > 0:
                results.append({
                    'region': region,
                    'year': year,
                    'years_from_event': year_offset,
                    'avg_growth': year_growth.mean(),
                    'is_epicenter': region == epicenter_region
                })
    
    return pd.DataFrame(results)


def identify_resilient_countries(
    df: pd.DataFrame,
    crisis_years: List[int],
    gdp_col: str = 'gdp_obs',
    entity_col: str = 'Entity',
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Identify countries that remained resilient during crises.
    
    Args:
        df: DataFrame with GDP data
        crisis_years: List of major crisis years
        gdp_col: GDP column name
        entity_col: Entity column name
        threshold: Growth threshold to be considered resilient
    
    Returns:
        DataFrame with resilience scores
    """
    results = []
    
    for entity in df[entity_col].unique():
        entity_data = df[df[entity_col] == entity].sort_values('Year')
        
        crisis_performance = []
        
        for crisis_year in crisis_years:
            crisis_growth = entity_data[entity_data['Year'] == crisis_year][gdp_col]
            
            if len(crisis_growth) > 0:
                crisis_performance.append(crisis_growth.iloc[0])
        
        if crisis_performance:
            resilient_count = sum(1 for growth in crisis_performance if growth > threshold)
            avg_crisis_growth = np.mean(crisis_performance)
            min_crisis_growth = np.min(crisis_performance)
            
            results.append({
                'Entity': entity,
                'n_crises_observed': len(crisis_performance),
                'resilient_count': resilient_count,
                'resilience_rate': resilient_count / len(crisis_performance),
                'avg_crisis_growth': avg_crisis_growth,
                'worst_crisis_growth': min_crisis_growth,
                'resilience_score': (resilient_count / len(crisis_performance)) * (avg_crisis_growth + 10)
            })
    
    return pd.DataFrame(results).sort_values('resilience_score', ascending=False)
