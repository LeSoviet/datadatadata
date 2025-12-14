import pandas as pd
import numpy as np
from scipy import stats


def compare_countries(df, countries, metric_col='GDP_Growth'):
    if len(countries) < 2:
        return None
    
    comparison_data = []
    
    for country in countries:
        country_data = df[df['Entity'] == country].copy()
        
        if len(country_data) == 0:
            continue
        
        growth = country_data[metric_col].dropna()
        
        if len(growth) == 0:
            continue
        
        years_available = len(growth)
        recent_growth = growth.tail(5).mean() if len(growth) >= 5 else growth.mean()
        historical_growth = growth.mean()
        
        volatility = growth.std()
        
        max_growth = growth.max()
        max_year = country_data.loc[growth.idxmax(), 'Year'] if len(growth) > 0 else None
        
        min_growth = growth.min()
        min_year = country_data.loc[growth.idxmin(), 'Year'] if len(growth) > 0 else None
        
        negative_years = (growth < 0).sum()
        negative_ratio = negative_years / len(growth) * 100
        
        if len(growth) >= 3:
            recent_trend = (growth.iloc[-1] - growth.iloc[-3]) / 3
        else:
            recent_trend = 0
        
        comparison_data.append({
            'Country': country,
            'Years Available': years_available,
            'Historical Avg Growth': historical_growth,
            'Recent 5Y Avg Growth': recent_growth,
            'Volatility (Std Dev)': volatility,
            'Max Growth': max_growth,
            'Max Growth Year': max_year,
            'Min Growth': min_growth,
            'Min Growth Year': min_year,
            'Negative Years': negative_years,
            'Negative Years %': negative_ratio,
            'Recent Trend': recent_trend
        })
    
    if not comparison_data:
        return None
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df


def side_by_side_comparison(df, country1, country2, metric_col='GDP_Growth'):
    data1 = df[df['Entity'] == country1].copy()
    data2 = df[df['Entity'] == country2].copy()
    
    if len(data1) == 0 or len(data2) == 0:
        return None
    
    merged = pd.merge(
        data1[['Year', metric_col]],
        data2[['Year', metric_col]],
        on='Year',
        suffixes=(f'_{country1}', f'_{country2}')
    )
    
    if len(merged) == 0:
        return None
    
    correlation = merged[[f'{metric_col}_{country1}', f'{metric_col}_{country2}']].corr().iloc[0, 1]
    
    diff = merged[f'{metric_col}_{country1}'] - merged[f'{metric_col}_{country2}']
    avg_difference = diff.mean()
    
    country1_better = (diff > 0).sum()
    country2_better = (diff < 0).sum()
    
    return {
        'merged_data': merged,
        'correlation': correlation,
        'avg_difference': avg_difference,
        'country1_outperformed': country1_better,
        'country2_outperformed': country2_better,
        'total_years': len(merged)
    }


def comparative_ranking(df, year, metric_col='GDP_Growth', top_n=20):
    year_data = df[df['Year'] == year].copy()
    
    year_data = year_data.dropna(subset=[metric_col])
    
    year_data = year_data.sort_values(metric_col, ascending=False)
    
    year_data['Rank'] = range(1, len(year_data) + 1)
    
    return year_data[['Rank', 'Entity', metric_col]].head(top_n)


def peer_group_comparison(df, country, metric_col='GDP_Growth', n_peers=5):
    country_data = df[df['Entity'] == country].copy()
    
    if len(country_data) == 0:
        return None
    
    country_growth = country_data[metric_col].mean()
    country_volatility = country_data[metric_col].std()
    
    all_countries = df['Entity'].unique()
    
    peer_scores = []
    
    for peer in all_countries:
        if peer == country:
            continue
        
        peer_data = df[df['Entity'] == peer].copy()
        
        if len(peer_data) < 5:
            continue
        
        peer_growth = peer_data[metric_col].mean()
        peer_volatility = peer_data[metric_col].std()
        
        growth_diff = abs(peer_growth - country_growth)
        volatility_diff = abs(peer_volatility - country_volatility)
        
        score = growth_diff + volatility_diff
        
        peer_scores.append({
            'Peer': peer,
            'Avg Growth': peer_growth,
            'Volatility': peer_volatility,
            'Similarity Score': score
        })
    
    if not peer_scores:
        return None
    
    peers_df = pd.DataFrame(peer_scores)
    peers_df = peers_df.sort_values('Similarity Score')
    
    return peers_df.head(n_peers)


def convergence_analysis(df, developed_countries, emerging_countries, metric_col='GDP_Growth'):
    developed_data = df[df['Entity'].isin(developed_countries)].copy()
    emerging_data = df[df['Entity'].isin(emerging_countries)].copy()
    
    developed_by_year = developed_data.groupby('Year')[metric_col].mean()
    emerging_by_year = emerging_data.groupby('Year')[metric_col].mean()
    
    convergence_data = pd.DataFrame({
        'Year': developed_by_year.index,
        'Developed Avg': developed_by_year.values,
        'Emerging Avg': emerging_by_year.values
    })
    
    convergence_data['Gap'] = convergence_data['Emerging Avg'] - convergence_data['Developed Avg']
    
    if len(convergence_data) >= 2:
        initial_gap = convergence_data['Gap'].iloc[0]
        final_gap = convergence_data['Gap'].iloc[-1]
        gap_change = final_gap - initial_gap
        
        is_converging = gap_change < 0
    else:
        initial_gap = None
        final_gap = None
        gap_change = None
        is_converging = None
    
    return {
        'data': convergence_data,
        'initial_gap': initial_gap,
        'final_gap': final_gap,
        'gap_change': gap_change,
        'is_converging': is_converging
    }


def relative_performance(df, country, benchmark_countries, metric_col='GDP_Growth'):
    country_data = df[df['Entity'] == country].copy()
    benchmark_data = df[df['Entity'].isin(benchmark_countries)].copy()
    
    if len(country_data) == 0 or len(benchmark_data) == 0:
        return None
    
    benchmark_by_year = benchmark_data.groupby('Year')[metric_col].mean().reset_index()
    benchmark_by_year.columns = ['Year', 'Benchmark_Avg']
    
    merged = pd.merge(
        country_data[['Year', metric_col]],
        benchmark_by_year,
        on='Year'
    )
    
    merged['Difference'] = merged[metric_col] - merged['Benchmark_Avg']
    merged['Outperforming'] = merged['Difference'] > 0
    
    avg_difference = merged['Difference'].mean()
    outperforming_years = merged['Outperforming'].sum()
    total_years = len(merged)
    
    return {
        'data': merged,
        'avg_difference': avg_difference,
        'outperforming_years': outperforming_years,
        'total_years': total_years,
        'outperforming_pct': (outperforming_years / total_years * 100) if total_years > 0 else 0
    }


def identify_leaders_laggards(df, year_range, metric_col='GDP_Growth'):
    filtered_data = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])].copy()
    
    country_stats = filtered_data.groupby('Entity').agg({
        metric_col: ['mean', 'std', 'count']
    }).reset_index()
    
    country_stats.columns = ['Entity', 'Avg_Growth', 'Volatility', 'Years']
    
    country_stats = country_stats[country_stats['Years'] >= (year_range[1] - year_range[0]) * 0.7]
    
    leaders = country_stats.nlargest(10, 'Avg_Growth')
    laggards = country_stats.nsmallest(10, 'Avg_Growth')
    
    return {
        'leaders': leaders,
        'laggards': laggards,
        'all_stats': country_stats
    }


def growth_trajectory_comparison(df, countries, metric_col='GDP_Growth', reference_year=2000):
    trajectories = []
    
    for country in countries:
        country_data = df[(df['Entity'] == country) & (df['Year'] >= reference_year)].copy()
        
        if len(country_data) == 0:
            continue
        
        country_data = country_data.sort_values('Year')
        
        base_growth = country_data[metric_col].iloc[0] if len(country_data) > 0 else 100
        
        cumulative_growth = 100
        cumulative_series = [cumulative_growth]
        
        for growth_rate in country_data[metric_col].iloc[1:]:
            cumulative_growth *= (1 + growth_rate / 100)
            cumulative_series.append(cumulative_growth)
        
        trajectories.append({
            'Country': country,
            'Years': country_data['Year'].tolist(),
            'Cumulative_Index': cumulative_series,
            'Final_Index': cumulative_series[-1] if cumulative_series else 100
        })
    
    return trajectories
