import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def calculate_growth_features(df, country):
    country_data = df[df['Country'] == country].copy()
    country_data = country_data.sort_values('Year')
    
    if len(country_data) < 5:
        return None
    
    growth = country_data['GDP_Growth'].values
    
    features = {
        'country': country,
        'mean_growth': np.mean(growth),
        'median_growth': np.median(growth),
        'std_growth': np.std(growth),
        'cv_growth': np.std(growth) / (abs(np.mean(growth)) + 0.001),
        'min_growth': np.min(growth),
        'max_growth': np.max(growth),
        'range_growth': np.max(growth) - np.min(growth),
        'negative_years': np.sum(growth < 0),
        'negative_ratio': np.sum(growth < 0) / len(growth)
    }
    
    if len(growth) >= 3:
        recent_growth = growth[-3:]
        features['recent_mean'] = np.mean(recent_growth)
        features['recent_trend'] = (recent_growth[-1] - recent_growth[0]) / 3
    else:
        features['recent_mean'] = features['mean_growth']
        features['recent_trend'] = 0
    
    features['skewness'] = stats.skew(growth)
    features['kurtosis'] = stats.kurtosis(growth)
    
    return features


def classify_growth_regimes(df, n_regimes=5):
    countries = df['Country'].unique()
    
    features_list = []
    for country in countries:
        features = calculate_growth_features(df, country)
        if features:
            features_list.append(features)
    
    if not features_list:
        return None
    
    features_df = pd.DataFrame(features_list)
    
    feature_cols = [col for col in features_df.columns if col != 'country']
    X = features_df[feature_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    features_df['cluster'] = clusters
    
    regime_names = {
        0: 'TBD',
        1: 'TBD',
        2: 'TBD',
        3: 'TBD',
        4: 'TBD'
    }
    
    cluster_stats = []
    for cluster_id in range(n_regimes):
        cluster_data = features_df[features_df['cluster'] == cluster_id]
        
        stats_dict = {
            'cluster': cluster_id,
            'n_countries': len(cluster_data),
            'avg_mean_growth': cluster_data['mean_growth'].mean(),
            'avg_std_growth': cluster_data['std_growth'].mean(),
            'avg_negative_ratio': cluster_data['negative_ratio'].mean(),
            'countries': cluster_data['country'].tolist()
        }
        
        cluster_stats.append(stats_dict)
    
    cluster_stats_df = pd.DataFrame(cluster_stats)
    cluster_stats_df = cluster_stats_df.sort_values('avg_mean_growth', ascending=False)
    
    for idx, row in cluster_stats_df.iterrows():
        cluster_id = row['cluster']
        avg_growth = row['avg_mean_growth']
        avg_volatility = row['avg_std_growth']
        avg_negative = row['avg_negative_ratio']
        
        if avg_growth > 4 and avg_volatility < 4:
            regime_names[cluster_id] = 'High-Growth Emerging'
        elif avg_growth > 2 and avg_growth <= 4 and avg_volatility < 3:
            regime_names[cluster_id] = 'Stable Developed'
        elif avg_volatility > 5:
            regime_names[cluster_id] = 'Volatile Commodity-Dependent'
        elif avg_negative > 0.3:
            regime_names[cluster_id] = 'Crisis-Prone'
        elif avg_growth < 2:
            regime_names[cluster_id] = 'Stagnant'
        else:
            regime_names[cluster_id] = 'Moderate Growth'
    
    features_df['regime'] = features_df['cluster'].map(regime_names)
    
    cluster_stats_df['regime_name'] = cluster_stats_df['cluster'].map(regime_names)
    
    return {
        'country_regimes': features_df[['country', 'cluster', 'regime']],
        'regime_statistics': cluster_stats_df,
        'feature_importance': {
            'mean_growth': 'High',
            'std_growth': 'High',
            'negative_ratio': 'Medium',
            'recent_trend': 'Medium'
        }
    }


def detect_regime_transitions(df, country, window=5):
    country_data = df[df['Country'] == country].copy()
    country_data = country_data.sort_values('Year')
    
    if len(country_data) < window * 2:
        return None
    
    transitions = []
    
    for i in range(len(country_data) - window):
        period1 = country_data.iloc[i:i+window]
        period2 = country_data.iloc[i+window:i+window+window]
        
        if len(period2) < window:
            break
        
        growth1 = period1['GDP_Growth'].values
        growth2 = period2['GDP_Growth'].values
        
        mean1 = np.mean(growth1)
        mean2 = np.mean(growth2)
        std1 = np.std(growth1)
        std2 = np.std(growth2)
        
        mean_change = mean2 - mean1
        volatility_change = std2 - std1
        
        if abs(mean_change) > 2 or abs(volatility_change) > 2:
            transitions.append({
                'year': period2['Year'].iloc[0],
                'from_period': f"{period1['Year'].iloc[0]}-{period1['Year'].iloc[-1]}",
                'to_period': f"{period2['Year'].iloc[0]}-{period2['Year'].iloc[-1]}",
                'mean_change': mean_change,
                'volatility_change': volatility_change,
                'type': 'Growth Acceleration' if mean_change > 2 else 
                        'Growth Deceleration' if mean_change < -2 else
                        'Volatility Increase' if volatility_change > 2 else
                        'Volatility Decrease'
            })
    
    return transitions


def calculate_transition_probability(df, regime_classification):
    if regime_classification is None:
        return None
    
    countries = df['Country'].unique()
    
    transitions_data = []
    
    for country in countries:
        transitions = detect_regime_transitions(df, country, window=5)
        if transitions:
            for trans in transitions:
                transitions_data.append({
                    'country': country,
                    'year': trans['year'],
                    'transition_type': trans['type']
                })
    
    if not transitions_data:
        return None
    
    transitions_df = pd.DataFrame(transitions_data)
    
    transition_counts = transitions_df.groupby('transition_type').size()
    total_observations = len(df['Country'].unique()) * len(df['Year'].unique())
    
    probabilities = (transition_counts / total_observations) * 100
    
    return {
        'transition_counts': transition_counts.to_dict(),
        'annual_probability': probabilities.to_dict(),
        'total_transitions': len(transitions_data)
    }


def identify_early_warning_signals(df, country, lookback=3):
    country_data = df[df['Country'] == country].copy()
    country_data = country_data.sort_values('Year')
    
    if len(country_data) < lookback + 3:
        return None
    
    recent_data = country_data.tail(lookback + 3)
    
    growth = recent_data['GDP_Growth'].values
    
    signals = {
        'country': country,
        'current_year': recent_data['Year'].iloc[-1],
        'warnings': []
    }
    
    recent_growth = growth[-lookback:]
    recent_mean = np.mean(recent_growth)
    recent_std = np.std(recent_growth)
    
    historical_mean = np.mean(growth[:-lookback])
    historical_std = np.std(growth[:-lookback])
    
    if recent_mean < historical_mean - 2:
        signals['warnings'].append({
            'type': 'Growth Slowdown',
            'severity': 'High',
            'description': f'Recent growth ({recent_mean:.2f}%) significantly below historical average ({historical_mean:.2f}%)'
        })
    
    if recent_std > historical_std * 1.5:
        signals['warnings'].append({
            'type': 'Increased Volatility',
            'severity': 'Medium',
            'description': f'Recent volatility ({recent_std:.2f}) much higher than historical ({historical_std:.2f})'
        })
    
    if np.sum(growth[-2:] < 0) == 2:
        signals['warnings'].append({
            'type': 'Consecutive Contractions',
            'severity': 'High',
            'description': 'Two consecutive years of negative growth'
        })
    
    if len(growth) >= 3:
        trend = (growth[-1] - growth[-3]) / 3
        if trend < -1.5:
            signals['warnings'].append({
                'type': 'Negative Trend',
                'severity': 'Medium',
                'description': f'Strong negative trend ({trend:.2f}% per year)'
            })
    
    signals['risk_score'] = len(signals['warnings']) * 25
    signals['risk_level'] = (
        'Critical' if signals['risk_score'] >= 75 else
        'High' if signals['risk_score'] >= 50 else
        'Medium' if signals['risk_score'] >= 25 else
        'Low'
    )
    
    return signals


def analyze_policy_effectiveness(df, regime_classification):
    if regime_classification is None:
        return None
    
    regime_data = regime_classification['country_regimes']
    
    regime_groups = {}
    
    for _, row in regime_data.iterrows():
        country = row['country']
        regime = row['regime']
        
        if regime not in regime_groups:
            regime_groups[regime] = []
        
        regime_groups[regime].append(country)
    
    policy_insights = []
    
    for regime, countries in regime_groups.items():
        regime_countries_data = df[df['Country'].isin(countries)]
        
        avg_growth = regime_countries_data['GDP_Growth'].mean()
        volatility = regime_countries_data['GDP_Growth'].std()
        
        if regime == 'High-Growth Emerging':
            recommendation = 'Maintain investment-friendly policies, infrastructure development, and human capital formation'
        elif regime == 'Stable Developed':
            recommendation = 'Focus on innovation, productivity gains, and sustainable growth strategies'
        elif regime == 'Volatile Commodity-Dependent':
            recommendation = 'Diversify economy, build fiscal buffers, and reduce commodity dependence'
        elif regime == 'Crisis-Prone':
            recommendation = 'Strengthen institutions, improve governance, and implement counter-cyclical policies'
        elif regime == 'Stagnant':
            recommendation = 'Implement structural reforms, liberalize markets, and attract investment'
        else:
            recommendation = 'Continue balanced growth policies with focus on stability'
        
        policy_insights.append({
            'regime': regime,
            'n_countries': len(countries),
            'avg_growth': avg_growth,
            'volatility': volatility,
            'policy_recommendation': recommendation
        })
    
    return pd.DataFrame(policy_insights)


def regime_success_stories(df, regime_classification):
    if regime_classification is None:
        return None
    
    regime_data = regime_classification['country_regimes']
    
    success_stories = []
    
    for country in regime_data['country'].unique():
        country_data = df[df['Country'] == country].copy()
        country_data = country_data.sort_values('Year')
        
        if len(country_data) < 10:
            continue
        
        early_period = country_data.head(5)
        recent_period = country_data.tail(5)
        
        early_growth = early_period['GDP_Growth'].mean()
        recent_growth = recent_period['GDP_Growth'].mean()
        
        improvement = recent_growth - early_growth
        
        if improvement > 2:
            success_stories.append({
                'country': country,
                'early_growth': early_growth,
                'recent_growth': recent_growth,
                'improvement': improvement,
                'current_regime': regime_data[regime_data['country'] == country]['regime'].iloc[0]
            })
    
    if not success_stories:
        return None
    
    success_df = pd.DataFrame(success_stories)
    success_df = success_df.sort_values('improvement', ascending=False)
    
    return success_df.head(10)
