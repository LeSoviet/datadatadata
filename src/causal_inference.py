import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


def granger_causality_test(df, country_x, country_y, max_lag=3):
    data_x = df[df['Country'] == country_x].sort_values('Year')[['Year', 'GDP_Growth']].copy()
    data_y = df[df['Country'] == country_y].sort_values('Year')[['Year', 'GDP_Growth']].copy()
    
    merged = pd.merge(data_x, data_y, on='Year', suffixes=('_x', '_y'))
    
    if len(merged) < max_lag + 5:
        return None
    
    results = {}
    
    for lag in range(1, max_lag + 1):
        y = merged['GDP_Growth_y'].values[lag:]
        
        X_restricted = merged['GDP_Growth_y'].shift(1).values[lag:]
        
        X_unrestricted = []
        for i in range(lag):
            X_unrestricted.append(merged['GDP_Growth_y'].shift(i+1).values[lag:])
            X_unrestricted.append(merged['GDP_Growth_x'].shift(i+1).values[lag:])
        
        X_restricted = X_restricted.reshape(-1, 1)
        X_unrestricted = np.column_stack(X_unrestricted)
        
        model_restricted = LinearRegression().fit(X_restricted, y)
        model_unrestricted = LinearRegression().fit(X_unrestricted, y)
        
        rss_restricted = np.sum((y - model_restricted.predict(X_restricted)) ** 2)
        rss_unrestricted = np.sum((y - model_unrestricted.predict(X_unrestricted)) ** 2)
        
        n = len(y)
        k = X_unrestricted.shape[1] - X_restricted.shape[1]
        
        f_stat = ((rss_restricted - rss_unrestricted) / k) / (rss_unrestricted / (n - X_unrestricted.shape[1] - 1))
        
        p_value = 1 - stats.f.cdf(f_stat, k, n - X_unrestricted.shape[1] - 1)
        
        results[f'lag_{lag}'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    best_lag = min(results.items(), key=lambda x: x[1]['p_value'])
    
    return {
        'country_x': country_x,
        'country_y': country_y,
        'test_type': 'Granger Causality',
        'lags_tested': list(range(1, max_lag + 1)),
        'results_by_lag': results,
        'best_lag': best_lag[0],
        'best_p_value': best_lag[1]['p_value'],
        'causes': best_lag[1]['significant'],
        'conclusion': f"{country_x} Granger-causes {country_y}" if best_lag[1]['significant'] 
                     else f"{country_x} does not Granger-cause {country_y}"
    }


def difference_in_differences(df, treatment_countries, control_countries, event_year, pre_years=3, post_years=3):
    treatment_data = df[df['Country'].isin(treatment_countries)].copy()
    control_data = df[df['Country'].isin(control_countries)].copy()
    
    treatment_data['group'] = 'treatment'
    control_data['group'] = 'control'
    
    combined = pd.concat([treatment_data, control_data])
    
    combined = combined[
        (combined['Year'] >= event_year - pre_years) & 
        (combined['Year'] <= event_year + post_years)
    ]
    
    if len(combined) == 0:
        return None
    
    combined['post'] = (combined['Year'] >= event_year).astype(int)
    combined['treated'] = (combined['group'] == 'treatment').astype(int)
    combined['did'] = combined['post'] * combined['treated']
    
    pre_treatment = combined[(combined['group'] == 'treatment') & (combined['post'] == 0)]['GDP_Growth'].mean()
    post_treatment = combined[(combined['group'] == 'treatment') & (combined['post'] == 1)]['GDP_Growth'].mean()
    pre_control = combined[(combined['group'] == 'control') & (combined['post'] == 0)]['GDP_Growth'].mean()
    post_control = combined[(combined['group'] == 'control') & (combined['post'] == 1)]['GDP_Growth'].mean()
    
    treatment_effect = (post_treatment - pre_treatment) - (post_control - pre_control)
    
    treatment_diff = post_treatment - pre_treatment
    control_diff = post_control - pre_control
    
    treatment_se = combined[combined['group'] == 'treatment']['GDP_Growth'].std() / np.sqrt(len(treatment_data))
    control_se = combined[combined['group'] == 'control']['GDP_Growth'].std() / np.sqrt(len(control_data))
    
    se_did = np.sqrt(treatment_se**2 + control_se**2)
    
    t_stat = treatment_effect / se_did if se_did > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(combined) - 4))
    
    return {
        'event_year': event_year,
        'treatment_countries': treatment_countries,
        'control_countries': control_countries,
        'pre_treatment_mean': pre_treatment,
        'post_treatment_mean': post_treatment,
        'pre_control_mean': pre_control,
        'post_control_mean': post_control,
        'treatment_diff': treatment_diff,
        'control_diff': control_diff,
        'did_estimate': treatment_effect,
        'standard_error': se_did,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': f"The treatment effect is {treatment_effect:.2f} percentage points " +
                         f"({'significant' if p_value < 0.05 else 'not significant'} at 5% level)"
    }


def synthetic_control(df, treatment_country, control_countries, event_year, pre_years=5):
    treatment_data = df[
        (df['Country'] == treatment_country) & 
        (df['Year'] < event_year) &
        (df['Year'] >= event_year - pre_years)
    ].sort_values('Year')
    
    if len(treatment_data) == 0:
        return None
    
    control_data_list = []
    for country in control_countries:
        country_data = df[
            (df['Country'] == country) & 
            (df['Year'] < event_year) &
            (df['Year'] >= event_year - pre_years)
        ].sort_values('Year')
        
        if len(country_data) == len(treatment_data):
            control_data_list.append(country_data[['Year', 'GDP_Growth']].rename(
                columns={'GDP_Growth': country}
            ))
    
    if len(control_data_list) == 0:
        return None
    
    control_matrix = control_data_list[0].copy()
    for i in range(1, len(control_data_list)):
        control_matrix = pd.merge(control_matrix, control_data_list[i], on='Year')
    
    y = treatment_data['GDP_Growth'].values
    X = control_matrix.drop('Year', axis=1).values
    
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=0.1, fit_intercept=False, positive=True)
    model.fit(X, y)
    
    weights = model.coef_
    weights = weights / weights.sum()
    
    synthetic_pre = X @ weights
    
    treatment_post = df[
        (df['Country'] == treatment_country) & 
        (df['Year'] >= event_year)
    ].sort_values('Year')
    
    control_post_list = []
    for country in control_countries:
        country_data = df[
            (df['Country'] == country) & 
            (df['Year'] >= event_year)
        ].sort_values('Year')
        control_post_list.append(country_data[['Year', 'GDP_Growth']].rename(
            columns={'GDP_Growth': country}
        ))
    
    if len(control_post_list) == 0:
        return {
            'treatment_country': treatment_country,
            'event_year': event_year,
            'weights': dict(zip(control_countries, weights)),
            'pre_treatment_fit': np.mean(np.abs(y - synthetic_pre)),
            'post_treatment_effect': None
        }
    
    control_post_matrix = control_post_list[0].copy()
    for i in range(1, len(control_post_list)):
        control_post_matrix = pd.merge(control_post_matrix, control_post_list[i], on='Year')
    
    X_post = control_post_matrix.drop('Year', axis=1).values
    synthetic_post = X_post @ weights
    
    treatment_effect = treatment_post['GDP_Growth'].values - synthetic_post
    
    return {
        'treatment_country': treatment_country,
        'event_year': event_year,
        'control_countries': control_countries,
        'weights': dict(zip(control_countries, weights)),
        'pre_treatment_fit': np.mean(np.abs(y - synthetic_pre)),
        'post_treatment_years': treatment_post['Year'].tolist(),
        'actual_growth': treatment_post['GDP_Growth'].tolist(),
        'synthetic_growth': synthetic_post.tolist(),
        'treatment_effect': treatment_effect.tolist(),
        'avg_treatment_effect': np.mean(treatment_effect),
        'cumulative_effect': np.sum(treatment_effect)
    }


def policy_impact_analysis(df, policy_name, treatment_countries, control_countries, event_year):
    did_result = difference_in_differences(
        df, treatment_countries, control_countries, 
        event_year, pre_years=3, post_years=3
    )
    
    sc_results = []
    for country in treatment_countries:
        sc_result = synthetic_control(
            df, country, control_countries, 
            event_year, pre_years=5
        )
        if sc_result:
            sc_results.append(sc_result)
    
    granger_results = []
    if len(treatment_countries) == 1 and len(control_countries) >= 1:
        for control in control_countries[:3]:
            granger = granger_causality_test(df, treatment_countries[0], control, max_lag=3)
            if granger:
                granger_results.append(granger)
    
    return {
        'policy_name': policy_name,
        'event_year': event_year,
        'treatment_countries': treatment_countries,
        'control_countries': control_countries,
        'difference_in_differences': did_result,
        'synthetic_control': sc_results,
        'granger_causality': granger_results,
        'overall_assessment': {
            'did_significant': did_result['significant'] if did_result else False,
            'did_effect': did_result['did_estimate'] if did_result else None,
            'avg_synthetic_effect': np.mean([sc['avg_treatment_effect'] for sc in sc_results if 'avg_treatment_effect' in sc]) if sc_results else None
        }
    }


def natural_experiment_finder(df, min_countries=2, threshold_change=3):
    countries = df['Country'].unique()
    
    experiments = []
    
    for year in range(df['Year'].min() + 3, df['Year'].max() - 3):
        year_data = df[df['Year'] == year]
        prev_year_data = df[df['Year'] == year - 1]
        
        merged = pd.merge(
            year_data[['Country', 'GDP_Growth']], 
            prev_year_data[['Country', 'GDP_Growth']], 
            on='Country', 
            suffixes=('_current', '_prev')
        )
        
        merged['change'] = merged['GDP_Growth_current'] - merged['GDP_Growth_prev']
        
        large_positive = merged[merged['change'] > threshold_change]
        large_negative = merged[merged['change'] < -threshold_change]
        
        if len(large_positive) >= min_countries:
            experiments.append({
                'year': year,
                'type': 'Positive Shock',
                'affected_countries': large_positive['Country'].tolist(),
                'n_countries': len(large_positive),
                'avg_change': large_positive['change'].mean()
            })
        
        if len(large_negative) >= min_countries:
            experiments.append({
                'year': year,
                'type': 'Negative Shock',
                'affected_countries': large_negative['Country'].tolist(),
                'n_countries': len(large_negative),
                'avg_change': large_negative['change'].mean()
            })
    
    return experiments


def evaluate_stimulus_effectiveness(df, stimulus_countries, event_year, stimulus_size_pct_gdp=None):
    control_candidates = [c for c in df['Country'].unique() if c not in stimulus_countries]
    
    treatment_growth = df[
        (df['Country'].isin(stimulus_countries)) & 
        (df['Year'] >= event_year - 3) & 
        (df['Year'] < event_year)
    ]['GDP_Growth'].mean()
    
    control_candidates_filtered = []
    for country in control_candidates:
        country_growth = df[
            (df['Country'] == country) & 
            (df['Year'] >= event_year - 3) & 
            (df['Year'] < event_year)
        ]['GDP_Growth'].mean()
        
        if abs(country_growth - treatment_growth) < 2:
            control_candidates_filtered.append(country)
    
    if len(control_candidates_filtered) < 3:
        control_candidates_filtered = control_candidates[:10]
    
    control_countries = control_candidates_filtered[:10]
    
    did = difference_in_differences(
        df, stimulus_countries, control_countries, 
        event_year, pre_years=3, post_years=3
    )
    
    sc_results = []
    for country in stimulus_countries[:3]:
        sc = synthetic_control(df, country, control_countries, event_year, pre_years=5)
        if sc:
            sc_results.append(sc)
    
    return {
        'stimulus_countries': stimulus_countries,
        'event_year': event_year,
        'stimulus_size_pct_gdp': stimulus_size_pct_gdp,
        'control_countries': control_countries,
        'difference_in_differences': did,
        'synthetic_control_samples': sc_results,
        'effectiveness_score': calculate_effectiveness_score(did, sc_results)
    }


def calculate_effectiveness_score(did_result, sc_results):
    if not did_result:
        return None
    
    score = 0
    
    if did_result['significant']:
        if did_result['did_estimate'] > 0:
            score += 50
        else:
            score -= 50
    
    if sc_results:
        avg_effect = np.mean([sc['avg_treatment_effect'] for sc in sc_results if 'avg_treatment_effect' in sc])
        if avg_effect > 1:
            score += 30
        elif avg_effect > 0:
            score += 15
        else:
            score -= 30
    
    return max(0, min(100, score + 50))
