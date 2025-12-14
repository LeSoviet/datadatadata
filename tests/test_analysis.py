"""
Test script for advanced analysis functions
"""

from src.data_utils import load_gdp_data
from src.advanced_analysis import (
    calculate_volatility,
    cluster_countries,
    detect_anomalies,
    detect_crisis_years,
    regional_comparison,
    forecast_entity
)

print("Loading data...")
df = load_gdp_data()
print(f"Loaded {len(df)} rows, {df['Entity'].nunique()} entities")

print("\n=== 1. Volatility Analysis ===")
vol_df = calculate_volatility(df, metric_col='gdp_obs')
print(f"Calculated volatility for {len(vol_df)} entities")
print("\nTop 5 most volatile:")
print(vol_df.head()[['Entity', 'mean_growth', 'coefficient_of_variation']])

print("\n=== 2. Clustering ===")
cluster_df, cluster_summary, pca_df = cluster_countries(df, metric_col='gdp_obs', n_clusters=4)
print(f"Clustered {len(cluster_df)} countries into 4 groups")
print("\nCluster summary:")
print(cluster_summary)

print("\n=== 3. Crisis Years Detection ===")
crisis_df = detect_crisis_years(df, metric_col='gdp_obs')
print(f"Analyzed {len(crisis_df)} years")
crisis_years = crisis_df[crisis_df['is_crisis']]['Year'].tolist()
print(f"Crisis years detected: {crisis_years}")

print("\n=== 4. Regional Comparison ===")
regional_df = regional_comparison(df, metric_col='gdp_obs')
print(f"Analyzed {len(regional_df)} regions")
print(regional_df[['Region', 'mean_growth', 'recent_5y_mean']])

print("\n=== 5. Anomaly Detection (USA) ===")
usa_anomalies = detect_anomalies(df, 'United States', metric_col='gdp_obs')
anomalies = usa_anomalies[usa_anomalies['is_anomaly']]
print(f"Found {len(anomalies)} anomalies for USA:")
if len(anomalies) > 0:
    print(anomalies[['Year', 'gdp_obs', 'z_score']])

print("\n=== 6. Forecasting (China) ===")
china_forecast = forecast_entity(df, 'China', metric_col='gdp_obs', periods=5, method='prophet')
print(f"Forecasted {len(china_forecast)} periods for China")
print(china_forecast.tail(10)[['Year', 'gdp_obs', 'forecast']])

print("\n✅ All tests completed successfully!")
