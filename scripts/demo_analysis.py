"""
Quick demo script showing all analysis capabilities
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

print("="*60)
print("GDP GROWTH ANALYSIS - COMPREHENSIVE DEMO")
print("="*60)

# Load data
print("\n[1/6] Loading dataset...")
df = load_gdp_data()
print(f"      Loaded: {len(df)} observations, {df['Entity'].nunique()} entities")
print(f"      Period: {df['Year'].min()}-{df['Year'].max()}")

# Volatility
print("\n[2/6] Analyzing volatility...")
vol = calculate_volatility(df)
print(f"      Calculated for {len(vol)} countries")
print("\n      Top 5 Most Volatile:")
for i, row in vol.head(5).iterrows():
    print(f"        {i+1}. {row['Entity']}: CV={row['coefficient_of_variation']:.2f}, Mean={row['mean_growth']:.2f}%")
print("\n      Top 5 Most Stable:")
stable = vol.sort_values('coefficient_of_variation').head(5)
for i, (idx, row) in enumerate(stable.iterrows()):
    print(f"        {i+1}. {row['Entity']}: CV={row['coefficient_of_variation']:.2f}, Mean={row['mean_growth']:.2f}%")

# Clustering
print("\n[3/6] Clustering countries...")
clusters, summary, pca = cluster_countries(df, n_clusters=5)
print(f"      Grouped {len(clusters)} countries into 5 clusters")
print("\n      Cluster Profiles:")
for idx, row in summary.iterrows():
    print(f"        Cluster {idx}: {int(row['n_countries'])} countries - {row['profile']}")
    print(f"          Mean Growth: {row['mean']:.2f}%, Std: {row['std']:.2f}%")

# Crisis Detection
print("\n[4/6] Detecting crisis years...")
crisis = detect_crisis_years(df)
crisis_list = crisis[crisis['is_crisis']]['Year'].tolist()
print(f"      Crisis years identified: {crisis_list}")
if len(crisis_list) > 0:
    for year in crisis_list:
        year_data = crisis[crisis['Year'] == year].iloc[0]
        print(f"        {year}: Mean growth={year_data['mean_growth']:.2f}%, {year_data['pct_negative']:.1f}% countries negative")

# Regional
print("\n[5/6] Regional comparison...")
regional = regional_comparison(df)
print(f"      Analyzed {len(regional)} regions")
print("\n      Regional Rankings (by mean growth):")
for i, (idx, row) in enumerate(regional.iterrows()):
    print(f"        {i+1}. {row['Region']}: {row['mean_growth']:.2f}% (Recent 5y: {row['recent_5y_mean']:.2f}%)")

# Anomalies & Forecast example
print("\n[6/6] Country-specific analysis (United States)...")
usa_data = df[df['Entity'] == 'United States'].sort_values('Year')
usa_anomalies = detect_anomalies(df, 'United States', threshold=2.0)
anomaly_years = usa_anomalies[usa_anomalies['is_anomaly']]['Year'].tolist()
print(f"      Anomaly years: {anomaly_years}")

print("\n      Forecasting next 5 years...")
usa_forecast = forecast_entity(df, 'United States', periods=5, method='prophet')
if 'forecast' in usa_forecast.columns:
    future = usa_forecast[usa_forecast['gdp_obs'].isna()]
    print("      Projected growth rates:")
    for _, row in future.iterrows():
        print(f"        {int(row['Year'])}: {row['forecast']:.2f}%")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nTo explore interactively, run:")
print("  streamlit run explore_gdp.py")
print("\nTo generate full reports, run:")
print("  python generate_reports.py")
print("\nOutput files will be in outputs/ and outputs/analysis/")
