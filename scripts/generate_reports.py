"""
Generate analysis reports and export to CSV
"""

from src.data_utils import load_gdp_data
from src.advanced_analysis import (
    calculate_volatility,
    cluster_countries,
    detect_crisis_years,
    regional_comparison
)
from pathlib import Path

# Create outputs directory
output_dir = Path('outputs/analysis')
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading data...")
df = load_gdp_data()

# 1. Volatility Analysis
print("1. Calculating volatility...")
vol_df = calculate_volatility(df, metric_col='gdp_obs')
vol_df.to_csv(output_dir / 'volatility_analysis.csv', index=False)
print(f"   Exported: {len(vol_df)} entities")

# 2. Clustering
print("2. Performing clustering...")
cluster_df, cluster_summary, pca_df = cluster_countries(df, metric_col='gdp_obs', n_clusters=5)
cluster_df.to_csv(output_dir / 'country_clusters.csv', index=False)
cluster_summary.to_csv(output_dir / 'cluster_summary.csv')
pca_df.to_csv(output_dir / 'cluster_pca_coords.csv', index=False)
print(f"   Exported: {len(cluster_df)} countries in 5 clusters")

# 3. Crisis Years
print("3. Detecting crisis years...")
crisis_df = detect_crisis_years(df, metric_col='gdp_obs')
crisis_df.to_csv(output_dir / 'crisis_years.csv', index=False)
crisis_years = crisis_df[crisis_df['is_crisis']]['Year'].tolist()
print(f"   Crisis years: {crisis_years}")

# 4. Regional Comparison
print("4. Regional analysis...")
regional_df = regional_comparison(df, metric_col='gdp_obs')
regional_df.to_csv(output_dir / 'regional_comparison.csv', index=False)
print(f"   Exported: {len(regional_df)} regions")

print(f"\nAll reports exported to: {output_dir.absolute()}")
print("\nSummary:")
print(f"  - volatility_analysis.csv: Volatility metrics for all countries")
print(f"  - country_clusters.csv: Country cluster assignments and features")
print(f"  - cluster_summary.csv: Summary statistics per cluster")
print(f"  - cluster_pca_coords.csv: PCA coordinates for visualization")
print(f"  - crisis_years.csv: Annual global statistics and crisis indicators")
print(f"  - regional_comparison.csv: Regional growth comparisons")
