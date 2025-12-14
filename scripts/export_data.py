"""Quick data export script to generate summary CSV from GDP dataset"""

import pandas as pd
from pathlib import Path

# Load data
df = pd.read_csv('dataset/real-gdp-growth.csv')

# Rename columns
COL_MAP = {
    'Gross domestic product, constant prices - Percent change - Observations': 'gdp_obs',
    'Gross domestic product, constant prices - Percent change - Forecasts': 'gdp_forecast'
}
df = df.rename(columns=COL_MAP)

# Basic cleaning
df['Year'] = df['Year'].astype(int)
for c in ['gdp_obs','gdp_forecast']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Export full processed data
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

df.to_csv('outputs/gdp_processed.csv', index=False)
print(f'✓ Exported full dataset: outputs/gdp_processed.csv ({len(df)} rows)')

# Create summary: top countries by latest year growth
latest_year = df['Year'].max()
latest = df[df['Year'] == latest_year].dropna(subset=['gdp_obs']).sort_values('gdp_obs', ascending=False)
latest[['Entity', 'Year', 'gdp_obs']].head(20).to_csv('outputs/top20_latest.csv', index=False)
print(f'✓ Exported top 20 countries ({latest_year}): outputs/top20_latest.csv')

# Create summary: average growth by entity (last 10 years)
recent = df[df['Year'] >= (latest_year - 9)]
avg_growth = recent.groupby('Entity')['gdp_obs'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False).reset_index()
avg_growth.columns = ['Entity', 'avg_growth_10y', 'std_growth_10y', 'n_years']
avg_growth.to_csv('outputs/avg_growth_10y.csv', index=False)
print(f'✓ Exported 10-year average growth: outputs/avg_growth_10y.csv ({len(avg_growth)} entities)')

print('\n📊 Summary stats:')
print(f'   Total entities: {df["Entity"].nunique()}')
print(f'   Year range: {df["Year"].min()} - {df["Year"].max()}')
print(f'   Total observations: {len(df)}')
print(f'   Non-null gdp_obs: {df["gdp_obs"].notna().sum()}')
print(f'   Non-null gdp_forecast: {df["gdp_forecast"].notna().sum()}')
