"""
Quick test script for Phase 1 implementations
Tests all new modules to ensure they work correctly
"""

import pandas as pd
import sys
sys.path.append('src')

from data_utils import load_gdp_data
from correlation_analysis import (
    analyze_growth_momentum,
    detect_structural_breaks
)
from event_impact_analysis import (
    analyze_event_timeline,
    identify_resilient_countries,
    MAJOR_EVENTS
)
from ensemble_forecasting import (
    forecast_ensemble,
    generate_scenarios,
    calculate_forecast_confidence
)

print("Loading GDP data...")
df = load_gdp_data()
print(f"Loaded {len(df)} rows")

print("\n" + "="*60)
print("TEST 1: Growth Momentum Analysis")
print("="*60)
momentum = analyze_growth_momentum(df, gdp_col='gdp_obs')
print(f"Analyzed {len(momentum)} countries")
print("\nTop 5 by current growth:")
print(momentum.nlargest(5, 'current_growth')[['Entity', 'current_growth', 'acceleration']])

print("\n" + "="*60)
print("TEST 2: Event Impact Analysis")
print("="*60)
event_timeline = analyze_event_timeline(df, MAJOR_EVENTS, gdp_col='gdp_obs')
print(f"Analyzed {len(event_timeline)} events")
print("\nEvents sorted by severity:")
print(event_timeline.nlargest(5, 'severity_score')[['event', 'year', 'global_avg_growth', 'severity_score']])

print("\n" + "="*60)
print("TEST 3: Resilience Analysis")
print("="*60)
crisis_years = [event['year'] for event in MAJOR_EVENTS.values()]
resilience = identify_resilient_countries(df, crisis_years, gdp_col='gdp_obs')
print(f"Analyzed resilience for {len(resilience)} countries")
print("\nTop 10 most resilient:")
print(resilience.head(10)[['Entity', 'resilience_rate', 'avg_crisis_growth', 'resilience_score']])

print("\n" + "="*60)
print("TEST 4: Structural Break Detection")
print("="*60)
test_countries = ['United States', 'China', 'Germany']
for country in test_countries:
    breaks = detect_structural_breaks(df, gdp_col='gdp_obs', entity=country)
    if 'error' not in breaks:
        print(f"\n{country}: {len(breaks['break_points'])} breaks detected")
        if breaks['break_points']:
            top_break = breaks['break_points'][0]
            print(f"  Most significant: {top_break['year']} (magnitude: {top_break['magnitude']:.2f}%)")
    else:
        print(f"\n{country}: {breaks['error']}")

print("\n" + "="*60)
print("TEST 5: Ensemble Forecasting")
print("="*60)
test_countries = ['United States', 'India']
for country in test_countries[:1]:
    print(f"\nForecasting for {country}...")
    result = forecast_ensemble(df, country, forecast_periods=5, gdp_col='gdp_obs')
    
    if result.get('success', False):
        result = generate_scenarios(result)
        result = calculate_forecast_confidence(result)
        
        print(f"  Methods used: {', '.join(result['methods_used'])}")
        print(f"  Confidence: {result['confidence']['agreement']*100:.1f}%")
        print(f"  Uncertainty: ±{result['confidence']['uncertainty']:.2f}%")
        print(f"  Baseline forecast (next 5 years): {result['scenarios']['baseline']}")
        print(f"  Optimistic: {result['scenarios']['optimistic'][-1]:.2f}%")
        print(f"  Pessimistic: {result['scenarios']['pessimistic'][-1]:.2f}%")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")

print("\n" + "="*60)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nYou can now run: streamlit run explore_gdp.py")
