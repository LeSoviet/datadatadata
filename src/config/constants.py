"""
Application Constants
Centralized configuration values used across the application.
"""

# Color palette for charts - colorful but professional
COLOR_PALETTE = [
    '#3b82f6',  # Blue
    '#ef4444',  # Red
    '#10b981',  # Green
    '#f59e0b',  # Amber
    '#8b5cf6',  # Purple
    '#ec4899',  # Pink
    '#14b8a6',  # Teal
    '#f97316',  # Orange
    '#06b6d4',  # Cyan
    '#84cc16',  # Lime
    '#6366f1',  # Indigo
    '#f43f5e',  # Rose
    '#22c55e',  # Emerald
    '#eab308',  # Yellow
    '#a855f7',  # Violet
]

# Default countries to show on initial load
DEFAULT_COUNTRIES = ["United States", "China", "India", "Germany", "Japan"]

# Data path
DATA_PATH = "dataset/real-gdp-growth.csv"

# Analysis modes
ANALYSIS_MODES = [
    "Overview",
    "Volatility Analysis",
    "Clustering",
    "Forecasting",
    "Ensemble Forecasting",
    "Anomaly Detection",
    "Regional Comparison",
    "Event Impact Analysis",
    "Growth Momentum",
    "Advanced Forecasting",
    "Growth Regimes",
    "Causal Inference",
    "Country Comparison",
    "Growth Story",
    "Custom Report"
]

# Metric options
METRIC_OPTIONS = [
    ("gdp_obs", "Observations"),
    ("gdp_forecast", "Forecasts")
]
