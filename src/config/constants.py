"""
Application Constants
Centralized configuration values used across the application.
"""

# Color palette for charts - expanded for better country differentiation
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
    '#0ea5e9',  # Sky
    '#dc2626',  # Red-700
    '#059669',  # Emerald-600
    '#d97706',  # Amber-600
    '#7c3aed',  # Violet-600
    '#db2777',  # Pink-600
    '#0d9488',  # Teal-600
    '#ea580c',  # Orange-600
    '#0891b2',  # Cyan-600
    '#65a30d',  # Lime-600
    '#4f46e5',  # Indigo-600
    '#e11d48',  # Rose-600
    '#16a34a',  # Green-600
    '#ca8a04',  # Yellow-600
    '#9333ea',  # Purple-600
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
