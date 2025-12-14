"""
Layout Components
Reusable layout components for the application.
"""

import streamlit as st
import pandas as pd
from src.config.constants import ANALYSIS_MODES, METRIC_OPTIONS, DEFAULT_COUNTRIES
from src.data_utils import get_countries


def render_sidebar(df: pd.DataFrame) -> dict:
    """
    Render the application sidebar with all controls.

    Args:
        df: DataFrame with GDP data

    Returns:
        dict: Configuration dictionary with user selections
    """
    with st.sidebar:
        st.header("Configuration")

        # Analysis mode
        analysis_mode = st.selectbox(
            "Analysis Mode",
            options=ANALYSIS_MODES
        )

        st.markdown("---")

        # Country selection
        all_countries = get_countries(df)
        countries = st.multiselect(
            "Countries / Regions",
            options=all_countries,
            default=DEFAULT_COUNTRIES
        )

        # Metric selection
        metric = st.radio(
            "Metric Type",
            options=METRIC_OPTIONS,
            format_func=lambda x: x[1]
        )
        metric_col = metric[0]

        # Year range filter
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        year_range = st.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(1990, max_year)
        )

        # Visualization options
        st.markdown("---")
        st.subheader("Display Options")
        show_markers = st.checkbox("Line markers", value=True)
        show_map = st.checkbox("World map", value=True)
        show_heatmap = st.checkbox("Heatmap", value=False)

    return {
        'analysis_mode': analysis_mode,
        'countries': countries,
        'metric_col': metric_col,
        'year_range': year_range,
        'min_year': min_year,
        'max_year': max_year,
        'show_markers': show_markers,
        'show_map': show_map,
        'show_heatmap': show_heatmap,
    }


def render_header():
    """Render the application header."""
    st.title("Annual GDP Growth Analysis")


def render_footer():
    """Render the application footer."""
    st.markdown("---")
    st.caption("Data source: International Monetary Fund (WEO) | Processed by Our World in Data")


def render_metrics_row(metrics: dict):
    """
    Render a row of metric cards.

    Args:
        metrics: Dictionary mapping metric labels to values
    """
    cols = st.columns(len(metrics))

    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label, value)
