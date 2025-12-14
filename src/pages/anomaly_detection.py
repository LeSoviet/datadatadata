"""
Anomaly Detection Page
Identify unusual GDP growth events and crisis years.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.ui.theme import get_color_palette
from src.advanced_analysis import detect_crisis_years, detect_anomalies


class AnomalyDetectionPage(BasePage):
    """Anomaly detection page for identifying unusual growth patterns."""

    def __init__(self):
        super().__init__(
            "Anomaly Detection",
            "Identify unusual GDP growth events and crisis years (scoped to selected countries)."
        )

    def _render_content(self, df: pd.DataFrame, config: dict):
        """Render the anomaly detection page content."""
        # Filter data
        sel = self._filter_data(df, config)

        if len(sel) == 0:
            self._show_no_data_warning()
            return

        metric_col = config['metric_col']
        countries = config['countries']

        # Render sections
        self._render_crisis_years(sel, metric_col)
        self._render_country_anomalies(sel, countries, metric_col)

    def _render_crisis_years(self, sel: pd.DataFrame, metric_col: str):
        """Render crisis years analysis."""
        with st.spinner("Analyzing crisis patterns (selected)..."):
            crisis_df = detect_crisis_years(sel, metric_col=metric_col)

        crisis_years = crisis_df[crisis_df['is_crisis']] if len(crisis_df) > 0 else pd.DataFrame()

        if len(crisis_years) > 0:
            st.subheader("Crisis Years (selected)")
            st.dataframe(
                crisis_years[['Year', 'mean_growth', 'median_growth', 'pct_negative']],
                use_container_width=True
            )

            # Timeline of mean growth
            self._render_crisis_timeline(crisis_df)

    def _render_crisis_timeline(self, crisis_df: pd.DataFrame):
        """Render timeline chart of mean growth with crisis indicators."""
        fig_crisis = go.Figure()
        colors = get_color_palette()

        fig_crisis.add_trace(go.Scatter(
            x=crisis_df['Year'],
            y=crisis_df['mean_growth'],
            mode='lines',
            name='Mean Growth (selected)',
            line=dict(color=colors[0], width=3)
        ))

        # Add reference lines
        fig_crisis.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig_crisis.add_hline(y=-2, line_dash="dot", line_color="#ef4444")

        ChartBuilder.apply_minimal_theme(fig_crisis)
        fig_crisis.update_layout(
            yaxis_title='Mean GDP Growth (%)',
            xaxis_title='Year'
        )

        st.plotly_chart(fig_crisis, use_container_width=True)

    def _render_country_anomalies(self, sel: pd.DataFrame, countries: list, metric_col: str):
        """Render country-specific anomaly detection."""
        if not countries:
            return

        st.subheader("Country-Specific Anomalies")

        threshold = st.slider(
            "Z-score threshold",
            min_value=1.5,
            max_value=4.0,
            value=2.5,
            step=0.1
        )

        # Limit to first 5 countries to avoid slowdown
        for country in countries[:5]:
            anomaly_df = detect_anomalies(sel, country, metric_col=metric_col, threshold=threshold)
            anomalies = anomaly_df[anomaly_df['is_anomaly']]

            if len(anomalies) > 0:
                with st.expander(f"{country} — {len(anomalies)} anomalies detected"):
                    st.dataframe(
                        anomalies[['Year', metric_col, 'z_score']],
                        use_container_width=True
                    )
