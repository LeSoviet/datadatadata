"""
Regional Comparison Page
Compare GDP growth across different geographic regions.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.ui.theme import get_color_palette
from src.advanced_analysis import regional_comparison


class RegionalComparisonPage(BasePage):
    """Regional comparison page showing cross-region analysis."""

    def __init__(self):
        super().__init__(
            "Regional Comparison",
            "Compare GDP growth across regions represented by your selection."
        )

    def _render_content(self, df: pd.DataFrame, config: dict):
        """Render the regional comparison page content."""
        # Filter data
        sel = self._filter_data(df, config)

        if len(sel) == 0:
            self._show_no_data_warning()
            return

        metric_col = config['metric_col']

        # Calculate regional statistics
        with st.spinner("Analyzing regional patterns (selected)..."):
            regional_df = regional_comparison(sel, metric_col=metric_col)

        if len(regional_df) == 0:
            self._show_insufficient_data_warning(
                "No regional statistics available for the selected countries."
            )
            return

        # Render sections
        self._render_regional_stats(regional_df)
        self._render_regional_bar_chart(regional_df)
        self._render_recent_vs_historical(regional_df)

    def _render_regional_stats(self, regional_df: pd.DataFrame):
        """Render regional statistics table."""
        st.subheader("Regional Statistics (selected)")
        st.dataframe(regional_df, use_container_width=True)

    def _render_regional_bar_chart(self, regional_df: pd.DataFrame):
        """Render bar chart of mean growth by region."""
        fig_regional = ChartBuilder.create_bar(
            regional_df,
            x='Region',
            y='mean_growth',
            color='Region',
            labels={'mean_growth': 'Mean Growth (%)'}
        )
        fig_regional.update_layout(showlegend=False)
        st.plotly_chart(fig_regional, use_container_width=True)

    def _render_recent_vs_historical(self, regional_df: pd.DataFrame):
        """Render comparison of recent vs historical growth."""
        st.subheader("Recent vs Historical Growth (selected)")

        # Create color mapping for consistent colors per region
        palette = get_color_palette()
        region_list = regional_df['Region'].tolist()
        palette_map = {
            region: palette[i % len(palette)]
            for i, region in enumerate(sorted(set(region_list)))
        }

        colors_all_time = [palette_map[r] for r in region_list]
        colors_recent = [palette_map[r] for r in region_list]

        fig_recent = go.Figure()

        fig_recent.add_trace(go.Bar(
            x=regional_df['Region'],
            y=regional_df['mean_growth'],
            name='All Time',
            marker_color=colors_all_time
        ))

        fig_recent.add_trace(go.Bar(
            x=regional_df['Region'],
            y=regional_df['recent_5y_mean'],
            name='Last 5 Years',
            marker_color=colors_recent
        ))

        ChartBuilder.apply_minimal_theme(fig_recent)
        fig_recent.update_layout(barmode='group')
        st.plotly_chart(fig_recent, use_container_width=True)
