"""
Volatility Analysis Page
Analyzes GDP growth variability and stability across countries.
"""

import streamlit as st
import pandas as pd
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.advanced_analysis import calculate_volatility


class VolatilityPage(BasePage):
    """Volatility analysis page showing risk and stability metrics."""

    def __init__(self):
        super().__init__(
            "Volatility Analysis",
            "Analysis of GDP growth variability and stability across countries."
        )

    def _render_content(self, df: pd.DataFrame, config: dict):
        """Render the volatility analysis page content."""
        # Filter data
        sel = self._filter_data(df, config)

        if len(sel) == 0:
            self._show_no_data_warning()
            return

        metric_col = config['metric_col']

        # Calculate volatility
        with st.spinner("Calculating volatility metrics..."):
            volatility_df = calculate_volatility(sel, metric_col=metric_col)

        if len(volatility_df) == 0:
            self._show_insufficient_data_warning(
                "No volatility data available for the selected countries and years."
            )
            return

        # Render sections
        self._render_volatility_comparison(volatility_df)
        self._render_volatility_scatter(volatility_df)

    def _render_volatility_comparison(self, volatility_df: pd.DataFrame):
        """Render comparison of most volatile vs most stable countries."""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Most Volatile (selected)")
            top_volatile = volatility_df.sort_values(
                'coefficient_of_variation',
                ascending=False
            )[['Entity', 'std_growth', 'coefficient_of_variation', 'recent_volatility_10y']].head(10)

            if len(top_volatile) > 0:
                st.dataframe(top_volatile, use_container_width=True)

        with col2:
            st.subheader("Most Stable (selected)")
            bottom_volatile = volatility_df.sort_values(
                'coefficient_of_variation'
            )[['Entity', 'std_growth', 'coefficient_of_variation']].head(10)

            if len(bottom_volatile) > 0:
                st.dataframe(bottom_volatile, use_container_width=True)

    def _render_volatility_scatter(self, volatility_df: pd.DataFrame):
        """Render scatter plot of mean growth vs volatility."""
        st.subheader("Growth vs Volatility Scatter")

        fig_scatter = ChartBuilder.create_scatter(
            volatility_df,
            x='mean_growth',
            y='std_growth',
            hover_name='Entity',
            size='n_observations',
            color='coefficient_of_variation',
            color_continuous_scale='Greys',
            labels={
                'mean_growth': 'Mean Growth (%)',
                'std_growth': 'Std Dev (%)',
                'coefficient_of_variation': 'CV'
            }
        )

        st.plotly_chart(fig_scatter, use_container_width=True)
