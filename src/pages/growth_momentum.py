"""
Growth Momentum Page
Analyze growth acceleration, momentum indicators, and structural breaks.
"""

import streamlit as st
import pandas as pd
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.correlation_analysis import analyze_growth_momentum, detect_structural_breaks


class GrowthMomentumPage(BasePage):
    """Growth momentum analysis page."""

    def __init__(self):
        super().__init__(
            "Growth Momentum Analysis",
            "Analyze growth acceleration, momentum indicators, and structural breaks."
        )

    def _render_content(self, df: pd.DataFrame, config: dict):
        """Render the growth momentum page content."""
        countries = config.get('countries', [])
        metric_col = config['metric_col']

        # Calculate momentum for all countries, then filter
        with st.spinner("Calculating momentum indicators..."):
            momentum_all = analyze_growth_momentum(df, gdp_col=metric_col)
            momentum = momentum_all[momentum_all['Entity'].isin(countries)]

        if len(momentum) == 0:
            self._show_insufficient_data_warning("No momentum data available for selected countries.")
            return

        # Render sections
        self._render_momentum_leaders(momentum, countries)
        self._render_momentum_scatter(momentum)
        self._render_structural_breaks(df, countries, metric_col)

    def _render_momentum_leaders(self, momentum: pd.DataFrame, countries: list):
        """Render momentum leaders comparison."""
        if len(countries) >= 5:
            st.subheader("Momentum Leaders (selected)")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Positive Acceleration**")
                positive = momentum[momentum['acceleration'] > 0].nlargest(15, 'acceleration')
                st.dataframe(
                    positive[['Entity', 'current_growth', 'acceleration', 'momentum_3y']],
                    use_container_width=True
                )

            with col2:
                st.write("**Negative Acceleration**")
                negative = momentum[momentum['acceleration'] < 0].nsmallest(15, 'acceleration')
                st.dataframe(
                    negative[['Entity', 'current_growth', 'acceleration', 'momentum_3y']],
                    use_container_width=True
                )
        else:
            st.subheader("Selected Countries Momentum")
            st.dataframe(
                momentum[['Entity', 'current_growth', 'acceleration', 'momentum_3y', 'momentum_5y', 'avg_growth_10y']],
                use_container_width=True
            )

    def _render_momentum_scatter(self, momentum: pd.DataFrame):
        """Render momentum scatter plot."""
        if len(momentum) == 0:
            return

        fig_momentum = ChartBuilder.create_scatter(
            momentum,
            x='momentum_5y',
            y='current_growth',
            color='acceleration',
            size=abs(momentum['acceleration']),
            hover_data=['Entity'],
            color_continuous_scale='Greys',
            labels={'momentum_5y': '5-Year Momentum', 'current_growth': 'Current Growth (%)'}
        )

        st.plotly_chart(fig_momentum, use_container_width=True)

    def _render_structural_breaks(self, df: pd.DataFrame, countries: list, metric_col: str):
        """Render structural break detection for countries."""
        if not countries:
            return

        st.subheader("Structural Break Detection")

        # Limit to first 3 countries
        for country in countries[:3]:
            with st.expander(f"{country} — Structural Breaks"):
                breaks = detect_structural_breaks(df, gdp_col=metric_col, entity=country)

                if 'error' not in breaks:
                    if breaks['break_points']:
                        for bp in breaks['break_points']:
                            st.write(
                                f"**{bp['year']}**: Magnitude {bp['magnitude']:.2f}% "
                                f"(before: {bp['before_mean']:.2f}%, after: {bp['after_mean']:.2f}%)"
                            )
                    else:
                        st.info("No significant structural breaks detected.")
                else:
                    st.warning(breaks['error'])
