"""
Ensemble Forecasting Page
Advanced forecasting combining multiple ML methods with scenario analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.ui.layout import render_metrics_row
from src.ensemble_forecasting import (
    forecast_ensemble,
    generate_scenarios,
    calculate_forecast_confidence
)


class EnsembleForecastingPage(BasePage):
    """Ensemble forecasting page with multiple models and scenarios."""

    def __init__(self):
        super().__init__(
            "Ensemble Forecasting",
            "Advanced forecasting combining multiple ML methods with scenario analysis."
        )

    def _render_content(self, df: pd.DataFrame, config: dict):
        """Render the ensemble forecasting page content."""
        countries = config.get('countries', [])

        if not countries:
            self._show_no_data_warning()
            return

        metric_col = config['metric_col']

        # Forecast controls
        forecast_horizon = st.slider("Forecast Horizon (years)", 3, 10, 5)

        # Limit to 3 countries to avoid slowdown
        for country in countries[:3]:
            self._render_country_ensemble(df, country, metric_col, forecast_horizon)

    def _render_country_ensemble(self, df: pd.DataFrame, country: str, metric_col: str, horizon: int):
        """Render ensemble forecast for a single country."""
        with st.expander(f"Forecast: {country}", expanded=True):
            with st.spinner(f"Generating ensemble forecast for {country}..."):
                result = forecast_ensemble(
                    df,
                    country,
                    forecast_periods=horizon,
                    gdp_col=metric_col
                )

            if not result.get('success', False):
                st.error(f"Forecast failed: {result.get('error', 'Unknown error')}")
                return

            # Generate scenarios and confidence
            result = generate_scenarios(result)
            result = calculate_forecast_confidence(result)

            # Display metrics
            metrics = {
                "Methods Used": len(result['methods_used']),
                "Confidence": f"{result['confidence']['agreement']*100:.1f}%",
                "Uncertainty": f"±{result['confidence']['uncertainty']:.2f}%"
            }
            render_metrics_row(metrics)

            # Create forecast chart
            self._render_forecast_chart(df, country, metric_col, result)

            # Show individual model forecasts
            self._render_individual_forecasts(result)

    def _render_forecast_chart(self, df: pd.DataFrame, country: str, metric_col: str, result: dict):
        """Render the forecast chart with scenarios."""
        historical = df[df['Entity'] == country].sort_values('Year')

        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=historical['Year'],
            y=historical[metric_col],
            mode='lines',
            name='Historical',
            line=dict(color='#3b82f6', width=2)
        ))

        # Scenarios
        scenarios = result['scenarios']
        years = result['forecast_years']

        fig.add_trace(go.Scatter(
            x=years,
            y=scenarios['optimistic'],
            mode='lines',
            name='Optimistic',
            line=dict(color='#10b981', dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=years,
            y=scenarios['baseline'],
            mode='lines+markers',
            name='Baseline',
            line=dict(color='#f59e0b', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=years,
            y=scenarios['pessimistic'],
            mode='lines',
            name='Pessimistic',
            line=dict(color='#ef4444', dash='dash')
        ))

        ChartBuilder.apply_minimal_theme(fig)
        fig.update_layout(
            yaxis_title='GDP Growth (%)',
            xaxis_title='Year'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_individual_forecasts(self, result: dict):
        """Render individual model forecast values."""
        st.write("**Individual Model Forecasts:**")

        for method in result['methods_used']:
            forecast_vals = result['individual_forecasts'][method]['forecast']
            st.write(f"- {method}: {forecast_vals[-1]:.2f}% (final year)")
