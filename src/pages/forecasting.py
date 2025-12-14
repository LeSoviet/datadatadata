"""
Forecasting Page
Forecast future GDP growth using Prophet time series model.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.advanced_analysis import forecast_entity


class ForecastingPage(BasePage):
    """Forecasting page using Prophet model."""

    def __init__(self):
        super().__init__(
            "GDP Growth Forecasting",
            "Forecast future GDP growth using Prophet time series model."
        )

    def _render_content(self, df: pd.DataFrame, config: dict):
        """Render the forecasting page content."""
        countries = config.get('countries', [])

        if len(countries) == 0:
            self._show_no_data_warning()
            return

        metric_col = config['metric_col']

        # Forecast controls
        forecast_periods = st.slider(
            "Forecast periods (years)",
            min_value=1,
            max_value=10,
            value=5
        )

        # Limit to 3 countries to avoid slowdown
        for country in countries[:3]:
            self._render_country_forecast(df, country, metric_col, forecast_periods)

    def _render_country_forecast(self, df: pd.DataFrame, country: str, metric_col: str, periods: int):
        """Render forecast for a single country."""
        st.subheader(f"Forecast: {country}")

        with st.spinner(f"Forecasting {country}..."):
            forecast_df = forecast_entity(
                df,
                country,
                metric_col=metric_col,
                periods=periods,
                method='prophet'
            )

        if 'forecast' not in forecast_df.columns:
            st.warning(f"Unable to generate forecast for {country}")
            return

        # Create forecast chart
        fig_forecast = go.Figure()

        # Historical data
        hist = forecast_df[forecast_df[metric_col].notna()]
        fig_forecast.add_trace(go.Scatter(
            x=hist['Year'],
            y=hist[metric_col],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#3b82f6', width=3)
        ))

        # Forecast
        fcast = forecast_df[forecast_df[metric_col].isna()]
        fig_forecast.add_trace(go.Scatter(
            x=fcast['Year'],
            y=fcast['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#8b5cf6', width=2, dash='dash')
        ))

        ChartBuilder.apply_minimal_theme(fig_forecast)
        fig_forecast.update_layout(
            yaxis_title='GDP Growth (%)',
            xaxis_title='Year'
        )

        st.plotly_chart(fig_forecast, use_container_width=True)
