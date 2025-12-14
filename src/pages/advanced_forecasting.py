"""Advanced Forecasting Page - Multi-model ensemble forecasting."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.ui.theme import get_color_palette
from src.advanced_forecasting import (
    ensemble_forecast,
    scenario_analysis,
    multi_horizon_forecast
)


class AdvancedForecastingPage(BasePage):
    """Advanced forecasting models with ensemble methods."""
    
    def __init__(self):
        super().__init__(
            "Advanced Forecasting Models",
            "Multi-model ensemble forecasting with Prophet, ARIMA, Random Forest, and Gradient Boosting."
        )
    
    def _render_content(self, df, config):
        """Render advanced forecasting analysis."""
        countries = config.get('countries', [])
        
        if not countries:
            st.warning("Please select at least one country for forecasting.")
            return
        
        selected_country = st.selectbox("Select country for forecast", countries)
        years_ahead = st.slider("Years to forecast", min_value=1, max_value=10, value=5)
        
        with st.spinner(f"Generating forecasts for {selected_country}..."):
            country_df = df[df['Entity'] == selected_country].copy()
            country_df = country_df.rename(columns={'Entity': 'Country'})
            
            ensemble_result = ensemble_forecast(country_df, selected_country, years_ahead=years_ahead)
            scenario_result = scenario_analysis(country_df, selected_country, years_ahead=years_ahead)
            multi_horizon = multi_horizon_forecast(country_df, selected_country)
        
        self._render_ensemble_forecast(ensemble_result, country_df, selected_country)
        self._render_scenario_analysis(scenario_result, selected_country)
        self._render_multi_horizon(multi_horizon)
    
    def _render_ensemble_forecast(self, ensemble_result, country_df, selected_country):
        """Render ensemble forecast section."""
        if not ensemble_result:
            return
        
        st.subheader("Ensemble Forecast")
        
        forecast_df = pd.DataFrame({
            'Year': ensemble_result['years'],
            'Prediction': ensemble_result['predictions'],
            'Lower Bound': ensemble_result['lower_bound'],
            'Upper Bound': ensemble_result['upper_bound']
        })
        
        fig_ensemble = go.Figure()
        colors = get_color_palette()
        
        historical = country_df[country_df['Year'] <= 2023].sort_values('Year')
        fig_ensemble.add_trace(go.Scatter(
            x=historical['Year'],
            y=historical['GDP_Growth'],
            mode='lines+markers',
            name='Historical',
            line=dict(color=colors[0], width=2)
        ))
        
        fig_ensemble.add_trace(go.Scatter(
            x=forecast_df['Year'],
            y=forecast_df['Prediction'],
            mode='lines+markers',
            name='Ensemble Forecast',
            line=dict(color=colors[4], width=2, dash='dash')
        ))
        
        # Convert hex to rgba for transparency
        hex_color = colors[4].lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgba_str = f'rgba({r}, {g}, {b}, 0.2)'
        
        fig_ensemble.add_trace(go.Scatter(
            x=forecast_df['Year'].tolist() + forecast_df['Year'].tolist()[::-1],
            y=forecast_df['Upper Bound'].tolist() + forecast_df['Lower Bound'].tolist()[::-1],
            fill='toself',
            fillcolor=rgba_str,
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
        
        ChartBuilder.apply_minimal_theme(fig_ensemble)
        fig_ensemble.update_layout(
            title=f"{selected_country} - Ensemble Forecast",
            xaxis_title='Year',
            yaxis_title='GDP Growth (%)'
        )
        st.plotly_chart(fig_ensemble, use_container_width=True)
        
        st.subheader("Individual Model Forecasts")
        for model in ensemble_result['individual_forecasts']:
            with st.expander(f"{model['model']} Model"):
                model_df = pd.DataFrame({
                    'Year': model['years'],
                    'Prediction': model['predictions'],
                    'Lower': model['lower_bound'],
                    'Upper': model['upper_bound']
                })
                st.dataframe(model_df, use_container_width=True)
    
    def _render_scenario_analysis(self, scenario_result, selected_country):
        """Render scenario analysis section."""
        if not scenario_result:
            return
        
        st.subheader("Scenario Analysis")
        
        scenario_df = pd.DataFrame({
            'Year': scenario_result['years'],
            'Optimistic': scenario_result['optimistic'],
            'Baseline': scenario_result['baseline'],
            'Pessimistic': scenario_result['pessimistic']
        })
        
        fig_scenario = go.Figure()
        colors = get_color_palette()
        
        fig_scenario.add_trace(go.Scatter(
            x=scenario_df['Year'], y=scenario_df['Optimistic'],
            name='Optimistic', line=dict(color=colors[2], width=2)
        ))
        fig_scenario.add_trace(go.Scatter(
            x=scenario_df['Year'], y=scenario_df['Baseline'],
            name='Baseline', line=dict(color=colors[0], width=2, dash='dash')
        ))
        fig_scenario.add_trace(go.Scatter(
            x=scenario_df['Year'], y=scenario_df['Pessimistic'],
            name='Pessimistic', line=dict(color=colors[1], width=2)
        ))
        
        ChartBuilder.apply_minimal_theme(fig_scenario)
        fig_scenario.update_layout(
            title=f"{selected_country} - Growth Scenarios",
            xaxis_title='Year',
            yaxis_title='GDP Growth (%)'
        )
        st.plotly_chart(fig_scenario, use_container_width=True)
        
        st.dataframe(scenario_df, use_container_width=True)
    
    def _render_multi_horizon(self, multi_horizon):
        """Render multi-horizon forecast section."""
        if not multi_horizon:
            return
        
        st.subheader("Multi-Horizon Forecasts")
        
        horizons_data = []
        for horizon, data in multi_horizon.items():
            horizons_data.append({
                'Horizon': horizon,
                'Average Growth': f"{data['average_growth']:.2f}%",
                'Years Covered': f"{data['years'][0]}-{data['years'][-1]}"
            })
        
        st.dataframe(pd.DataFrame(horizons_data), use_container_width=True)
