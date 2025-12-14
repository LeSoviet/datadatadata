"""Growth Regimes Classification Page."""

import streamlit as st
import plotly.express as px
import pandas as pd
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.config.constants import COLOR_PALETTE
from src.growth_regime_classifier import (
    classify_growth_regimes,
    identify_early_warning_signals,
    analyze_policy_effectiveness
)


class GrowthRegimesPage(BasePage):
    """Growth regime classification and transition analysis."""
    
    def __init__(self):
        super().__init__(
            "Growth Regime Classification",
            "Classify countries into growth regimes and identify transitions."
        )
    
    def _render_content(self, df, config):
        """Render growth regimes analysis."""
        countries = config.get('countries', [])
        
        n_regimes = st.slider("Number of regimes", min_value=3, max_value=7, value=5)
        
        with st.spinner("Classifying growth regimes..."):
            country_df = df.copy()
            country_df = country_df.rename(columns={'Entity': 'Country'})
            
            regime_classification = classify_growth_regimes(country_df, n_regimes=n_regimes)
        
        if not regime_classification:
            st.warning("Unable to perform regime classification with current data.")
            return
        
        self._render_regime_stats(regime_classification)
        self._render_country_regimes(regime_classification, countries)
        self._render_regime_distribution(regime_classification)
        self._render_policy_recommendations(country_df, regime_classification)
        self._render_early_warnings(country_df, countries)
    
    def _render_regime_stats(self, regime_classification):
        """Render regime statistics."""
        st.subheader("Regime Classification Results")
        
        regime_stats = regime_classification['regime_statistics']
        st.dataframe(
            regime_stats[['regime_name', 'n_countries', 'avg_mean_growth', 'avg_std_growth', 'avg_negative_ratio']],
            use_container_width=True
        )
    
    def _render_country_regimes(self, regime_classification, countries):
        """Render country regime assignments."""
        st.subheader("Country Regimes")
        country_regimes = regime_classification['country_regimes']
        
        if countries:
            selected_regimes = country_regimes[country_regimes['country'].isin(countries)]
            st.dataframe(selected_regimes, use_container_width=True)
        else:
            st.dataframe(country_regimes, use_container_width=True)
    
    def _render_regime_distribution(self, regime_classification):
        """Render regime distribution chart."""
        country_regimes = regime_classification['country_regimes']
        
        fig_regimes = px.histogram(
            country_regimes,
            x='regime',
            color='regime',
            color_discrete_sequence=COLOR_PALETTE,
            labels={'regime': 'Growth Regime', 'count': 'Number of Countries'}
        )
        ChartBuilder.apply_minimal_theme(fig_regimes)
        fig_regimes.update_layout(showlegend=False)
        st.plotly_chart(fig_regimes, use_container_width=True)
    
    def _render_policy_recommendations(self, country_df, regime_classification):
        """Render policy recommendations by regime."""
        st.subheader("Policy Recommendations by Regime")
        policy_analysis = analyze_policy_effectiveness(country_df, regime_classification)
        
        if policy_analysis is not None:
            st.dataframe(policy_analysis, use_container_width=True)
    
    def _render_early_warnings(self, country_df, countries):
        """Render early warning signals for selected countries."""
        if not countries:
            return
        
        st.subheader("Early Warning Signals")
        for country in countries[:3]:
            with st.expander(f"{country} - Risk Assessment"):
                warning_signals = identify_early_warning_signals(country_df, country, lookback=3)
                
                if warning_signals and warning_signals['warnings']:
                    st.warning(f"Risk Level: {warning_signals['risk_level']} (Score: {warning_signals['risk_score']})")
                    for warning in warning_signals['warnings']:
                        st.write(f"**{warning['type']}** ({warning['severity']}): {warning['description']}")
                else:
                    st.success(f"No significant warnings for {country}")
