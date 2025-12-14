"""Causal Inference Analysis Page."""

import streamlit as st
import plotly.express as px
import pandas as pd
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.config.constants import COLOR_PALETTE
from src.data_utils import get_countries
from src.causal_inference import (
    difference_in_differences,
    natural_experiment_finder,
    evaluate_stimulus_effectiveness
)


class CausalInferencePage(BasePage):
    """Econometric causal inference analysis."""
    
    def __init__(self):
        super().__init__(
            "Causal Inference Analysis",
            "Analyze causal relationships and policy impacts using advanced econometric methods."
        )
    
    def _render_content(self, df, config):
        """Render causal inference analysis."""
        countries = config.get('countries', [])
        
        inference_type = st.selectbox(
            "Analysis Type",
            ["Policy Impact (Difference-in-Differences)", "Natural Experiments", "Stimulus Effectiveness"]
        )
        
        country_df = df.copy()
        country_df = country_df.rename(columns={'Entity': 'Country'})
        
        if inference_type == "Policy Impact (Difference-in-Differences)":
            self._render_did_analysis(country_df, countries)
        elif inference_type == "Natural Experiments":
            self._render_natural_experiments(country_df)
        elif inference_type == "Stimulus Effectiveness":
            self._render_stimulus_evaluation(country_df, countries)
    
    def _render_did_analysis(self, country_df, countries):
        """Render difference-in-differences analysis."""
        st.subheader("Difference-in-Differences Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            treatment_countries_did = st.multiselect(
                "Treatment Countries",
                options=countries if countries else get_countries(country_df)[:20]
            )
        with col2:
            control_countries_did = st.multiselect(
                "Control Countries",
                options=get_countries(country_df)[:50]
            )
        
        event_year_did = st.slider("Event Year", min_value=1985, max_value=2020, value=2008)
        policy_name = st.text_input("Policy Name", value="Economic Reform")
        
        if st.button("Run DiD Analysis") and treatment_countries_did and control_countries_did:
            with st.spinner("Running difference-in-differences analysis..."):
                did_result = difference_in_differences(
                    country_df, treatment_countries_did, control_countries_did,
                    event_year_did, pre_years=3, post_years=3
                )
            
            if did_result:
                st.success(did_result['interpretation'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Treatment Effect", f"{did_result['did_estimate']:.2f}%")
                with col2:
                    st.metric("P-value", f"{did_result['p_value']:.4f}")
                with col3:
                    st.metric("Significant", "Yes" if did_result['significant'] else "No")
                
                results_df = pd.DataFrame({
                    'Group': ['Treatment (Pre)', 'Treatment (Post)', 'Control (Pre)', 'Control (Post)'],
                    'Mean Growth': [
                        did_result['pre_treatment_mean'],
                        did_result['post_treatment_mean'],
                        did_result['pre_control_mean'],
                        did_result['post_control_mean']
                    ]
                })
                
                fig_did = px.bar(
                    results_df,
                    x='Group',
                    y='Mean Growth',
                    color='Group',
                    color_discrete_sequence=COLOR_PALETTE
                )
                ChartBuilder.apply_minimal_theme(fig_did)
                st.plotly_chart(fig_did, use_container_width=True)
            else:
                st.error("Unable to perform analysis with selected parameters")
    
    def _render_natural_experiments(self, country_df):
        """Render natural experiment detection."""
        st.subheader("Natural Experiment Detection")
        
        threshold = st.slider("Change Threshold (%)", min_value=2.0, max_value=10.0, value=3.0, step=0.5)
        
        if st.button("Find Natural Experiments"):
            with st.spinner("Detecting natural experiments..."):
                experiments = natural_experiment_finder(country_df, min_countries=2, threshold_change=threshold)
            
            if experiments:
                st.write(f"Found {len(experiments)} potential natural experiments")
                
                experiments_df = pd.DataFrame(experiments)
                st.dataframe(experiments_df, use_container_width=True)
                
                fig_experiments = px.scatter(
                    experiments_df,
                    x='year',
                    y='avg_change',
                    size='n_countries',
                    color='type',
                    hover_data=['affected_countries'],
                    color_discrete_sequence=COLOR_PALETTE
                )
                ChartBuilder.apply_minimal_theme(fig_experiments)
                st.plotly_chart(fig_experiments, use_container_width=True)
            else:
                st.info("No significant natural experiments detected with current threshold")
    
    def _render_stimulus_evaluation(self, country_df, countries):
        """Render stimulus effectiveness evaluation."""
        st.subheader("Stimulus Program Evaluation")
        
        stimulus_countries_input = st.multiselect(
            "Countries that implemented stimulus",
            options=countries if countries else get_countries(country_df)[:30]
        )
        stimulus_year = st.slider("Stimulus Year", min_value=1985, max_value=2020, value=2009)
        
        if st.button("Evaluate Stimulus") and stimulus_countries_input:
            with st.spinner("Evaluating stimulus effectiveness..."):
                stimulus_eval = evaluate_stimulus_effectiveness(
                    country_df, stimulus_countries_input, stimulus_year
                )
            
            if stimulus_eval:
                st.metric("Effectiveness Score", f"{stimulus_eval['effectiveness_score']:.0f}/100")
                
                if stimulus_eval['difference_in_differences']:
                    did = stimulus_eval['difference_in_differences']
                    st.write(f"**Treatment Effect**: {did['did_estimate']:.2f}%")
                    st.write(f"**Statistical Significance**: {'Yes' if did['significant'] else 'No'}")
                
                if stimulus_eval['synthetic_control_samples']:
                    st.subheader("Synthetic Control Results")
                    for sc in stimulus_eval['synthetic_control_samples']:
                        with st.expander(f"{sc['treatment_country']}"):
                            st.write(f"Average Treatment Effect: {sc.get('avg_treatment_effect', 'N/A')}")
                            st.write(f"Cumulative Effect: {sc.get('cumulative_effect', 'N/A')}")
            else:
                st.error("Unable to evaluate stimulus with selected parameters")
