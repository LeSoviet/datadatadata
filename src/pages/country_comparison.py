"""Country Comparison Tool Page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.ui.theme import get_color_palette
from src.country_comparison import (
    compare_countries,
    side_by_side_comparison,
    peer_group_comparison,
    growth_trajectory_comparison
)


class CountryComparisonPage(BasePage):
    """Multi-country comparison and analysis."""
    
    def __init__(self):
        super().__init__(
            "Country Comparison Tool",
            "Comprehensive side-by-side comparison of selected countries."
        )
    
    def _render_content(self, df, config):
        """Render country comparison analysis."""
        countries = config.get('countries', [])
        metric_col = config.get('metric_col', 'GDP_Growth')
        
        if len(countries) < 2:
            st.warning("Please select at least 2 countries for comparison.")
            return
        
        with st.spinner("Generating comparison..."):
            comparison_df = compare_countries(df, countries, metric_col=metric_col)
        
        if comparison_df is None:
            st.error("Unable to generate comparison")
            return
        
        self._render_summary_stats(comparison_df)
        self._render_comparative_charts(comparison_df)
        self._render_detailed_comparison(df, countries, metric_col)
        self._render_trajectories(df, countries, metric_col)
        self._render_peer_analysis(df, countries, metric_col)
    
    def _render_summary_stats(self, comparison_df):
        """Render summary statistics."""
        st.subheader("Summary Statistics")
        st.dataframe(comparison_df, use_container_width=True)
    
    def _render_comparative_charts(self, comparison_df):
        """Render comparative bar charts."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Growth Comparison")
            fig_avg = px.bar(
                comparison_df,
                x='Country',
                y='Historical Avg Growth',
                color='Country',
                color_discrete_sequence=get_color_palette()
            )
            ChartBuilder.apply_minimal_theme(fig_avg, height=400)
            st.plotly_chart(fig_avg, use_container_width=True)
        
        with col2:
            st.subheader("Volatility Comparison")
            fig_vol = px.bar(
                comparison_df,
                x='Country',
                y='Volatility (Std Dev)',
                color='Country',
                color_discrete_sequence=get_color_palette()
            )
            ChartBuilder.apply_minimal_theme(fig_vol, height=400)
            st.plotly_chart(fig_vol, use_container_width=True)
    
    def _render_detailed_comparison(self, df, countries, metric_col):
        """Render detailed side-by-side comparison for 2 countries."""
        if len(countries) != 2:
            return
        
        st.subheader("Detailed Side-by-Side Analysis")
        sbs_result = side_by_side_comparison(df, countries[0], countries[1], metric_col=metric_col)
        
        if not sbs_result:
            return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correlation", f"{sbs_result['correlation']:.3f}")
        with col2:
            st.metric("Avg Difference", f"{sbs_result['avg_difference']:.2f}%")
        with col3:
            outperform_pct = (sbs_result['country1_outperformed'] / sbs_result['total_years']) * 100
            st.metric(f"{countries[0]} Better", f"{outperform_pct:.0f}%")
        
        fig_sbs = go.Figure()
        
        merged_data = sbs_result['merged_data']
        colors = get_color_palette()
        fig_sbs.add_trace(go.Scatter(
            x=merged_data['Year'],
            y=merged_data[f'{metric_col}_{countries[0]}'],
            mode='lines+markers',
            name=countries[0],
            line=dict(color=colors[0], width=2)
        ))
        
        fig_sbs.add_trace(go.Scatter(
            x=merged_data['Year'],
            y=merged_data[f'{metric_col}_{countries[1]}'],
            mode='lines+markers',
            name=countries[1],
            line=dict(color=colors[1], width=2)
        ))
        
        ChartBuilder.apply_minimal_theme(fig_sbs)
        fig_sbs.update_layout(
            title=f"{countries[0]} vs {countries[1]}",
            xaxis_title='Year',
            yaxis_title='GDP Growth (%)'
        )
        st.plotly_chart(fig_sbs, use_container_width=True)
    
    def _render_trajectories(self, df, countries, metric_col):
        """Render growth trajectories."""
        st.subheader("Growth Trajectories")
        trajectories = growth_trajectory_comparison(df, countries, metric_col=metric_col, reference_year=2000)
        
        if not trajectories:
            return
        
        fig_traj = go.Figure()
        colors = get_color_palette()
        
        for idx, traj in enumerate(trajectories):
            fig_traj.add_trace(go.Scatter(
                x=traj['Years'],
                y=traj['Cumulative_Index'],
                mode='lines+markers',
                name=traj['Country'],
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
        
        ChartBuilder.apply_minimal_theme(fig_traj)
        fig_traj.update_layout(
            title='Cumulative Growth Index (Base = 100 in 2000)',
            xaxis_title='Year',
            yaxis_title='Index (2000=100)'
        )
        st.plotly_chart(fig_traj, use_container_width=True)
    
    def _render_peer_analysis(self, df, countries, metric_col):
        """Render peer group analysis."""
        if len(countries) < 1:
            return
        
        st.subheader("Peer Group Analysis")
        selected_for_peers = st.selectbox("Select country to find peers", countries)
        
        if st.button("Find Similar Countries"):
            with st.spinner("Finding peer countries..."):
                peers = peer_group_comparison(df, selected_for_peers, metric_col=metric_col, n_peers=10)
            
            if peers is not None:
                st.write(f"**Countries most similar to {selected_for_peers}:**")
                st.dataframe(peers, use_container_width=True)
