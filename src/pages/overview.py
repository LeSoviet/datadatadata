"""
Overview Page
Main dashboard with time series, maps, rankings, and data export.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.ui.layout import render_metrics_row


class OverviewPage(BasePage):
    """Overview analysis page showing main GDP growth visualizations."""

    def __init__(self):
        super().__init__(
            "Overview",
            "General view of GDP growth trends, rankings, and geographic distribution."
        )

    def _render_content(self, df: pd.DataFrame, config: dict):
        """Render the overview page content."""
        # Filter data
        sel = self._filter_data(df, config)

        if len(sel) == 0:
            self._show_no_data_warning()
            return

        metric_col = config['metric_col']

        # Main metrics
        self._render_summary_metrics(sel, config)

        st.markdown("---")

        # Time series chart
        self._render_time_series(sel, config)

        # World map (conditional)
        if config['show_map'] and len(config['countries']) > 5:
            self._render_world_map(df, sel, config)

        # Latest year ranking
        self._render_rankings(sel, config)

        # Heatmap (conditional)
        if config['show_heatmap']:
            self._render_heatmap(sel, config)

        # Data export
        st.markdown("---")
        self._render_data_export(sel, config)

    def _render_summary_metrics(self, sel: pd.DataFrame, config: dict):
        """Render summary metrics row."""
        metric_col = config['metric_col']
        year_range = config['year_range']
        countries = config['countries']

        avg_growth = sel[metric_col].mean()
        max_growth = sel[metric_col].max()

        metrics = {
            "Countries selected": len(countries),
            "Years": f"{year_range[0]}-{year_range[1]}",
            "Avg growth": f"{avg_growth:.2f}%" if pd.notna(avg_growth) else "N/A",
            "Max growth": f"{max_growth:.2f}%" if pd.notna(max_growth) else "N/A"
        }

        render_metrics_row(metrics)

    def _render_time_series(self, sel: pd.DataFrame, config: dict):
        """Render GDP growth time series chart."""
        st.subheader("GDP Growth Time Series")

        metric_col = config['metric_col']
        show_markers = config['show_markers']

        fig = ChartBuilder.create_time_series(
            sel,
            x='Year',
            y=metric_col,
            color='Entity',
            markers=show_markers
        )

        fig.update_layout(
            yaxis_title='Percent Change',
            xaxis_title='Year'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_world_map(self, df: pd.DataFrame, sel: pd.DataFrame, config: dict):
        """Render world choropleth map for latest year."""
        metric_col = config['metric_col']
        latest_year = sel['Year'].max()
        map_data = df[df['Year'] == latest_year].dropna(subset=[metric_col, 'Code'])

        if len(map_data) > 0:
            st.subheader("World Map: Latest Year GDP Growth")

            fig_map = ChartBuilder.create_choropleth(
                map_data,
                locations='Code',
                color=metric_col,
                hover_name='Entity',
                title=f'GDP Growth ({latest_year}) — World View',
                hover_data={metric_col: ':.2f', 'Code': False},
                labels={metric_col: '% change'}
            )

            st.plotly_chart(fig_map, use_container_width=True)

    def _render_rankings(self, sel: pd.DataFrame, config: dict):
        """Render latest year rankings."""
        metric_col = config['metric_col']
        latest_year_val = int(sel['Year'].max()) if len(sel) > 0 else None
        latest = sel[sel['Year'] == sel['Year'].max()].sort_values(
            metric_col, ascending=False
        ).dropna(subset=[metric_col])

        if len(latest) > 0 and latest_year_val is not None:
            st.subheader(f"Top Performers: {latest_year_val}")
            top_display = latest[['Entity', 'Year', metric_col]].head(10).reset_index(drop=True)
            top_display.index += 1
            st.dataframe(top_display, use_container_width=True)

    def _render_heatmap(self, sel: pd.DataFrame, config: dict):
        """Render heatmap of countries × years."""
        st.markdown("---")
        st.subheader("Heatmap: Countries × Years")

        metric_col = config['metric_col']
        countries = config['countries']

        pivot = sel.pivot(index='Entity', columns='Year', values=metric_col)

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlBu_r',
            hovertemplate='Country: %{y}<br>Year: %{x}<br>Growth: %{z:.2f}%<extra></extra>'
        ))

        ChartBuilder.apply_minimal_theme(fig_heat, height=max(400, len(countries) * 30))
        fig_heat.update_layout(xaxis_title='Year', yaxis_title='Country')

        st.plotly_chart(fig_heat, use_container_width=True)

    def _render_data_export(self, sel: pd.DataFrame, config: dict):
        """Render data export section."""
        st.subheader("Data Export")

        year_range = config['year_range']

        # Download button
        csv = sel.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name=f"gdp_growth_filtered_{year_range[0]}_{year_range[1]}.csv",
            mime="text/csv"
        )

        # Data table expander
        with st.expander("View data table"):
            st.dataframe(sel.sort_values(['Entity', 'Year']), use_container_width=True)
