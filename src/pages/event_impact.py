"""
Event Impact Analysis Page
Analyze the impact of major economic and geopolitical events.
"""

import streamlit as st
import pandas as pd
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.event_impact_analysis import (
    analyze_event_timeline,
    detect_crisis_impact,
    identify_resilient_countries,
    MAJOR_EVENTS
)


class EventImpactPage(BasePage):
    """Event impact analysis page for major economic events."""

    def __init__(self):
        super().__init__(
            "Event Impact Analysis",
            "Analyze the impact of major economic and geopolitical events."
        )

    def _render_content(self, df: pd.DataFrame, config: dict):
        """Render the event impact analysis page content."""
        # Filter data
        sel = self._filter_data(df, config)

        if len(sel) == 0:
            self._show_no_data_warning()
            return

        metric_col = config['metric_col']

        # Render sections
        self._render_event_timeline(sel, metric_col)
        self._render_crisis_impact(sel, metric_col)
        self._render_resilience_analysis(sel, metric_col)

    def _render_event_timeline(self, sel: pd.DataFrame, metric_col: str):
        """Render event timeline visualization."""
        st.subheader("Event Timeline (selected)")

        event_timeline = analyze_event_timeline(sel, MAJOR_EVENTS, gdp_col=metric_col)

        if len(event_timeline) > 0:
            fig_events = ChartBuilder.create_scatter(
                event_timeline,
                x='year',
                y='global_avg_growth',
                size='severity_score',
                color='type',
                hover_data=['event', 'pct_negative_growth'],
                labels={'global_avg_growth': 'Selected Avg Growth (%)', 'year': 'Year'}
            )

            st.plotly_chart(fig_events, use_container_width=True)
            st.dataframe(event_timeline, use_container_width=True)

    def _render_crisis_impact(self, sel: pd.DataFrame, metric_col: str):
        """Render crisis impact analysis for selected event."""
        st.subheader("Crisis Impact Analysis")

        selected_event = st.selectbox("Select Event", list(MAJOR_EVENTS.keys()))
        event_year = MAJOR_EVENTS[selected_event]['year']

        with st.spinner("Analyzing crisis impact (selected)..."):
            crisis_impact = detect_crisis_impact(sel, event_year, gdp_col=metric_col)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Most Affected (selected)**")
            if len(crisis_impact) > 0:
                st.dataframe(
                    crisis_impact.nlargest(10, 'severity')[['Entity', 'immediate_impact', 'severity']],
                    use_container_width=True
                )

        with col2:
            st.write("**Fastest Recovery (selected)**")
            recovered = crisis_impact[crisis_impact['recovered']] if len(crisis_impact) > 0 else pd.DataFrame()
            if len(recovered) > 0:
                st.dataframe(
                    recovered.nlargest(10, 'recovery_speed')[['Entity', 'recovery_speed', 'after_avg_growth']],
                    use_container_width=True
                )

    def _render_resilience_analysis(self, sel: pd.DataFrame, metric_col: str):
        """Render resilience analysis across multiple crises."""
        st.subheader("Resilience Analysis")

        crisis_years = [event['year'] for event in MAJOR_EVENTS.values()]
        resilience = identify_resilient_countries(sel, crisis_years, gdp_col=metric_col)

        if len(resilience) == 0:
            return

        st.write("**Resilient Countries (selected)**")
        st.dataframe(
            resilience[['Entity', 'n_crises_observed', 'resilience_rate', 'avg_crisis_growth', 'resilience_score']],
            use_container_width=True
        )

        # Resilience bar chart
        fig_resilience = ChartBuilder.create_bar(
            resilience,
            x='Entity',
            y='resilience_score',
            color='resilience_rate',
            color_continuous_scale='Greys'
        )
        fig_resilience.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_resilience, use_container_width=True)
