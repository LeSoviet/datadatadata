"""
Base Page Class
Abstract base class for all analysis pages.
"""

import streamlit as st
import pandas as pd
from abc import ABC, abstractmethod


class BasePage(ABC):
    """
    Abstract base class for analysis pages.

    All page implementations should inherit from this class and implement
    the _render_content method.
    """

    def __init__(self, title: str, description: str):
        """
        Initialize the page.

        Args:
            title: Page title
            description: Brief description of the analysis
        """
        self.title = title
        self.description = description

    def render(self, df: pd.DataFrame, config: dict):
        """
        Main render method called by the application.

        Args:
            df: Full GDP DataFrame
            config: Configuration dictionary from sidebar
        """
        self._render_header()
        self._render_content(df, config)

    def _render_header(self):
        """Render the page header with title and description."""
        st.markdown("---")
        st.header(self.title)
        st.write(self.description)

    @abstractmethod
    def _render_content(self, df: pd.DataFrame, config: dict):
        """
        Render the main page content.

        This method must be implemented by subclasses.

        Args:
            df: Full GDP DataFrame
            config: Configuration dictionary from sidebar
        """
        pass

    def _filter_data(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """
        Filter data based on configuration.

        Args:
            df: Full GDP DataFrame
            config: Configuration dictionary

        Returns:
            Filtered DataFrame
        """
        countries = config.get('countries', [])
        year_range = config.get('year_range', (df['Year'].min(), df['Year'].max()))

        if not countries:
            return pd.DataFrame()

        # Filter by countries and year range
        filtered = df[df['Entity'].isin(countries)].copy()
        filtered = filtered[
            (filtered['Year'] >= year_range[0]) &
            (filtered['Year'] <= year_range[1])
        ]
        filtered['Year'] = filtered['Year'].astype(int)

        return filtered

    def _show_no_data_warning(self):
        """Display warning when no data is available."""
        st.warning("Please select at least one country or region from the sidebar.")

    def _show_insufficient_data_warning(self, message: str = None):
        """
        Display warning for insufficient data.

        Args:
            message: Custom message to display
        """
        default_message = "Insufficient data available for the selected countries and year range."
        st.info(message or default_message)
