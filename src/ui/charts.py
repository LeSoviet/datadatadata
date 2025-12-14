"""
Chart Builder Utilities
Provides consistent chart creation and styling across the application.
"""

import plotly.express as px
import plotly.graph_objects as go
from src.ui.theme import get_plot_theme, get_color_palette


class ChartBuilder:
    """
    Utility class for creating consistently styled Plotly charts.
    """

    @staticmethod
    def apply_minimal_theme(fig, height=500):
        """
        Apply minimal light theme to any Plotly figure.

        Args:
            fig: Plotly figure object
            height: Height of the chart in pixels

        Returns:
            Plotly figure with theme applied
        """
        theme = get_plot_theme()

        fig.update_layout(
            plot_bgcolor=theme['plot_bg'],
            paper_bgcolor=theme['paper_bg'],
            font=dict(color=theme['font_color'], size=11, family='Inter'),
            xaxis=dict(
                gridcolor=theme['axis_grid'],
                linecolor=theme['axis_grid'],
                showline=True
            ),
            yaxis=dict(
                gridcolor=theme['axis_grid'],
                linecolor=theme['axis_grid'],
                showline=True,
                zeroline=True,
                zerolinecolor=theme['axis_grid']
            ),
            height=height,
            margin=dict(l=60, r=60, t=40, b=60)
        )
        return fig

    @staticmethod
    def create_time_series(df, x, y, color=None, markers=True, title=None, **kwargs):
        """
        Create a styled time series line chart.

        Args:
            df: DataFrame with data
            x: Column name for x-axis
            y: Column name for y-axis
            color: Column name for color grouping
            markers: Show markers on lines
            title: Chart title
            **kwargs: Additional arguments passed to px.line

        Returns:
            Styled Plotly figure
        """
        fig = px.line(
            df,
            x=x,
            y=y,
            color=color,
            markers=markers,
            color_discrete_sequence=get_color_palette(),
            **kwargs
        )

        ChartBuilder.apply_minimal_theme(fig)

        if title:
            fig.update_layout(title=title)

        fig.update_layout(
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        fig.update_traces(
            line=dict(width=1.5),
            marker=dict(size=5 if markers else 0)
        )

        return fig

    @staticmethod
    def create_scatter(df, x, y, size=None, color=None, hover_name=None, title=None, **kwargs):
        """
        Create a styled scatter plot.

        Args:
            df: DataFrame with data
            x: Column name for x-axis
            y: Column name for y-axis
            size: Column name for marker size
            color: Column name for color
            hover_name: Column name for hover labels
            title: Chart title
            **kwargs: Additional arguments passed to px.scatter

        Returns:
            Styled Plotly figure
        """
        fig = px.scatter(
            df,
            x=x,
            y=y,
            size=size,
            color=color,
            hover_name=hover_name,
            color_discrete_sequence=get_color_palette(),
            **kwargs
        )

        ChartBuilder.apply_minimal_theme(fig)

        if title:
            fig.update_layout(title=title)

        return fig

    @staticmethod
    def create_bar(df, x, y, color=None, title=None, orientation='v', **kwargs):
        """
        Create a styled bar chart.

        Args:
            df: DataFrame with data
            x: Column name for x-axis
            y: Column name for y-axis
            color: Column name for color
            title: Chart title
            orientation: 'v' for vertical, 'h' for horizontal
            **kwargs: Additional arguments passed to px.bar

        Returns:
            Styled Plotly figure
        """
        fig = px.bar(
            df,
            x=x,
            y=y,
            color=color,
            color_discrete_sequence=get_color_palette(),
            orientation=orientation,
            **kwargs
        )

        ChartBuilder.apply_minimal_theme(fig)

        if title:
            fig.update_layout(title=title)

        return fig

    @staticmethod
    def create_heatmap(z, x=None, y=None, colorscale='RdYlBu_r', title=None, **kwargs):
        """
        Create a styled heatmap.

        Args:
            z: 2D array or matrix data
            x: X-axis labels
            y: Y-axis labels
            colorscale: Plotly colorscale name
            title: Chart title
            **kwargs: Additional arguments

        Returns:
            Styled Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            **kwargs
        ))

        ChartBuilder.apply_minimal_theme(fig)

        if title:
            fig.update_layout(title=title)

        return fig

    @staticmethod
    def create_choropleth(df, locations, color, hover_name=None, title=None, **kwargs):
        """
        Create a styled choropleth map.

        Args:
            df: DataFrame with data
            locations: Column name for location codes
            color: Column name for color values
            hover_name: Column name for hover labels
            title: Chart title
            **kwargs: Additional arguments passed to px.choropleth

        Returns:
            Styled Plotly figure
        """
        fig = px.choropleth(
            df,
            locations=locations,
            color=color,
            hover_name=hover_name,
            color_continuous_scale='Greys',
            color_continuous_midpoint=0,
            **kwargs
        )

        ChartBuilder.apply_minimal_theme(fig, height=500)

        if title:
            fig.update_layout(title=title)

        return fig

    @staticmethod
    def create_histogram(df, x, color=None, title=None, **kwargs):
        """
        Create a styled histogram.

        Args:
            df: DataFrame with data
            x: Column name for x-axis
            color: Column name for color grouping
            title: Chart title
            **kwargs: Additional arguments passed to px.histogram

        Returns:
            Styled Plotly figure
        """
        fig = px.histogram(
            df,
            x=x,
            color=color,
            color_discrete_sequence=get_color_palette(),
            **kwargs
        )

        ChartBuilder.apply_minimal_theme(fig)

        if title:
            fig.update_layout(title=title)

        return fig
