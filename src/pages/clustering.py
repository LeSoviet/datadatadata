"""
Clustering Page
Group countries by similar GDP growth patterns using K-means.
"""

import streamlit as st
import pandas as pd
from .base import BasePage
from src.ui.charts import ChartBuilder
from src.advanced_analysis import cluster_countries


class ClusteringPage(BasePage):
    """Clustering page for grouping countries by growth patterns."""

    def __init__(self):
        super().__init__(
            "Country Clustering",
            "Group countries by similar GDP growth patterns and characteristics."
        )

    def _render_content(self, df: pd.DataFrame, config: dict):
        """Render the clustering page content."""
        # Filter data
        sel = self._filter_data(df, config)

        if len(sel) == 0:
            self._show_no_data_warning()
            return

        metric_col = config['metric_col']

        # Clustering controls
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=8, value=4)

        # Perform clustering
        with st.spinner("Performing cluster analysis..."):
            # Ensure feasible number of clusters given selection size
            n_available = sel['Entity'].nunique()
            effective_clusters = max(2, min(n_clusters, n_available))
            cluster_df, cluster_summary, pca_df = cluster_countries(
                sel,
                metric_col=metric_col,
                n_clusters=effective_clusters
            )

        # Render sections
        self._render_cluster_summary(cluster_summary)
        self._render_pca_visualization(pca_df)
        self._render_cluster_details(cluster_df)

    def _render_cluster_summary(self, cluster_summary: pd.DataFrame):
        """Render cluster profile summary."""
        if len(cluster_summary) > 0:
            st.subheader("Cluster Profiles (selected)")
            st.dataframe(cluster_summary, use_container_width=True)

    def _render_pca_visualization(self, pca_df: pd.DataFrame):
        """Render PCA visualization of clusters."""
        if len(pca_df) > 0:
            st.subheader("Cluster Visualization (PCA)")

            fig_pca = ChartBuilder.create_scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                hover_name='Entity'
            )

            st.plotly_chart(fig_pca, use_container_width=True)

    def _render_cluster_details(self, cluster_df: pd.DataFrame):
        """Render detailed country assignments by cluster."""
        if len(cluster_df) == 0:
            return

        st.subheader("Countries by Cluster (selected)")

        for cluster_id in sorted(cluster_df['Cluster'].unique()):
            cluster_countries_df = cluster_df[cluster_df['Cluster'] == cluster_id]
            n_countries = len(cluster_countries_df)

            with st.expander(f"Cluster {cluster_id} ({n_countries} countries)"):
                cluster_display = cluster_countries_df[
                    ['Entity', 'mean', 'std', 'trend']
                ].sort_values('mean', ascending=False)

                st.dataframe(cluster_display, use_container_width=True)
