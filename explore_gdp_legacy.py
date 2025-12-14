import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_utils import load_gdp_data, get_countries
from src.advanced_analysis import (
    calculate_volatility, 
    cluster_countries, 
    detect_anomalies, 
    detect_crisis_years,
    regional_comparison,
    forecast_entity
)
from src.correlation_analysis import (
    calculate_correlation_matrix,
    analyze_growth_momentum,
    detect_structural_breaks
)
from src.event_impact_analysis import (
    detect_crisis_impact,
    analyze_event_timeline,
    identify_resilient_countries,
    MAJOR_EVENTS
)
from src.ensemble_forecasting import (
    forecast_ensemble,
    generate_scenarios,
    calculate_forecast_confidence
)
from src.advanced_forecasting import (
    ensemble_forecast,
    scenario_analysis,
    multi_horizon_forecast,
    evaluate_forecast_accuracy
)
from src.growth_regime_classifier import (
    classify_growth_regimes,
    identify_early_warning_signals,
    analyze_policy_effectiveness
)
from src.causal_inference import (
    policy_impact_analysis,
    natural_experiment_finder,
    evaluate_stimulus_effectiveness
)
from src.country_comparison import (
    compare_countries,
    side_by_side_comparison,
    peer_group_comparison,
    growth_trajectory_comparison
)
from src.story_generator import (
    generate_growth_story,
    generate_comparative_story,
    format_story_as_text
)
from src.report_builder import (
    create_report_template,
    build_custom_report,
    format_report_as_markdown
)

st.set_page_config(
    page_title="GDP Growth Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal light theme tokens
CT = {
    'sidebar_bg': '#f8f9fa',
    'sidebar_text': '#1f2937',
    'off_black': '#1f2937',
    'gray_900': '#374151',
    'gray_700': '#6b7280',
    'gray_500': '#9ca3af',
    'gray_200': '#e5e7eb',
    'gray_100': '#f3f4f6',
    'off_white': '#ffffff',
    'plot_bg': '#ffffff',
    'paper_bg': '#fafafa',
    'input_bg': '#ffffff',
    'input_border': '#d1d5db',
    'tag_bg': '#d1d5db',
    'tag_text': '#1f2937'
}

# Inject minimal CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

        .stApp {
            background: #fafafa !important;
            color: #1f2937 !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #f8f9fa !important;
            border-right: 1px solid #d1d5db !important;
        }
        section[data-testid="stSidebar"] * { 
            color: #1f2937 !important; 
        }
        section[data-testid="stSidebar"] .stMarkdown h2 {
            color: #1f2937 !important;
            font-weight: 600; font-size: 0.9rem; 
            margin-bottom: 0.5rem; margin-top: 1rem;
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #6b7280 !important;
            font-weight: 500; font-size: 0.85rem;
        }

        /* Sidebar inputs */
        section[data-testid="stSidebar"] div[data-baseweb="select"] {
            background-color: #ffffff !important;
            border: 1px solid #d1d5db !important;
            border-radius: 6px !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] input {
            color: #1f2937 !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] span[data-baseweb="tag"] {
            background-color: #d1d5db !important;
            color: #1f2937 !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 3px 8px !important;
            margin: 2px !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="radio"] label {
            color: #1f2937 !important;
        }

        /* Main content */
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        h1 { color: #1f2937 !important; font-weight: 600; font-size: 2rem; margin-bottom: 1.5rem; }
        h2 { color: #1f2937 !important; font-weight: 600; font-size: 1.4rem; margin-top: 2rem; }
        h3 { color: #6b7280 !important; font-weight: 500; font-size: 1.1rem; }
        
        /* Metrics */
        div[data-testid="stMetricValue"] { color: #1f2937 !important; }
        div[data-testid="stMetricLabel"] { color: #6b7280 !important; }

        /* Tables */
        .stDataFrame, .stTable {
            background: #ffffff !important;
            border: 1px solid #e5e7eb !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("Annual GDP Growth Analysis")

# Load data
df = load_gdp_data()
all_countries = get_countries(df)
metric_col = 'Annual growth (percent)'

# Colorful palette for distinguishing countries/entities in charts
COLOR_PALETTE = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
    '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16',
    '#6366f1', '#f43f5e', '#22c55e', '#eab308', '#a855f7'
]

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    
    # Analysis mode
    analysis_mode = st.selectbox(
        "Analysis Mode",
        options=[
            "Overview", 
            "Volatility Analysis", 
            "Clustering", 
            "Forecasting",
            "Ensemble Forecasting",
            "Anomaly Detection", 
            "Regional Comparison",
            "Event Impact Analysis",
            "Growth Momentum",
            "Advanced Forecasting",
            "Growth Regimes",
            "Causal Inference",
            "Country Comparison",
            "Growth Story",
            "Custom Report"
        ]
    )
    
    st.markdown("---")
    
    # Country selection
    countries = st.multiselect(
        "Countries / Regions", 
        options=all_countries, 
        default=["United States", "China", "India", "Germany", "Japan"]
    )
    
    # Metric selection
    metric = st.radio(
        "Metric Type",
        options=[("gdp_obs", "Observations"), ("gdp_forecast", "Forecasts")],
        format_func=lambda x: x[1]
    )
    metric_col = metric[0]
    
    # Year range filter
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    year_range = st.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(1990, max_year)
    )
    
    # Visualization options
    st.markdown("---")
    st.subheader("Display Options")
    show_markers = st.checkbox("Line markers", value=True)
    show_map = st.checkbox("World map", value=True)
    show_heatmap = st.checkbox("Heatmap", value=False)

# Chart style configuration
plot_bg = '#ffffff'
paper_bg = '#fafafa'
axis_grid = '#e5e7eb'
font_color = '#1f2937'

# Helper function for consistent chart styling
def apply_minimal_theme(fig, height=500):
    """Apply minimal light theme to any Plotly figure"""
    fig.update_layout(
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(color=font_color, size=11, family='Inter'),
        xaxis=dict(gridcolor=axis_grid, linecolor=axis_grid, showline=True),
        yaxis=dict(gridcolor=axis_grid, linecolor=axis_grid, showline=True, zeroline=True, zerolinecolor=axis_grid),
        height=height,
        margin=dict(l=60, r=60, t=40, b=60)
    )
    return fig

if not countries:
    st.warning("Please select at least one country or region from the sidebar.")
else:
    # Filter data
    sel = df[df['Entity'].isin(countries)].copy()
    sel = sel[(sel['Year'] >= year_range[0]) & (sel['Year'] <= year_range[1])]
    sel['Year'] = sel['Year'].astype(int)
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Countries selected", len(countries))
    with col2:
        st.metric("Years", f"{year_range[0]}-{year_range[1]}")
    with col3:
        avg_growth = sel[metric_col].mean()
        st.metric("Avg growth", f"{avg_growth:.2f}%" if pd.notna(avg_growth) else "N/A")
    with col4:
        max_growth = sel[metric_col].max()
        st.metric("Max growth", f"{max_growth:.2f}%" if pd.notna(max_growth) else "N/A")
    
    st.markdown("---")
    
    # Line chart
    st.subheader("GDP Growth Time Series")
    
    fig = px.line(
        sel, 
        x='Year', 
        y=metric_col, 
        color='Entity', 
        markers=show_markers,
        color_discrete_sequence=COLOR_PALETTE
    )
    apply_minimal_theme(fig, height=500)
    fig.update_layout(
        yaxis_title='Percent Change',
        xaxis_title='Year',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    fig.update_traces(line=dict(width=1.5), marker=dict(size=5 if show_markers else 0))
    st.plotly_chart(fig, use_container_width=True)
    
    # World map (choropleth) for latest year — only show for broader selections
    if show_map and len(countries) > 5:
        latest_year = sel['Year'].max()
        map_data = df[df['Year'] == latest_year].dropna(subset=[metric_col, 'Code'])
        if len(map_data) > 0:
            st.subheader("World Map: Latest Year GDP Growth")
            fig_map = px.choropleth(
                map_data,
                locations='Code',
                color=metric_col,
                hover_name='Entity',
                hover_data={metric_col: ':.2f', 'Code': False},
                color_continuous_scale='Greys',
                color_continuous_midpoint=0,
                title=f'GDP Growth ({latest_year}) — World View',
                labels={metric_col: '% change'}
            )
            apply_minimal_theme(fig_map, height=500)
            st.plotly_chart(fig_map, use_container_width=True)
    
    # Latest year ranking
    latest_year_val = int(sel['Year'].max()) if len(sel) > 0 else None
    latest = sel[sel['Year'] == sel['Year'].max()].sort_values(metric_col, ascending=False).dropna(subset=[metric_col])
    if len(latest) > 0 and latest_year_val is not None:
        st.subheader(f"Top Performers: {latest_year_val}")
        top_display = latest[['Entity', 'Year', metric_col]].head(10).reset_index(drop=True)
        top_display.index += 1
        st.dataframe(top_display, width='stretch')
    
    # Heatmap
    if show_heatmap:
        st.markdown("---")
        st.subheader("Heatmap: Countries × Years")
        pivot = sel.pivot(index='Entity', columns='Year', values=metric_col)
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlBu_r',
            hovertemplate='Country: %{y}<br>Year: %{x}<br>Growth: %{z:.2f}%<extra></extra>'
        ))
        apply_minimal_theme(fig_heat, height=max(400, len(countries) * 30))
        fig_heat.update_layout(xaxis_title='Year', yaxis_title='Country')
        st.plotly_chart(fig_heat, use_container_width=True)
    
    # Data table and download
    st.markdown("---")
    st.subheader("Data Export")
    
    # Allow user to download filtered data
    csv = sel.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data (CSV)",
        data=csv,
        file_name=f"gdp_growth_filtered_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv"
    )
    
    with st.expander("View data table"):
        st.dataframe(sel.sort_values(['Entity', 'Year']), width='stretch')
    
    # === ADVANCED ANALYSIS SECTIONS ===
    
    if analysis_mode == "Volatility Analysis":
        st.markdown("---")
        st.header("Volatility Analysis")
        st.write("Analysis of GDP growth variability and stability across countries.")
        
        with st.spinner("Calculating volatility metrics..."):
            volatility_df = calculate_volatility(sel, metric_col=metric_col)
        
        if len(volatility_df) == 0:
            st.info("No volatility data available for the selected countries and years.")
        else:
            # Top volatile countries
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Most Volatile (selected)")
                top_volatile = volatility_df.sort_values('coefficient_of_variation', ascending=False)[['Entity', 'std_growth', 'coefficient_of_variation', 'recent_volatility_10y']].head(10)
                if len(top_volatile) > 0:
                    st.dataframe(top_volatile, width='stretch')
            
            with col2:
                st.subheader("Most Stable (selected)")
                bottom_volatile = volatility_df.sort_values('coefficient_of_variation')[['Entity', 'std_growth', 'coefficient_of_variation']].head(10)
                if len(bottom_volatile) > 0:
                    st.dataframe(bottom_volatile, width='stretch')
            
            # Scatter plot: mean vs std
            st.subheader("Growth vs Volatility Scatter")
            fig_scatter = px.scatter(
                volatility_df,
                x='mean_growth',
                y='std_growth',
                hover_name='Entity',
                size='n_observations',
                color='coefficient_of_variation',
                color_continuous_scale='Greys',
                labels={'mean_growth': 'Mean Growth (%)', 'std_growth': 'Std Dev (%)', 'coefficient_of_variation': 'CV'}
            )
            apply_minimal_theme(fig_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    elif analysis_mode == "Clustering":
        st.markdown("---")
        st.header("Country Clustering")
        st.write("Group countries by similar GDP growth patterns and characteristics.")
        
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=8, value=4)
        
        with st.spinner("Performing cluster analysis..."):
            # Ensure feasible number of clusters given selection size
            n_available = sel['Entity'].nunique()
            effective_clusters = max(2, min(n_clusters, n_available))
            cluster_df, cluster_summary, pca_df = cluster_countries(sel, metric_col=metric_col, n_clusters=effective_clusters)
        
        # Cluster summary (selected)
        if len(cluster_summary) > 0:
            st.subheader("Cluster Profiles (selected)")
            st.dataframe(cluster_summary, width='stretch')
        
        # PCA visualization
        if len(pca_df) > 0:
            st.subheader("Cluster Visualization (PCA)")
            fig_pca = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                hover_name='Entity',
                color_discrete_sequence=COLOR_PALETTE
            )
            apply_minimal_theme(fig_pca)
            st.plotly_chart(fig_pca, use_container_width=True)
        
        # Countries by cluster
        if len(cluster_df) > 0:
            st.subheader("Countries by Cluster (selected)")
            for cluster_id in sorted(cluster_df['Cluster'].unique()):
                with st.expander(f"Cluster {cluster_id} ({len(cluster_df[cluster_df['Cluster']==cluster_id])} countries)"):
                    cluster_countries = cluster_df[cluster_df['Cluster']==cluster_id][['Entity', 'mean', 'std', 'trend']].sort_values('mean', ascending=False)
                    st.dataframe(cluster_countries, width='stretch')
    
    elif analysis_mode == "Forecasting":
        st.markdown("---")
        st.header("GDP Growth Forecasting")
        st.write("Forecast future GDP growth using Prophet time series model.")
        
        if len(countries) == 0:
            st.info("Please select at least one country for forecasting.")
        else:
            forecast_periods = st.slider("Forecast periods (years)", min_value=1, max_value=10, value=5)
            
            for country in countries[:3]:  # Limit to 3 countries to avoid slowdown
                st.subheader(f"Forecast: {country}")
                
                with st.spinner(f"Forecasting {country}..."):
                    forecast_df = forecast_entity(df, country, metric_col=metric_col, periods=forecast_periods, method='prophet')
                
                if 'forecast' in forecast_df.columns:
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
                    
                    apply_minimal_theme(fig_forecast)
                    fig_forecast.update_layout(yaxis_title='GDP Growth (%)', xaxis_title='Year')
                    st.plotly_chart(fig_forecast, use_container_width=True)
    
    elif analysis_mode == "Anomaly Detection":
        st.markdown("---")
        st.header("Anomaly Detection")
        st.write("Identify unusual GDP growth events and crisis years (scoped to selected countries).")
        
        # Crisis years based on selected countries
        with st.spinner("Analyzing crisis patterns (selected)..."):
            crisis_df = detect_crisis_years(sel, metric_col=metric_col)
        
        crisis_years = crisis_df[crisis_df['is_crisis']] if len(crisis_df) > 0 else pd.DataFrame()
        if len(crisis_years) > 0:
            st.subheader("Crisis Years (selected)")
            st.dataframe(crisis_years[['Year', 'mean_growth', 'median_growth', 'pct_negative']], width='stretch')
            
            # Timeline of mean growth (selected)
            fig_crisis = go.Figure()
            fig_crisis.add_trace(go.Scatter(
                x=crisis_df['Year'],
                y=crisis_df['mean_growth'],
                mode='lines',
                name='Mean Growth (selected)',
                line=dict(color='#3b82f6', width=3)
            ))
            fig_crisis.add_hline(y=0, line_dash="dash", line_color="#64748b")
            fig_crisis.add_hline(y=-2, line_dash="dot", line_color="#ef4444")
            
            apply_minimal_theme(fig_crisis)
            fig_crisis.update_layout(yaxis_title='Mean GDP Growth (%)', xaxis_title='Year')
            st.plotly_chart(fig_crisis, use_container_width=True)
        
        # Country-specific anomalies
        if countries:
            st.subheader("Country-Specific Anomalies")
            threshold = st.slider("Z-score threshold", min_value=1.5, max_value=4.0, value=2.5, step=0.1)
            
            for country in countries[:5]:
                anomaly_df = detect_anomalies(sel, country, metric_col=metric_col, threshold=threshold)
                anomalies = anomaly_df[anomaly_df['is_anomaly']]
                
                if len(anomalies) > 0:
                    with st.expander(f"{country} — {len(anomalies)} anomalies detected"):
                        st.dataframe(anomalies[['Year', metric_col, 'z_score']], width='stretch')
    
    elif analysis_mode == "Regional Comparison":
        st.markdown("---")
        st.header("Regional Comparison")
        st.write("Compare GDP growth across regions represented by your selection.")
        
        with st.spinner("Analyzing regional patterns (selected)..."):
            regional_df = regional_comparison(sel, metric_col=metric_col)
        
        if len(regional_df) == 0:
            st.info("No regional statistics available for the selected countries.")
        else:
            st.subheader("Regional Statistics (selected)")
            st.dataframe(regional_df, width='stretch')
        
        # Bar chart comparison
        # Multicolor bars per region for clear distinction
        if len(regional_df) > 0:
            fig_regional = px.bar(
                regional_df,
                x='Region',
                y='mean_growth',
                color='Region',
                color_discrete_sequence=COLOR_PALETTE,
                labels={'mean_growth': 'Mean Growth (%)'}
            )
            apply_minimal_theme(fig_regional)
            fig_regional.update_layout(showlegend=False)
            st.plotly_chart(fig_regional, use_container_width=True)
        
        # Recent vs historical
        if len(regional_df) > 0:
            st.subheader("Recent vs Historical Growth (selected)")
            # Assign distinct colors per region for grouped bars
            region_list = regional_df['Region'].tolist()
            palette_map = {region: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, region in enumerate(sorted(set(region_list)))}
            colors_all_time = [palette_map[r] for r in region_list]
            colors_recent = [palette_map[r] for r in region_list]

            fig_recent = go.Figure()
            fig_recent.add_trace(go.Bar(
                x=regional_df['Region'],
                y=regional_df['mean_growth'],
                name='All Time',
                marker_color=colors_all_time
            ))
            fig_recent.add_trace(go.Bar(
                x=regional_df['Region'],
                y=regional_df['recent_5y_mean'],
                name='Last 5 Years',
                marker_color=colors_recent
            ))
            apply_minimal_theme(fig_recent)
            fig_recent.update_layout(barmode='group')
            st.plotly_chart(fig_recent, use_container_width=True)
    
    elif analysis_mode == "Ensemble Forecasting":
        st.markdown("---")
        st.header("Ensemble Forecasting")
        st.write("Advanced forecasting combining multiple ML methods with scenario analysis.")
        
        if not countries:
            st.warning("Please select at least one country.")
        else:
            forecast_horizon = st.slider("Forecast Horizon (years)", 3, 10, 5)
            
            for country in countries[:3]:
                with st.expander(f"Forecast: {country}", expanded=True):
                    with st.spinner(f"Generating ensemble forecast for {country}..."):
                        result = forecast_ensemble(
                            df, 
                            country, 
                            forecast_periods=forecast_horizon,
                            gdp_col=metric_col
                        )
                        
                        if result.get('success', False):
                            result = generate_scenarios(result)
                            result = calculate_forecast_confidence(result)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Methods Used", len(result['methods_used']))
                            with col2:
                                st.metric("Confidence", f"{result['confidence']['agreement']*100:.1f}%")
                            with col3:
                                st.metric("Uncertainty", f"±{result['confidence']['uncertainty']:.2f}%")
                            
                            historical = df[df['Entity'] == country].sort_values('Year')
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=historical['Year'],
                                y=historical[metric_col],
                                mode='lines',
                                name='Historical',
                                line=dict(color='#3b82f6', width=2)
                            ))
                            
                            scenarios = result['scenarios']
                            years = result['forecast_years']
                            
                            fig.add_trace(go.Scatter(
                                x=years,
                                y=scenarios['optimistic'],
                                mode='lines',
                                name='Optimistic',
                                line=dict(color='#10b981', dash='dash')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=years,
                                y=scenarios['baseline'],
                                mode='lines+markers',
                                name='Baseline',
                                line=dict(color='#f59e0b', width=3)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=years,
                                y=scenarios['pessimistic'],
                                mode='lines',
                                name='Pessimistic',
                                line=dict(color='#ef4444', dash='dash')
                            ))
                            
                            apply_minimal_theme(fig)
                            fig.update_layout(yaxis_title='GDP Growth (%)', xaxis_title='Year')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.write("**Individual Model Forecasts:**")
                            for method in result['methods_used']:
                                forecast_vals = result['individual_forecasts'][method]['forecast']
                                st.write(f"- {method}: {forecast_vals[-1]:.2f}% (final year)")
                        else:
                            st.error(f"Forecast failed: {result.get('error', 'Unknown error')}")
    
    elif analysis_mode == "Event Impact Analysis":
        st.markdown("---")
        st.header("Event Impact Analysis")
        st.write("Analyze the impact of major economic and geopolitical events.")
        
        st.subheader("Event Timeline (selected)")
        event_timeline = analyze_event_timeline(sel, MAJOR_EVENTS, gdp_col=metric_col)
        if len(event_timeline) > 0:
            fig_events = px.scatter(
                event_timeline,
                x='year',
                y='global_avg_growth',
                size='severity_score',
                color='type',
                hover_data=['event', 'pct_negative_growth'],
                labels={'global_avg_growth': 'Selected Avg Growth (%)', 'year': 'Year'}
            )
            apply_minimal_theme(fig_events)
            st.plotly_chart(fig_events, use_container_width=True)
            st.dataframe(event_timeline, use_container_width=True)
        
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
        
        st.subheader("Resilience Analysis")
        crisis_years = [event['year'] for event in MAJOR_EVENTS.values()]
        resilience = identify_resilient_countries(sel, crisis_years, gdp_col=metric_col)
        
        if len(resilience) > 0:
            st.write("**Resilient Countries (selected)**")
            st.dataframe(
                resilience[['Entity', 'n_crises_observed', 'resilience_rate', 'avg_crisis_growth', 'resilience_score']],
                use_container_width=True
            )
        
        if len(resilience) > 0:
            fig_resilience = px.bar(
                resilience,
                x='Entity',
                y='resilience_score',
                color='resilience_rate',
                color_continuous_scale='Greys'
            )
            apply_minimal_theme(fig_resilience)
            fig_resilience.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_resilience, use_container_width=True)
    
    elif analysis_mode == "Growth Momentum":
        st.markdown("---")
        st.header("Growth Momentum Analysis")
        st.write("Analyze growth acceleration, momentum indicators, and structural breaks.")
        
        with st.spinner("Calculating momentum indicators..."):
            momentum_all = analyze_growth_momentum(df, gdp_col=metric_col)
            momentum = momentum_all[momentum_all['Entity'].isin(countries)]

        if len(momentum) == 0:
            st.info("No momentum data available for selected countries.")
        else:
            if len(countries) >= 5:
                st.subheader("Momentum Leaders (selected)")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Positive Acceleration**")
                    positive = momentum[momentum['acceleration'] > 0].nlargest(15, 'acceleration')
                    st.dataframe(
                        positive[['Entity', 'current_growth', 'acceleration', 'momentum_3y']],
                        use_container_width=True
                    )
                with col2:
                    st.write("**Negative Acceleration**")
                    negative = momentum[momentum['acceleration'] < 0].nsmallest(15, 'acceleration')
                    st.dataframe(
                        negative[['Entity', 'current_growth', 'acceleration', 'momentum_3y']],
                        use_container_width=True
                    )
            else:
                st.subheader("Selected Countries Momentum")
                st.dataframe(
                    momentum[['Entity', 'current_growth', 'acceleration', 'momentum_3y', 'momentum_5y', 'avg_growth_10y']],
                    use_container_width=True
                )

        fig_momentum = px.scatter(
            momentum,
            x='momentum_5y',
            y='current_growth',
            color='acceleration',
            size=abs(momentum['acceleration']) if len(momentum) > 0 else None,
            hover_data=['Entity'],
            color_continuous_scale='Greys',
            labels={'momentum_5y': '5-Year Momentum', 'current_growth': 'Current Growth (%)'}
        )
        apply_minimal_theme(fig_momentum)
        st.plotly_chart(fig_momentum, use_container_width=True)
        
        if countries:
            st.subheader("Structural Break Detection")
            for country in countries[:3]:
                with st.expander(f"{country} — Structural Breaks"):
                    breaks = detect_structural_breaks(df, gdp_col=metric_col, entity=country)
                    if 'error' not in breaks:
                        if breaks['break_points']:
                            for bp in breaks['break_points']:
                                st.write(f"**{bp['year']}**: Magnitude {bp['magnitude']:.2f}% (before: {bp['before_mean']:.2f}%, after: {bp['after_mean']:.2f}%)")
                        else:
                            st.info("No significant structural breaks detected.")
                    else:
                        st.warning(breaks['error'])
    
    elif analysis_mode == "Advanced Forecasting":
        st.markdown("---")
        st.header("Advanced Forecasting Models")
        st.write("Multi-model ensemble forecasting with Prophet, ARIMA, Random Forest, and Gradient Boosting.")
        
        if not countries:
            st.warning("Please select at least one country for forecasting.")
        else:
            selected_country = st.selectbox("Select country for forecast", countries)
            years_ahead = st.slider("Years to forecast", min_value=1, max_value=10, value=5)
            
            with st.spinner(f"Generating forecasts for {selected_country}..."):
                country_df = df[df['Entity'] == selected_country].copy()
                country_df = country_df.rename(columns={'Entity': 'Country'})
                
                ensemble_result = ensemble_forecast(country_df, selected_country, years_ahead=years_ahead)
                scenario_result = scenario_analysis(country_df, selected_country, years_ahead=years_ahead)
                multi_horizon = multi_horizon_forecast(country_df, selected_country)
            
            if ensemble_result:
                st.subheader("Ensemble Forecast")
                
                forecast_df = pd.DataFrame({
                    'Year': ensemble_result['years'],
                    'Prediction': ensemble_result['predictions'],
                    'Lower Bound': ensemble_result['lower_bound'],
                    'Upper Bound': ensemble_result['upper_bound']
                })
                
                fig_ensemble = go.Figure()
                
                historical = country_df[country_df['Year'] <= 2023].sort_values('Year')
                fig_ensemble.add_trace(go.Scatter(
                    x=historical['Year'],
                    y=historical['GDP_Growth'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='#374151', width=2)
                ))
                
                fig_ensemble.add_trace(go.Scatter(
                    x=forecast_df['Year'],
                    y=forecast_df['Prediction'],
                    mode='lines+markers',
                    name='Ensemble Forecast',
                    line=dict(color='#6b7280', width=2, dash='dash')
                ))
                
                fig_ensemble.add_trace(go.Scatter(
                    x=forecast_df['Year'].tolist() + forecast_df['Year'].tolist()[::-1],
                    y=forecast_df['Upper Bound'].tolist() + forecast_df['Lower Bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(107, 114, 128, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ))
                
                apply_minimal_theme(fig_ensemble)
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
            
            if scenario_result:
                st.subheader("Scenario Analysis")
                
                scenario_df = pd.DataFrame({
                    'Year': scenario_result['years'],
                    'Optimistic': scenario_result['optimistic'],
                    'Baseline': scenario_result['baseline'],
                    'Pessimistic': scenario_result['pessimistic']
                })
                
                fig_scenario = go.Figure()
                
                fig_scenario.add_trace(go.Scatter(
                    x=scenario_df['Year'], y=scenario_df['Optimistic'],
                    name='Optimistic', line=dict(color='#059669', width=2)
                ))
                fig_scenario.add_trace(go.Scatter(
                    x=scenario_df['Year'], y=scenario_df['Baseline'],
                    name='Baseline', line=dict(color='#374151', width=2, dash='dash')
                ))
                fig_scenario.add_trace(go.Scatter(
                    x=scenario_df['Year'], y=scenario_df['Pessimistic'],
                    name='Pessimistic', line=dict(color='#dc2626', width=2)
                ))
                
                apply_minimal_theme(fig_scenario)
                fig_scenario.update_layout(
                    title=f"{selected_country} - Growth Scenarios",
                    xaxis_title='Year',
                    yaxis_title='GDP Growth (%)'
                )
                st.plotly_chart(fig_scenario, use_container_width=True)
                
                st.dataframe(scenario_df, use_container_width=True)
            
            if multi_horizon:
                st.subheader("Multi-Horizon Forecasts")
                
                horizons_data = []
                for horizon, data in multi_horizon.items():
                    horizons_data.append({
                        'Horizon': horizon,
                        'Average Growth': f"{data['average_growth']:.2f}%",
                        'Years Covered': f"{data['years'][0]}-{data['years'][-1]}"
                    })
                
                st.dataframe(pd.DataFrame(horizons_data), use_container_width=True)
    
    elif analysis_mode == "Growth Regimes":
        st.markdown("---")
        st.header("Growth Regime Classification")
        st.write("Classify countries into growth regimes and identify transitions.")
        
        n_regimes = st.slider("Number of regimes", min_value=3, max_value=7, value=5)
        
        with st.spinner("Classifying growth regimes..."):
            country_df = df.copy()
            country_df = country_df.rename(columns={'Entity': 'Country'})
            
            regime_classification = classify_growth_regimes(country_df, n_regimes=n_regimes)
        
        if regime_classification:
            st.subheader("Regime Classification Results")
            
            regime_stats = regime_classification['regime_statistics']
            st.dataframe(
                regime_stats[['regime_name', 'n_countries', 'avg_mean_growth', 'avg_std_growth', 'avg_negative_ratio']],
                use_container_width=True
            )
            
            st.subheader("Country Regimes")
            country_regimes = regime_classification['country_regimes']
            
            if countries:
                selected_regimes = country_regimes[country_regimes['country'].isin(countries)]
                st.dataframe(selected_regimes, use_container_width=True)
            else:
                st.dataframe(country_regimes, use_container_width=True)
            
            fig_regimes = px.histogram(
                country_regimes,
                x='regime',
                color='regime',
                color_discrete_sequence=COLOR_PALETTE,
                labels={'regime': 'Growth Regime', 'count': 'Number of Countries'}
            )
            apply_minimal_theme(fig_regimes)
            fig_regimes.update_layout(showlegend=False)
            st.plotly_chart(fig_regimes, use_container_width=True)
            
            st.subheader("Policy Recommendations by Regime")
            policy_analysis = analyze_policy_effectiveness(country_df, regime_classification)
            if policy_analysis is not None:
                st.dataframe(policy_analysis, use_container_width=True)
            
            if countries:
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
        else:
            st.warning("Unable to perform regime classification with current data.")
    
    elif analysis_mode == "Causal Inference":
        st.markdown("---")
        st.header("Causal Inference Analysis")
        st.write("Analyze causal relationships and policy impacts using advanced econometric methods.")
        
        inference_type = st.selectbox(
            "Analysis Type",
            ["Policy Impact (Difference-in-Differences)", "Natural Experiments", "Stimulus Effectiveness"]
        )
        
        country_df = df.copy()
        country_df = country_df.rename(columns={'Entity': 'Country'})
        
        if inference_type == "Policy Impact (Difference-in-Differences)":
            st.subheader("Difference-in-Differences Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                treatment_countries_did = st.multiselect("Treatment Countries", options=countries if countries else get_countries(df)[:20])
            with col2:
                control_countries_did = st.multiselect("Control Countries", options=get_countries(df)[:50])
            
            event_year_did = st.slider("Event Year", min_value=1985, max_value=2020, value=2008)
            policy_name = st.text_input("Policy Name", value="Economic Reform")
            
            if st.button("Run DiD Analysis") and treatment_countries_did and control_countries_did:
                with st.spinner("Running difference-in-differences analysis..."):
                    from src.causal_inference import difference_in_differences
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
                    apply_minimal_theme(fig_did)
                    st.plotly_chart(fig_did, use_container_width=True)
                else:
                    st.error("Unable to perform analysis with selected parameters")
        
        elif inference_type == "Natural Experiments":
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
                    apply_minimal_theme(fig_experiments)
                    st.plotly_chart(fig_experiments, use_container_width=True)
                else:
                    st.info("No significant natural experiments detected with current threshold")
        
        elif inference_type == "Stimulus Effectiveness":
            st.subheader("Stimulus Program Evaluation")
            
            stimulus_countries_input = st.multiselect(
                "Countries that implemented stimulus", 
                options=countries if countries else get_countries(df)[:30]
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
    
    elif analysis_mode == "Country Comparison":
        st.markdown("---")
        st.header("Country Comparison Tool")
        st.write("Comprehensive side-by-side comparison of selected countries.")
        
        if len(countries) < 2:
            st.warning("Please select at least 2 countries for comparison.")
        else:
            with st.spinner("Generating comparison..."):
                comparison_df = compare_countries(df, countries, metric_col=metric_col)
            
            if comparison_df is not None:
                st.subheader("Summary Statistics")
                st.dataframe(comparison_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Average Growth Comparison")
                    fig_avg = px.bar(
                        comparison_df,
                        x='Country',
                        y='Historical Avg Growth',
                        color='Historical Avg Growth',
                        color_continuous_scale='Greys'
                    )
                    apply_minimal_theme(fig_avg, height=400)
                    st.plotly_chart(fig_avg, use_container_width=True)
                
                with col2:
                    st.subheader("Volatility Comparison")
                    fig_vol = px.bar(
                        comparison_df,
                        x='Country',
                        y='Volatility (Std Dev)',
                        color='Volatility (Std Dev)',
                        color_continuous_scale='Greys'
                    )
                    apply_minimal_theme(fig_vol, height=400)
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                if len(countries) == 2:
                    st.subheader("Detailed Side-by-Side Analysis")
                    sbs_result = side_by_side_comparison(df, countries[0], countries[1], metric_col=metric_col)
                    
                    if sbs_result:
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
                        fig_sbs.add_trace(go.Scatter(
                            x=merged_data['Year'],
                            y=merged_data[f'{metric_col}_{countries[0]}'],
                            mode='lines+markers',
                            name=countries[0],
                            line=dict(color='#374151', width=2)
                        ))
                        
                        fig_sbs.add_trace(go.Scatter(
                            x=merged_data['Year'],
                            y=merged_data[f'{metric_col}_{countries[1]}'],
                            mode='lines+markers',
                            name=countries[1],
                            line=dict(color='#6b7280', width=2)
                        ))
                        
                        apply_minimal_theme(fig_sbs)
                        fig_sbs.update_layout(
                            title=f"{countries[0]} vs {countries[1]}",
                            xaxis_title='Year',
                            yaxis_title='GDP Growth (%)'
                        )
                        st.plotly_chart(fig_sbs, use_container_width=True)
                
                st.subheader("Growth Trajectories")
                trajectories = growth_trajectory_comparison(df, countries, metric_col=metric_col, reference_year=2000)
                
                if trajectories:
                    fig_traj = go.Figure()
                    
                    for traj in trajectories:
                        fig_traj.add_trace(go.Scatter(
                            x=traj['Years'],
                            y=traj['Cumulative_Index'],
                            mode='lines+markers',
                            name=traj['Country']
                        ))
                    
                    apply_minimal_theme(fig_traj)
                    fig_traj.update_layout(
                        title='Cumulative Growth Index (Base = 100 in 2000)',
                        xaxis_title='Year',
                        yaxis_title='Index (2000=100)'
                    )
                    st.plotly_chart(fig_traj, use_container_width=True)
                
                if len(countries) >= 1:
                    st.subheader("Peer Group Analysis")
                    selected_for_peers = st.selectbox("Select country to find peers", countries)
                    
                    if st.button("Find Similar Countries"):
                        with st.spinner("Finding peer countries..."):
                            peers = peer_group_comparison(df, selected_for_peers, metric_col=metric_col, n_peers=10)
                        
                        if peers is not None:
                            st.write(f"**Countries most similar to {selected_for_peers}:**")
                            st.dataframe(peers, use_container_width=True)
            else:
                st.error("Unable to generate comparison")
    
    elif analysis_mode == "Growth Story":
        st.markdown("---")
        st.header("Growth Story Generator")
        st.write("Auto-generate narrative summaries of country economic performance.")
        
        if not countries:
            st.warning("Please select at least one country.")
        else:
            story_type = st.radio("Story Type", ["Individual Country", "Comparative Analysis"])
            
            if story_type == "Individual Country":
                selected_country = st.selectbox("Select country", countries)
                
                if st.button("Generate Story"):
                    with st.spinner(f"Generating growth story for {selected_country}..."):
                        story = generate_growth_story(df, selected_country, metric_col=metric_col)
                    
                    if story:
                        st.subheader(story['title'])
                        
                        for section in story['sections']:
                            with st.expander(section['title'], expanded=True):
                                st.write(section['content'])
                        
                        st.download_button(
                            label="Download Story (Markdown)",
                            data=format_story_as_text(story),
                            file_name=f"growth_story_{selected_country}.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error("Unable to generate story")
            
            else:
                if len(countries) < 2:
                    st.warning("Please select at least 2 countries for comparative story.")
                else:
                    if st.button("Generate Comparative Story"):
                        with st.spinner("Generating comparative story..."):
                            story = generate_comparative_story(df, countries, metric_col=metric_col)
                        
                        if story:
                            st.subheader(story['title'])
                            
                            for section in story['sections']:
                                with st.expander(section['title'], expanded=True):
                                    st.write(section['content'])
                            
                            st.download_button(
                                label="Download Story (Markdown)",
                                data=format_story_as_text(story),
                                file_name="comparative_growth_story.md",
                                mime="text/markdown"
                            )
                        else:
                            st.error("Unable to generate comparative story")
    
    elif analysis_mode == "Custom Report":
        st.markdown("---")
        st.header("Custom Report Builder")
        st.write("Create customized PDF-ready reports with selected charts and data.")
        
        if not countries:
            st.warning("Please select at least one country.")
        else:
            st.subheader("Report Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_type = st.selectbox(
                    "Report Template",
                    ["standard", "executive", "detailed"]
                )
            
            with col2:
                report_format = st.selectbox(
                    "Output Format",
                    ["markdown", "html"]
                )
            
            template = create_report_template(report_type)
            
            st.info(f"**{template['title']}**: {template['description']}")
            
            st.subheader("Select Sections")
            
            for i, section in enumerate(template['sections']):
                template['sections'][i]['include'] = st.checkbox(
                    section['title'],
                    value=section['include'],
                    key=f"section_{section['id']}"
                )
            
            if st.button("Generate Report"):
                with st.spinner("Building custom report..."):
                    report = build_custom_report(
                        df, 
                        countries, 
                        template, 
                        metric_col=metric_col
                    )
                
                if report:
                    st.success("Report generated successfully!")
                    
                    if report_format == 'markdown':
                        content = format_report_as_markdown(report)
                        mime_type = "text/markdown"
                        file_ext = "md"
                    else:
                        from src.report_builder import format_report_as_html
                        content = format_report_as_html(report)
                        mime_type = "text/html"
                        file_ext = "html"
                    
                    st.download_button(
                        label=f"Download Report ({file_ext.upper()})",
                        data=content,
                        file_name=f"gdp_report_{datetime.now().strftime('%Y%m%d')}.{file_ext}",
                        mime=mime_type
                    )
                    
                    with st.expander("Preview Report"):
                        if report_format == 'markdown':
                            st.markdown(content)
                        else:
                            st.components.v1.html(content, height=800, scrolling=True)
                else:
                    st.error("Unable to generate report")

st.markdown("---")
st.caption("Data source: International Monetary Fund (WEO) | Processed by Our World in Data")
