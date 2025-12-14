"""
GDP Growth Analysis - Refactored Application
Main entry point for the Streamlit application.
"""

import streamlit as st
from src.ui.theme import apply_theme
from src.ui.layout import render_sidebar, render_header, render_footer
from src.pages import get_page, PAGE_REGISTRY
from src.data_utils import load_gdp_data

# Page configuration
st.set_page_config(
    page_title="GDP Growth Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply theme
apply_theme()

# Load data
df = load_gdp_data()

# Render header
render_header()

# Render sidebar and get configuration
config = render_sidebar(df)

# Get and render the selected page
analysis_mode = config['analysis_mode']

if analysis_mode in PAGE_REGISTRY:
    # Render the page
    page = get_page(analysis_mode)
    page.render(df, config)
else:
    # Fallback for pages not yet migrated
    st.warning(f"⚠️ '{analysis_mode}' is not yet available in the refactored version.")
    st.info("This analysis mode is being migrated. Please use `explore_gdp.py` for now.")
    st.markdown("---")
    st.markdown("### Available Modes in Refactored App")
    for mode in PAGE_REGISTRY.keys():
        st.markdown(f"- ✅ {mode}")

# Render footer
render_footer()
