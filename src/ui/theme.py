"""
Theme Configuration
Manages visual theming, colors, and styling for the application.
"""

import streamlit as st
from src.config.constants import COLOR_PALETTE


def get_color_palette():
    """
    Get the application color palette.

    Returns:
        list: List of hex color codes
    """
    return COLOR_PALETTE


def get_theme_colors():
    """
    Get theme color tokens for UI elements.

    Returns:
        dict: Dictionary of theme color variables
    """
    return {
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


def get_plot_theme():
    """
    Get theme configuration for Plotly charts.

    Returns:
        dict: Plot theme configuration
    """
    return {
        'plot_bg': '#ffffff',
        'paper_bg': '#fafafa',
        'axis_grid': '#e5e7eb',
        'font_color': '#1f2937',
    }


def apply_theme():
    """
    Apply custom CSS theme to the Streamlit application.
    Injects minimal light theme styling.
    """
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
