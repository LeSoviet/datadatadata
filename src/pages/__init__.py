"""
Pages Module
Individual analysis pages for the GDP Analysis application.
"""

from .overview import OverviewPage
from .volatility import VolatilityPage
from .clustering import ClusteringPage
from .forecasting import ForecastingPage
from .anomaly_detection import AnomalyDetectionPage
from .regional_comparison import RegionalComparisonPage
from .event_impact import EventImpactPage
from .growth_momentum import GrowthMomentumPage
from .ensemble_forecasting import EnsembleForecastingPage
from .advanced_forecasting import AdvancedForecastingPage
from .growth_regimes import GrowthRegimesPage
from .causal_inference import CausalInferencePage
from .country_comparison import CountryComparisonPage
from .growth_story import GrowthStoryPage
from .custom_report import CustomReportPage

# Registry of available pages
# Maps analysis mode name to page instance
PAGE_REGISTRY = {
    "Overview": OverviewPage(),
    "Volatility Analysis": VolatilityPage(),
    "Clustering": ClusteringPage(),
    "Forecasting": ForecastingPage(),
    "Anomaly Detection": AnomalyDetectionPage(),
    "Regional Comparison": RegionalComparisonPage(),
    "Event Impact Analysis": EventImpactPage(),
    "Growth Momentum": GrowthMomentumPage(),
    "Ensemble Forecasting": EnsembleForecastingPage(),
    "Advanced Forecasting": AdvancedForecastingPage(),
    "Growth Regimes": GrowthRegimesPage(),
    "Causal Inference": CausalInferencePage(),
    "Country Comparison": CountryComparisonPage(),
    "Growth Story": GrowthStoryPage(),
    "Custom Report": CustomReportPage(),
}

def get_page(analysis_mode: str):
    """
    Get page instance by analysis mode name.

    Args:
        analysis_mode: Name of the analysis mode

    Returns:
        BasePage instance

    Raises:
        ValueError: If analysis mode is not registered
    """
    if analysis_mode not in PAGE_REGISTRY:
        raise ValueError(f"Unknown analysis mode: {analysis_mode}")
    return PAGE_REGISTRY[analysis_mode]

__all__ = [
    'PAGE_REGISTRY',
    'get_page',
    'OverviewPage',
    'VolatilityPage',
    'ClusteringPage',
    'ForecastingPage',
    'AnomalyDetectionPage',
    'RegionalComparisonPage',
    'EventImpactPage',
    'GrowthMomentumPage',
    'EnsembleForecastingPage',
    'AdvancedForecastingPage',
    'GrowthRegimesPage',
    'CausalInferencePage',
    'CountryComparisonPage',
    'GrowthStoryPage',
    'CustomReportPage',
]
