"""
Test Migration - Validate all refactored pages
"""

import sys
import traceback

def test_imports():
    """Test that all pages can be imported."""
    print("=" * 60)
    print("Testing Page Imports")
    print("=" * 60)
    
    pages_to_test = [
        ("overview", "OverviewPage"),
        ("volatility", "VolatilityPage"),
        ("clustering", "ClusteringPage"),
        ("forecasting", "ForecastingPage"),
        ("anomaly_detection", "AnomalyDetectionPage"),
        ("regional_comparison", "RegionalComparisonPage"),
        ("event_impact", "EventImpactPage"),
        ("growth_momentum", "GrowthMomentumPage"),
        ("ensemble_forecasting", "EnsembleForecastingPage"),
        ("advanced_forecasting", "AdvancedForecastingPage"),
        ("growth_regimes", "GrowthRegimesPage"),
        ("causal_inference", "CausalInferencePage"),
        ("country_comparison", "CountryComparisonPage"),
        ("growth_story", "GrowthStoryPage"),
        ("custom_report", "CustomReportPage"),
    ]
    
    successful = 0
    failed = 0
    
    for module_name, class_name in pages_to_test:
        try:
            module = __import__(f'src.pages.{module_name}', fromlist=[class_name])
            page_class = getattr(module, class_name)
            page_instance = page_class()
            print(f"✅ {module_name:30} → {class_name:30} OK")
            successful += 1
        except Exception as e:
            print(f"❌ {module_name:30} → {class_name:30} FAILED")
            print(f"   Error: {str(e)[:60]}")
            failed += 1
    
    print(f"\n{successful}/{len(pages_to_test)} pages imported successfully")
    return failed == 0


def test_page_registry():
    """Test that PAGE_REGISTRY contains all pages."""
    print("\n" + "=" * 60)
    print("Testing PAGE_REGISTRY")
    print("=" * 60)
    
    try:
        from src.pages import PAGE_REGISTRY
        
        expected_modes = [
            "Overview",
            "Volatility Analysis",
            "Clustering",
            "Forecasting",
            "Anomaly Detection",
            "Regional Comparison",
            "Event Impact Analysis",
            "Growth Momentum",
            "Ensemble Forecasting",
            "Advanced Forecasting",
            "Growth Regimes",
            "Causal Inference",
            "Country Comparison",
            "Growth Story",
            "Custom Report"
        ]
        
        print(f"Expected pages: {len(expected_modes)}")
        print(f"Registered pages: {len(PAGE_REGISTRY)}")
        
        all_present = True
        for mode in expected_modes:
            if mode in PAGE_REGISTRY:
                print(f"✅ {mode}")
            else:
                print(f"❌ {mode} - NOT FOUND")
                all_present = False
        
        return all_present
    
    except Exception as e:
        print(f"❌ Failed to test PAGE_REGISTRY: {str(e)}")
        traceback.print_exc()
        return False


def test_constants():
    """Test that ANALYSIS_MODES constants are correct."""
    print("\n" + "=" * 60)
    print("Testing Constants")
    print("=" * 60)
    
    try:
        from src.config.constants import ANALYSIS_MODES, COLOR_PALETTE
        
        print(f"✅ ANALYSIS_MODES loaded: {len(ANALYSIS_MODES)} modes")
        for mode in ANALYSIS_MODES:
            print(f"   - {mode}")
        
        print(f"\n✅ COLOR_PALETTE loaded: {len(COLOR_PALETTE)} colors")
        
        return len(ANALYSIS_MODES) == 15
    
    except Exception as e:
        print(f"❌ Failed to load constants: {str(e)}")
        return False


def test_ui_components():
    """Test that UI components can be imported."""
    print("\n" + "=" * 60)
    print("Testing UI Components")
    print("=" * 60)
    
    components = [
        ("src.ui.theme", "apply_theme"),
        ("src.ui.layout", "render_sidebar"),
        ("src.ui.charts", "ChartBuilder"),
    ]
    
    all_ok = True
    for module_name, component_name in components:
        try:
            module = __import__(module_name, fromlist=[component_name])
            component = getattr(module, component_name)
            print(f"✅ {module_name:30} → {component_name}")
        except Exception as e:
            print(f"❌ {module_name:30} → {component_name}")
            print(f"   Error: {str(e)[:60]}")
            all_ok = False
    
    return all_ok


def test_base_page():
    """Test that BasePage is working."""
    print("\n" + "=" * 60)
    print("Testing BasePage")
    print("=" * 60)
    
    try:
        from src.pages.base import BasePage
        
        # Try to create a simple test page
        class TestPage(BasePage):
            def __init__(self):
                super().__init__("Test Page", "Test Description")
            
            def _render_content(self, df, config):
                pass
        
        test_page = TestPage()
        print(f"✅ BasePage inheritance works")
        print(f"   Title: {test_page.title}")
        print(f"   Description: {test_page.description}")
        return True
    
    except Exception as e:
        print(f"❌ BasePage test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + "MIGRATION TEST SUITE - Refactored Pages".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = {
        "Imports": test_imports(),
        "PAGE_REGISTRY": test_page_registry(),
        "Constants": test_constants(),
        "UI Components": test_ui_components(),
        "BasePage": test_base_page(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    print(f"\n{passed}/{total} test groups passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Refactoring is complete and working.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test group(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
