import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.advanced_forecasting import (
    ensemble_forecast,
    scenario_analysis,
    multi_horizon_forecast,
    prophet_forecast,
    arima_style_forecast,
    ml_forecast
)
from src.growth_regime_classifier import (
    classify_growth_regimes,
    identify_early_warning_signals,
    calculate_growth_features
)
from src.causal_inference import (
    difference_in_differences,
    natural_experiment_finder,
    granger_causality_test
)


class TestPhase2Forecasting(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        years = list(range(2000, 2024))
        cls.test_data = pd.DataFrame({
            'Country': ['United States'] * len(years) + ['China'] * len(years),
            'Year': years * 2,
            'GDP_Growth': [2.0 + np.random.randn() for _ in range(len(years))] + 
                         [8.0 + np.random.randn() * 2 for _ in range(len(years))]
        })
    
    def test_prophet_forecast(self):
        result = prophet_forecast(self.test_data, 'United States', years_ahead=5)
        self.assertIsNotNone(result)
        self.assertEqual(len(result['predictions']), 5)
        self.assertIn('model', result)
        self.assertEqual(result['model'], 'Prophet')
    
    def test_arima_forecast(self):
        result = arima_style_forecast(self.test_data, 'United States', years_ahead=5)
        self.assertIsNotNone(result)
        self.assertEqual(len(result['predictions']), 5)
        self.assertIn('lower_bound', result)
        self.assertIn('upper_bound', result)
    
    def test_ml_forecast(self):
        result = ml_forecast(self.test_data, 'United States', years_ahead=5, model_type='rf')
        self.assertIsNotNone(result)
        self.assertEqual(len(result['predictions']), 5)
        self.assertEqual(result['model'], 'Random Forest')
    
    def test_ensemble_forecast(self):
        result = ensemble_forecast(self.test_data, 'United States', years_ahead=5)
        self.assertIsNotNone(result)
        self.assertEqual(len(result['predictions']), 5)
        self.assertIn('individual_forecasts', result)
        self.assertTrue(len(result['individual_forecasts']) > 0)
    
    def test_scenario_analysis(self):
        result = scenario_analysis(self.test_data, 'United States', years_ahead=5)
        self.assertIsNotNone(result)
        self.assertIn('optimistic', result)
        self.assertIn('baseline', result)
        self.assertIn('pessimistic', result)
        self.assertEqual(len(result['optimistic']), 5)
    
    def test_multi_horizon_forecast(self):
        result = multi_horizon_forecast(self.test_data, 'United States')
        self.assertIsNotNone(result)
        self.assertIn('1-year', result)
        self.assertIn('5-year', result)


class TestPhase2GrowthRegimes(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        countries = ['USA', 'China', 'Germany', 'Brazil', 'India', 'Japan', 'Mexico', 'France']
        years = list(range(2000, 2024))
        
        data = []
        for country in countries:
            base_growth = np.random.uniform(1, 6)
            volatility = np.random.uniform(1, 4)
            
            for year in years:
                growth = base_growth + np.random.randn() * volatility
                data.append({'Country': country, 'Year': year, 'GDP_Growth': growth})
        
        cls.test_data = pd.DataFrame(data)
    
    def test_classify_growth_regimes(self):
        result = classify_growth_regimes(self.test_data, n_regimes=4)
        self.assertIsNotNone(result)
        self.assertIn('country_regimes', result)
        self.assertIn('regime_statistics', result)
        
        country_regimes = result['country_regimes']
        self.assertEqual(len(country_regimes), 8)
    
    def test_calculate_growth_features(self):
        result = calculate_growth_features(self.test_data, 'USA')
        self.assertIsNotNone(result)
        self.assertIn('mean_growth', result)
        self.assertIn('std_growth', result)
        self.assertIn('cv_growth', result)
    
    def test_early_warning_signals(self):
        result = identify_early_warning_signals(self.test_data, 'USA', lookback=3)
        self.assertIsNotNone(result)
        self.assertIn('warnings', result)
        self.assertIn('risk_score', result)
        self.assertIn('risk_level', result)


class TestPhase2CausalInference(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        countries = ['Treatment1', 'Treatment2', 'Control1', 'Control2', 'Control3']
        years = list(range(2000, 2020))
        
        data = []
        for country in countries:
            is_treatment = 'Treatment' in country
            
            for year in years:
                if is_treatment and year >= 2010:
                    base_growth = 4.0 + np.random.randn()
                elif is_treatment:
                    base_growth = 2.0 + np.random.randn()
                else:
                    base_growth = 2.5 + np.random.randn()
                
                data.append({'Country': country, 'Year': year, 'GDP_Growth': base_growth})
        
        cls.test_data = pd.DataFrame(data)
    
    def test_difference_in_differences(self):
        result = difference_in_differences(
            self.test_data,
            ['Treatment1', 'Treatment2'],
            ['Control1', 'Control2', 'Control3'],
            event_year=2010,
            pre_years=3,
            post_years=3
        )
        
        self.assertIsNotNone(result)
        self.assertIn('did_estimate', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)
        self.assertIn('interpretation', result)
    
    def test_natural_experiment_finder(self):
        data = []
        for country in ['A', 'B', 'C', 'D']:
            for year in range(2000, 2020):
                if year == 2010 and country in ['A', 'B']:
                    growth = 8.0 + np.random.randn()
                else:
                    growth = 2.0 + np.random.randn()
                data.append({'Country': country, 'Year': year, 'GDP_Growth': growth})
        
        test_df = pd.DataFrame(data)
        result = natural_experiment_finder(test_df, min_countries=2, threshold_change=3)
        
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, list))
    
    def test_granger_causality(self):
        years = list(range(2000, 2024))
        data = pd.DataFrame({
            'Country': ['A'] * len(years) + ['B'] * len(years),
            'Year': years * 2,
            'GDP_Growth': [2 + np.random.randn() for _ in range(len(years))] + 
                         [2 + np.random.randn() for _ in range(len(years))]
        })
        
        result = granger_causality_test(data, 'A', 'B', max_lag=3)
        self.assertIsNotNone(result)
        self.assertIn('causes', result)
        self.assertIn('conclusion', result)


class TestPhase2Integration(unittest.TestCase):
    
    def test_forecasting_with_insufficient_data(self):
        small_data = pd.DataFrame({
            'Country': ['X', 'X'],
            'Year': [2022, 2023],
            'GDP_Growth': [2.0, 2.5]
        })
        
        result = ensemble_forecast(small_data, 'X', years_ahead=5)
        self.assertIsNone(result)
    
    def test_regime_classification_with_minimal_countries(self):
        minimal_data = pd.DataFrame({
            'Country': ['A', 'B'],
            'Year': [2020, 2020],
            'GDP_Growth': [2.0, 3.0]
        })
        
        result = classify_growth_regimes(minimal_data, n_regimes=5)
        self.assertIsNone(result)


def run_phase2_tests():
    suite = unittest.TestSuite()
    
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPhase2Forecasting))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPhase2GrowthRegimes))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPhase2CausalInference))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPhase2Integration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_phase2_tests()
    sys.exit(0 if success else 1)
