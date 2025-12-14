"""
Ensemble Forecasting Module
Combines multiple forecasting methods for robust predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def prepare_time_series(
    df: pd.DataFrame,
    entity: str,
    gdp_col: str = 'gdp_obs',
    entity_col: str = 'Entity'
) -> pd.Series:
    """
    Prepare time series data for forecasting.
    
    Args:
        df: DataFrame with GDP data
        entity: Entity name
        gdp_col: GDP column name
        entity_col: Entity column name
    
    Returns:
        Time series as pandas Series
    """
    entity_data = df[df[entity_col] == entity].sort_values('Year').copy()
    ts = pd.Series(
        entity_data[gdp_col].values,
        index=pd.DatetimeIndex(pd.to_datetime(entity_data['Year'], format='%Y'))
    )
    return ts


def forecast_arima(
    ts: pd.Series,
    forecast_periods: int = 5,
    order: Tuple[int, int, int] = (1, 1, 1)
) -> Dict:
    """
    ARIMA forecasting model.
    
    Args:
        ts: Time series data
        forecast_periods: Number of periods to forecast
        order: ARIMA order (p, d, q)
    
    Returns:
        Dictionary with forecast values and metadata
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        model = ARIMA(ts, order=order)
        fitted_model = model.fit()
        
        forecast = fitted_model.forecast(steps=forecast_periods)
        
        return {
            'method': 'ARIMA',
            'forecast': forecast.values,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'success': True
        }
    except Exception as e:
        return {
            'method': 'ARIMA',
            'forecast': None,
            'error': str(e),
            'success': False
        }


def forecast_prophet(
    df: pd.DataFrame,
    entity: str,
    forecast_periods: int = 5,
    gdp_col: str = 'gdp_obs',
    entity_col: str = 'Entity'
) -> Dict:
    """
    Prophet forecasting model.
    
    Args:
        df: DataFrame with GDP data
        entity: Entity name
        forecast_periods: Number of periods to forecast
        gdp_col: GDP column name
        entity_col: Entity column name
    
    Returns:
        Dictionary with forecast values and metadata
    """
    try:
        from prophet import Prophet
        
        entity_data = df[df[entity_col] == entity].sort_values('Year').copy()
        
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(entity_data['Year'], format='%Y'),
            'y': entity_data[gdp_col].values
        })
        
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=forecast_periods, freq='YS')
        forecast = model.predict(future)
        
        forecast_values = forecast.tail(forecast_periods)['yhat'].values
        
        return {
            'method': 'Prophet',
            'forecast': forecast_values,
            'lower_bound': forecast.tail(forecast_periods)['yhat_lower'].values,
            'upper_bound': forecast.tail(forecast_periods)['yhat_upper'].values,
            'success': True
        }
    except Exception as e:
        return {
            'method': 'Prophet',
            'forecast': None,
            'error': str(e),
            'success': False
        }


def forecast_xgboost(
    ts: pd.Series,
    forecast_periods: int = 5,
    n_lags: int = 5
) -> Dict:
    """
    XGBoost time series forecasting.
    
    Args:
        ts: Time series data
        forecast_periods: Number of periods to forecast
        n_lags: Number of lag features
    
    Returns:
        Dictionary with forecast values and metadata
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error
        
        values = ts.values
        
        X, y = [], []
        for i in range(n_lags, len(values)):
            X.append(values[i-n_lags:i])
            y.append(values[i])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) < 10:
            return {
                'method': 'XGBoost',
                'forecast': None,
                'error': 'Insufficient data',
                'success': False
            }
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        
        forecasts = []
        last_values = values[-n_lags:].tolist()
        
        for _ in range(forecast_periods):
            next_pred = model.predict(np.array([last_values]))[0]
            forecasts.append(next_pred)
            last_values.append(next_pred)
            last_values.pop(0)
        
        return {
            'method': 'XGBoost',
            'forecast': np.array(forecasts),
            'success': True
        }
    except Exception as e:
        return {
            'method': 'XGBoost',
            'forecast': None,
            'error': str(e),
            'success': False
        }


def forecast_exponential_smoothing(
    ts: pd.Series,
    forecast_periods: int = 5
) -> Dict:
    """
    Exponential smoothing forecasting.
    
    Args:
        ts: Time series data
        forecast_periods: Number of periods to forecast
    
    Returns:
        Dictionary with forecast values and metadata
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        model = ExponentialSmoothing(
            ts,
            trend='add',
            seasonal=None,
            damped_trend=True
        )
        fitted_model = model.fit()
        
        forecast = fitted_model.forecast(steps=forecast_periods)
        
        return {
            'method': 'Exponential_Smoothing',
            'forecast': forecast.values,
            'aic': fitted_model.aic,
            'success': True
        }
    except Exception as e:
        return {
            'method': 'Exponential_Smoothing',
            'forecast': None,
            'error': str(e),
            'success': False
        }


def forecast_ensemble(
    df: pd.DataFrame,
    entity: str,
    forecast_periods: int = 5,
    gdp_col: str = 'gdp_obs',
    entity_col: str = 'Entity',
    methods: List[str] = None,
    weights: Dict[str, float] = None
) -> Dict:
    """
    Ensemble forecast combining multiple methods.
    
    Args:
        df: DataFrame with GDP data
        entity: Entity name
        forecast_periods: Number of periods to forecast
        gdp_col: GDP column name
        entity_col: Entity column name
        methods: List of methods to use (default: all)
        weights: Custom weights for each method
    
    Returns:
        Dictionary with ensemble forecast and individual forecasts
    """
    if methods is None:
        methods = ['ARIMA', 'Prophet', 'XGBoost', 'Exponential_Smoothing']
    
    ts = prepare_time_series(df, entity, gdp_col, entity_col)
    
    if len(ts) < 10:
        return {
            'entity': entity,
            'error': 'Insufficient data for forecasting',
            'success': False
        }
    
    forecasts = {}
    
    if 'ARIMA' in methods:
        forecasts['ARIMA'] = forecast_arima(ts, forecast_periods)
    
    if 'Prophet' in methods:
        forecasts['Prophet'] = forecast_prophet(df, entity, forecast_periods, gdp_col, entity_col)
    
    if 'XGBoost' in methods:
        forecasts['XGBoost'] = forecast_xgboost(ts, forecast_periods)
    
    if 'Exponential_Smoothing' in methods:
        forecasts['Exponential_Smoothing'] = forecast_exponential_smoothing(ts, forecast_periods)
    
    successful_forecasts = {
        method: result['forecast'] 
        for method, result in forecasts.items() 
        if result.get('success', False) and result['forecast'] is not None
    }
    
    if not successful_forecasts:
        return {
            'entity': entity,
            'error': 'All forecast methods failed',
            'individual_forecasts': forecasts,
            'success': False
        }
    
    if weights is None:
        weights = {method: 1.0 / len(successful_forecasts) for method in successful_forecasts}
    
    ensemble_forecast = np.zeros(forecast_periods)
    for method, forecast_values in successful_forecasts.items():
        weight = weights.get(method, 1.0 / len(successful_forecasts))
        if len(forecast_values) == forecast_periods and not np.any(np.isnan(forecast_values)):
            ensemble_forecast += weight * forecast_values
        else:
            ensemble_forecast += weight * np.nan_to_num(forecast_values[:forecast_periods], nan=0.0)
    
    last_year = df[df[entity_col] == entity]['Year'].max()
    forecast_years = [last_year + i + 1 for i in range(forecast_periods)]
    
    return {
        'entity': entity,
        'forecast_years': forecast_years,
        'ensemble_forecast': ensemble_forecast,
        'individual_forecasts': forecasts,
        'methods_used': list(successful_forecasts.keys()),
        'success': True
    }


def generate_scenarios(
    ensemble_result: Dict,
    volatility: float = None
) -> Dict:
    """
    Generate optimistic, baseline, and pessimistic scenarios.
    
    Args:
        ensemble_result: Result from ensemble_forecast
        volatility: Historical volatility (std dev)
    
    Returns:
        Dictionary with scenarios
    """
    if not ensemble_result.get('success', False):
        return ensemble_result
    
    baseline = ensemble_result['ensemble_forecast']
    
    if volatility is None:
        individual_forecasts = ensemble_result['individual_forecasts']
        successful = [
            f['forecast'] for f in individual_forecasts.values() 
            if f.get('success', False) and f['forecast'] is not None
        ]
        valid_successful = [f for f in successful if not np.any(np.isnan(f))]
        if valid_successful:
            volatility = np.std([np.mean(f) for f in valid_successful])
        else:
            volatility = 2.0
    
    optimistic = baseline + 1.5 * volatility
    pessimistic = baseline - 1.5 * volatility
    
    ensemble_result['scenarios'] = {
        'baseline': baseline,
        'optimistic': optimistic,
        'pessimistic': pessimistic,
        'volatility': volatility
    }
    
    return ensemble_result


def calculate_forecast_confidence(
    ensemble_result: Dict
) -> Dict:
    """
    Calculate confidence metrics for ensemble forecast.
    
    Args:
        ensemble_result: Result from ensemble_forecast
    
    Returns:
        Dictionary with confidence metrics
    """
    if not ensemble_result.get('success', False):
        return ensemble_result
    
    individual_forecasts = ensemble_result['individual_forecasts']
    successful = [
        f['forecast'] for f in individual_forecasts.values() 
        if f.get('success', False) and f['forecast'] is not None
    ]
    
    if len(successful) < 2:
        ensemble_result['confidence'] = {
            'agreement': 0.0,
            'uncertainty': 100.0
        }
        return ensemble_result
    
    forecast_array = np.array(successful)
    
    valid_forecasts = []
    for f in forecast_array:
        if not np.any(np.isnan(f)):
            valid_forecasts.append(f)
    
    if len(valid_forecasts) < 2:
        ensemble_result['confidence'] = {
            'agreement': 0.5,
            'uncertainty': 5.0
        }
        return ensemble_result
    
    valid_array = np.array(valid_forecasts)
    std_across_methods = np.std(valid_array, axis=0)
    mean_std = np.mean(std_across_methods)
    
    agreement = 1.0 / (1.0 + mean_std)
    uncertainty = mean_std
    
    ensemble_result['confidence'] = {
        'agreement': agreement,
        'uncertainty': uncertainty,
        'n_methods': len(successful)
    }
    
    return ensemble_result
