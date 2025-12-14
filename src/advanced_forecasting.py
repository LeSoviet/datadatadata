import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def prepare_forecast_data(df, country, years_history=10):
    country_data = df[df['Country'] == country].copy()
    country_data = country_data.sort_values('Year')
    
    historical = country_data[country_data['Year'] <= 2023]
    
    if len(historical) > years_history:
        historical = historical.tail(years_history)
    
    return historical


def prophet_forecast(df, country, years_ahead=5):
    historical = prepare_forecast_data(df, country)
    
    if len(historical) < 3:
        return None
    
    prophet_df = historical[['Year', 'GDP_Growth']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')
    
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=years_ahead, freq='YS')
    forecast = model.predict(future)
    
    forecast['year'] = forecast['ds'].dt.year
    forecast = forecast[forecast['year'] > 2023].copy()
    
    return {
        'model': 'Prophet',
        'years': forecast['year'].tolist(),
        'predictions': forecast['yhat'].tolist(),
        'lower_bound': forecast['yhat_lower'].tolist(),
        'upper_bound': forecast['yhat_upper'].tolist()
    }


def arima_style_forecast(df, country, years_ahead=5):
    historical = prepare_forecast_data(df, country)
    
    if len(historical) < 5:
        return None
    
    growth_values = historical['GDP_Growth'].values
    
    ma_window = min(3, len(growth_values) - 1)
    trend = np.convolve(growth_values, np.ones(ma_window) / ma_window, mode='valid')
    
    if len(trend) == 0:
        last_value = growth_values[-1]
        trend_slope = 0
    else:
        last_value = trend[-1]
        if len(trend) >= 2:
            trend_slope = (trend[-1] - trend[0]) / len(trend)
        else:
            trend_slope = 0
    
    predictions = []
    current_value = last_value
    
    for i in range(years_ahead):
        current_value = current_value + trend_slope * 0.5
        predictions.append(current_value)
    
    std_dev = np.std(growth_values)
    lower = [p - 1.96 * std_dev for p in predictions]
    upper = [p + 1.96 * std_dev for p in predictions]
    
    return {
        'model': 'ARIMA-style',
        'years': list(range(2024, 2024 + years_ahead)),
        'predictions': predictions,
        'lower_bound': lower,
        'upper_bound': upper
    }


def ml_forecast(df, country, years_ahead=5, model_type='rf'):
    historical = prepare_forecast_data(df, country, years_history=15)
    
    if len(historical) < 8:
        return None
    
    historical = historical.copy()
    historical['lag_1'] = historical['GDP_Growth'].shift(1)
    historical['lag_2'] = historical['GDP_Growth'].shift(2)
    historical['lag_3'] = historical['GDP_Growth'].shift(3)
    historical['rolling_mean_3'] = historical['GDP_Growth'].rolling(window=3).mean()
    historical['rolling_std_3'] = historical['GDP_Growth'].rolling(window=3).std()
    historical['year_index'] = range(len(historical))
    
    historical = historical.dropna()
    
    if len(historical) < 5:
        return None
    
    features = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'year_index']
    X = historical[features].values
    y = historical['GDP_Growth'].values
    
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    
    model.fit(X, y)
    
    predictions = []
    current_features = X[-1].copy()
    
    last_values = historical['GDP_Growth'].tail(3).values.tolist()
    
    for i in range(years_ahead):
        pred = model.predict(current_features.reshape(1, -1))[0]
        predictions.append(pred)
        
        last_values.append(pred)
        if len(last_values) > 3:
            last_values.pop(0)
        
        current_features[0] = last_values[-1] if len(last_values) >= 1 else pred
        current_features[1] = last_values[-2] if len(last_values) >= 2 else current_features[0]
        current_features[2] = last_values[-3] if len(last_values) >= 3 else current_features[1]
        current_features[3] = np.mean(last_values)
        current_features[4] = np.std(last_values) if len(last_values) > 1 else 0
        current_features[5] += 1
    
    std_dev = np.std(y)
    lower = [p - 1.96 * std_dev for p in predictions]
    upper = [p + 1.96 * std_dev for p in predictions]
    
    model_name = 'Random Forest' if model_type == 'rf' else 'Gradient Boosting'
    
    return {
        'model': model_name,
        'years': list(range(2024, 2024 + years_ahead)),
        'predictions': predictions,
        'lower_bound': lower,
        'upper_bound': upper
    }


def ensemble_forecast(df, country, years_ahead=5):
    forecasts = []
    
    prophet_result = prophet_forecast(df, country, years_ahead)
    if prophet_result:
        forecasts.append(prophet_result)
    
    arima_result = arima_style_forecast(df, country, years_ahead)
    if arima_result:
        forecasts.append(arima_result)
    
    rf_result = ml_forecast(df, country, years_ahead, model_type='rf')
    if rf_result:
        forecasts.append(rf_result)
    
    gb_result = ml_forecast(df, country, years_ahead, model_type='gb')
    if gb_result:
        forecasts.append(gb_result)
    
    if not forecasts:
        return None
    
    all_predictions = np.array([f['predictions'] for f in forecasts])
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    all_lower = np.array([f['lower_bound'] for f in forecasts])
    ensemble_lower = np.mean(all_lower, axis=0)
    
    all_upper = np.array([f['upper_bound'] for f in forecasts])
    ensemble_upper = np.mean(all_upper, axis=0)
    
    ensemble_result = {
        'model': 'Ensemble',
        'years': list(range(2024, 2024 + years_ahead)),
        'predictions': ensemble_predictions.tolist(),
        'lower_bound': ensemble_lower.tolist(),
        'upper_bound': ensemble_upper.tolist(),
        'individual_forecasts': forecasts
    }
    
    return ensemble_result


def scenario_analysis(df, country, years_ahead=5):
    ensemble = ensemble_forecast(df, country, years_ahead)
    
    if not ensemble:
        return None
    
    baseline = ensemble['predictions']
    
    historical = prepare_forecast_data(df, country, years_history=10)
    growth_volatility = historical['GDP_Growth'].std()
    
    optimistic = [p + 0.7 * growth_volatility for p in baseline]
    pessimistic = [p - 0.7 * growth_volatility for p in baseline]
    
    return {
        'years': ensemble['years'],
        'optimistic': optimistic,
        'baseline': baseline,
        'pessimistic': pessimistic,
        'lower_bound': ensemble['lower_bound'],
        'upper_bound': ensemble['upper_bound']
    }


def multi_horizon_forecast(df, country):
    horizons = {
        '1-year': 1,
        '3-year': 3,
        '5-year': 5,
        '10-year': 10
    }
    
    results = {}
    
    for name, years in horizons.items():
        forecast = ensemble_forecast(df, country, years_ahead=years)
        if forecast:
            avg_growth = np.mean(forecast['predictions'])
            results[name] = {
                'average_growth': avg_growth,
                'predictions': forecast['predictions'],
                'years': forecast['years']
            }
    
    return results


def evaluate_forecast_accuracy(df, country, test_year=2020):
    historical = df[(df['Country'] == country) & (df['Year'] < test_year)].copy()
    actual = df[(df['Country'] == country) & (df['Year'] >= test_year) & (df['Year'] <= 2023)].copy()
    
    if len(historical) < 5 or len(actual) == 0:
        return None
    
    years_ahead = len(actual)
    
    temp_df = pd.concat([historical, actual])
    
    forecasts = {
        'Prophet': prophet_forecast(historical, country, years_ahead),
        'ARIMA-style': arima_style_forecast(historical, country, years_ahead),
        'Random Forest': ml_forecast(historical, country, years_ahead, model_type='rf'),
        'Gradient Boosting': ml_forecast(historical, country, years_ahead, model_type='gb'),
        'Ensemble': ensemble_forecast(historical, country, years_ahead)
    }
    
    actual_values = actual['GDP_Growth'].values
    
    results = {}
    
    for model_name, forecast in forecasts.items():
        if forecast is None:
            continue
        
        predictions = forecast['predictions'][:len(actual_values)]
        
        if len(predictions) < len(actual_values):
            continue
        
        mae = mean_absolute_error(actual_values, predictions)
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))
        
        try:
            r2 = r2_score(actual_values, predictions)
        except:
            r2 = None
        
        results[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
    
    return results
