import pandas as pd

DATA_PATH = "dataset/real-gdp-growth.csv"


def load_gdp_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Clean column names
    df = df.rename(columns={
        'Gross domestic product, constant prices - Percent change - Observations': 'gdp_obs',
        'Gross domestic product, constant prices - Percent change - Forecasts': 'gdp_forecast'
    })
    return df


def get_countries(df: pd.DataFrame) -> list:
    return sorted(df['Entity'].dropna().unique())
