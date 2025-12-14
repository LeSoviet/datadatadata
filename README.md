# 📊 GDP Growth Analysis Dashboard

A comprehensive Streamlit application for analyzing global GDP growth trends, volatility, forecasting, and economic insights.

**Status**: ✅ 100% Refactored | Production Ready

## ✨ Features

- **15 Analysis Modes** - Comprehensive economic analyses
- **Interactive Visualizations** - Plotly charts with filtering
- **Multiple Forecasting Models** - ARIMA, Prophet, Ensemble, ML
- **Regime Classification** - Growth regime identification
- **Comparative Analysis** - Country and regional comparisons
- **Custom Reports** - Generate tailored analysis reports

## 🚀 Quick Start

### 1) Setup Environment

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2) Run the Application

```powershell
# Main entry point (recommended)
streamlit run app.py

# Or with explicit path
.\.venv\Scripts\streamlit.exe run app.py
```

The dashboard opens at **http://localhost:8501**

## 🔍 The 15 Analysis Pages

| # | Page | Purpose |
|---|------|--------|
| 1 | Overview | Key metrics and summary statistics |
| 2 | Volatility | Growth volatility and risk analysis |
| 3 | Clustering | Country grouping with K-means and PCA |
| 4 | Forecasting | ARIMA and Prophet time series forecasts |
| 5 | Ensemble | Multi-model ensemble forecasting |
| 6 | Anomaly | Outlier detection and crisis identification |
| 7 | Regional | Regional trend analysis and benchmarks |
| 8 | Event Impact | Economic crisis and shock analysis |
| 9 | Momentum | Growth acceleration and momentum indicators |
| 10 | Advanced Forecast | ML models (RF, GBM) with scenarios |
| 11 | Growth Regimes | Regime classification with early warnings |
| 12 | Causal | Econometric analysis (DiD, IV, Granger) |
| 13 | Country Comp | Multi-country comparative analysis |
| 14 | Growth Story | Auto-generated economic narratives |
| 15 | Custom Report | Build exportable custom reports |

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_migration.py

# Quick validation
python tests/validate_refactor.py
```

## 📊 Data

- **Source**: World Bank GDP data
- **Coverage**: 190+ countries
- **Period**: 1960-2024
- **File**: `dataset/real-gdp-growth.csv`

## 📁 Project Structure

```
datadatadata/
├── app.py                          # ⭐ Main entry point
├── explore_gdp.py                  # Info/redirect page
├── explore_gdp_legacy.py           # Original monolith (backup)
├── requirements.txt
├── README.md
│
├── dataset/                        # GDP data
│   ├── real-gdp-growth.csv
│   └── real-gdp-growth.metadata.json
│
├── src/
│   ├── pages/                      # 15 modular pages
│   │   ├── base.py                 # BasePage class
│   │   ├── overview.py
│   │   ├── volatility.py
│   │   ├── clustering.py
│   │   ├── [... 12 more pages]
│   │   └── README.md
│   ├── ui/                         # Reusable components
│   │   ├── theme.py
│   │   ├── charts.py
│   │   └── layout.py
│   ├── config/
│   │   └── constants.py            # Configuration
│   └── [analysis modules]          # Core logic
│
├── tests/                          # Test suite
│   ├── test_migration.py
│   ├── test_phase1.py
│   ├── test_phase2.py
│   └── validate_refactor.py
│
├── scripts/                        # Utility scripts
│   ├── demo_analysis.py
│   ├── export_data.py
│   └── generate_reports.py
│
├── outputs/                        # Generated reports
├── docs/                           # Documentation
└── notebooks/                      # Jupyter notebooks
```

## 🔧 Technology Stack

- **Framework**: Streamlit
- **Data**: Pandas, NumPy
- **Visualization**: Plotly
- **ML**: Scikit-learn, LightGBM, Prophet
- **Stats**: Statsmodels, Scipy

## 🏗️ Architecture

**Design Pattern**: Modular page system
- **BasePage** abstract class
- **PAGE_REGISTRY** for dynamic loading
- **Reusable components** (UI, theme, charts)
- **Centralized configuration**

**Stats:**
- Main file: 1,365 → 35 lines (-97%)
- Files: 1 monolith → 15 modular pages
- Code reuse: 0% → 80%+
- Testability: Difficult → Easy

## 🛠️ Development

### Adding a New Analysis Page

1. Create `src/pages/my_page.py`:
```python
from src.pages.base import BasePage
import streamlit as st

class MyPage(BasePage):
    @property
    def page_name(self):
        return "My Analysis"
    
    @property
    def page_icon(self):
        return "📊"
    
    def render(self):
        st.write("Content here")
```

2. Register in `src/pages/__init__.py`:
```python
PAGE_REGISTRY["my_page"] = MyPage()
```

3. Done! Appears in sidebar automatically.

See [src/pages/README.md](src/pages/README.md) for details.

## 🐛 Troubleshooting

**Streamlit not found:**
```bash
.\.venv\Scripts\streamlit.exe run app.py
```

**Import errors:**
```bash
pip install -r requirements.txt --force-reinstall
```

**Port in use:**
```bash
streamlit run app.py --server.port 8502
```

## 📚 Documentation

- [REFACTOR_COMPLETE.md](REFACTOR_COMPLETE.md) - Complete refactoring report
- [REFACTOR_QUICK_START.md](REFACTOR_QUICK_START.md) - Quick reference
- [MIGRATION_PROGRESS.md](MIGRATION_PROGRESS.md) - Migration tracking
- [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md) - Future plans
- [src/pages/README.md](src/pages/README.md) - Page development guide

## 📄 License

Data: World Bank terms  
Code: [Your License]

---

**Version**: 1.0.0 (Fully Refactored)  
**Last Updated**: December 2024  
**Status**: ✅ Production Ready
