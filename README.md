# SARIMA Forecasting System — Vehicle Sales Prediction

A two-app Streamlit system for training SARIMA time series models and visualising demand forecasts for vehicle sales. Built with role-based access control for multiple user profiles.

---

## Applications

| App | File | Purpose | Roles |
|-----|------|---------|-------|
| Training | `01. app_entrenamiento.py` | Upload data, train models, export results | Admin, Analyst |
| Dashboard | `02. app_dashboard_negocio.py` | Visualise predictions and KPIs | Admin, Analyst, Manager, Viewer |

---

## Project Structure

```
.
├── 01. app_entrenamiento.py       # SARIMA training app
├── 02. app_dashboard_negocio.py   # Business dashboard
├── auth_system.py                 # Authentication & session management
├── utils_validacion.py            # Data validation utilities
└── data dashboard/                # Pre-loaded model outputs (not committed)
    ├── metricas_mejoradas.json
    ├── prediccion_total_mejorada.xlsx
    ├── grid_search_results.xlsx
    ├── walk_forward_validation.xlsx
    ├── historico_total_mejorado.xlsx
    ├── acf_plot.png
    └── pacf_plot.png
```

> The `data dashboard/` folder is populated by exporting the ZIP package from the Training app and extracting its contents there.

---

## Requirements

```
streamlit
pandas
numpy
statsmodels
scikit-learn
matplotlib
plotly
pillow
openpyxl
```

```bash
pip install -r requirements.txt
```

---

## Getting Started

### 1. Configure authentication

The system uses a custom `auth_system` module. Set up users and roles per that module's documentation.

| Role | Permissions |
|------|------------|
| `admin` | All tabs + `entrenar_modelos`, `exportar`, `ver_acf_pacf`, `ver_grid_search` |
| `analyst` | All technical tabs + `entrenar_modelos`, `exportar`, `ver_acf_pacf`, `ver_grid_search` |
| `manager` | Dashboard, Predictions, Recommendations + `exportar` |
| `viewer` | Dashboard, Predictions only |

> **Never commit credentials or user configuration files to the repository.**

### 2. Configure data paths

`02. app_dashboard_negocio.py` loads files from a local `data dashboard/` subfolder. Update the paths in `load_precargados()` to use relative paths or environment variables before deploying:

```python
# Replace hardcoded absolute paths with:
DATA_DIR = os.environ.get("DASHBOARD_DATA_DIR", "./data dashboard")
```

### 3. Run the apps

```bash
# Training app (Admin / Analyst)
streamlit run "01. app_entrenamiento.py"

# Business dashboard (all roles)
streamlit run "02. app_dashboard_negocio.py"
```

---

## Workflow

```
1. [Training App]   Upload Excel sales data
2. [Training App]   Validate data quality
3. [Training App]   Run grid search → best SARIMA model
4. [Training App]   Walk-forward validation
5. [Training App]   Download ZIP package
6. [Dashboard]      Extract ZIP into data dashboard/
7. [Dashboard]      View predictions, KPIs, and technical metrics
```

---

## App 1 — Training (`01. app_entrenamiento.py`)

### Input data format

Excel files (`.xlsx` / `.xls`) with a sheet named `Hoja1`:

| Column | Description |
|--------|-------------|
| `FECHA-VENTA` | Sale date (parseable by pandas) |
| `MARCA` | Vehicle brand |
| `MODELO3` | Vehicle model name |

### SARIMA model details

- **Algorithm**: SARIMAX (Seasonal ARIMA with Exogenous variables)
- **Exogenous variable**: Monthly sales of other models from the same brand
- **Search space**: `p ∈ {1,2,3,5,7}`, `d=1`, `q ∈ {0,1,2}`, seasonal `(P=1, D=1, Q∈{0,1,2}, m=12)`
- **Selection criterion**: Lowest AIC with MAPE < 100%
- **Training period**: January 2021 onwards (configurable in UI)
- **Forecast horizon**: 6 months ahead with 95% confidence intervals

### Export package

| File | Description |
|------|-------------|
| `modelo_total_mejorado.pkl` | Serialised trained model |
| `prediccion_total_mejorada.xlsx` | 6-month forecast with confidence intervals |
| `grid_search_results.xlsx` | All evaluated parameter combinations |
| `walk_forward_validation.xlsx` | Rolling validation results |
| `historico_total_mejorado.xlsx` | Historical monthly sales series |
| `metricas_mejoradas.json` | Model metadata and performance metrics |
| `acf_plot.png` / `pacf_plot.png` | Autocorrelation diagnostics |

---

## App 2 — Business Dashboard (`02. app_dashboard_negocio.py`)

### Tabs by role

| Tab | Admin | Analyst | Manager | Viewer |
|-----|:-----:|:-------:|:-------:|:------:|
| Dashboard (KPIs + historical chart) | ✅ | ✅ | ✅ | ✅ |
| Predictions (6-month forecast) | ✅ | ✅ | ✅ | ✅ |
| Recommendations (purchase strategy) | — | — | ✅ | — |
| ACF/PACF analysis | ✅ | ✅ | — | — |
| Grid search results | ✅ | ✅ | — | — |
| Walk-forward validation | ✅ | ✅ | — | — |
| Full technical metrics | ✅ | ✅ | — | — |

### Manager recommendations

The Manager tab automatically suggests conservative and aggressive purchase quantities based on the 95% confidence interval upper bound (+10% / +20%), and analyses the 3-month trend relative to the historical average.

---

## Security Notes

- Passwords and credentials are **not** stored in this repository
- Session timeout is enforced by `auth_system`
- Do not commit `.pkl` model files or Excel files containing business sales data
- Do not commit the `data dashboard/` folder — add it to `.gitignore`
- Replace hardcoded local paths in `load_precargados()` before any deployment

---

## .gitignore recommendations

```
data dashboard/
*.pkl
*.xlsx
*.xls
*.csv
metricas_mejoradas.json
__pycache__/
*.pyc
.env
```

---

## License

Internal use only. Not for public distribution.
