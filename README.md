# SARIMA Forecasting System — Vehicle Sales Prediction

A two-app Streamlit system for training SARIMA time series models and visualising demand forecasts for vehicle sales. Built with role-based access control for multiple user profiles.

---

## Applications

| App | File | Purpose | Roles |
|-----|------|---------|-------|
| Training | `app_entrenamiento.py` | Upload data, train models, approve results | Admin, Analyst |
| Dashboard | `app_dashboard_negocio.py` | Visualise predictions and KPIs | Admin, Analyst, Manager, Viewer |

---

## Project Structure

```
.
├── app_entrenamiento.py           # SARIMA training app
├── app_dashboard_negocio.py       # Business dashboard
├── auth_system.py                 # Authentication & session management
├── utils_validacion.py            # Data validation utilities
├── .streamlit/
│   ├── secrets.toml               # User credentials (NOT committed)
│   └── secrets.toml.example       # Credentials template (committed)
└── data dashboard/                # Model artifacts — auto-managed (NOT committed)
    ├── latest.txt                 # Pointer to the active production run
    ├── training_log.json          # Full history of training runs
    └── YYYYMMDD_HHMMSS/           # One folder per training run
        ├── metricas_mejoradas.json
        ├── prediccion_total_mejorada.xlsx
        ├── grid_search_results.xlsx
        ├── walk_forward_validation.xlsx
        ├── historico_total_mejorado.xlsx
        ├── modelo_total_mejorado.pkl
        ├── acf_plot.png
        └── pacf_plot.png
```

> `data dashboard/` is created and managed automatically by the training app. No manual file handling required.

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

### 1. Configure credentials

Copy the template and fill in your passwords:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with real passwords
```

> **Never commit `.streamlit/secrets.toml` to the repository.**

### 2. Run the apps

Both apps run independently. Start whichever you need:

```bash
# Training app (Admin / Analyst)
streamlit run app_entrenamiento.py

# Business dashboard (all roles)
streamlit run app_dashboard_negocio.py
```

---

## Roles & Permissions

| Role | Training app | Dashboard tabs |
|------|:------------:|----------------|
| `admin` | ✅ | Dashboard, Predictions, ACF/PACF, Grid Search, Walk-Forward, Technical Metrics |
| `analyst` | ✅ | Dashboard, Predictions, ACF/PACF, Grid Search, Walk-Forward, Technical Metrics |
| `manager` | — | Dashboard, Predictions, Recommendations |
| `viewer` | — | Dashboard, Predictions |

---

## Workflow

```
1. [Training]   Upload one or more Excel files with sales data
2. [Training]   Automatic data validation and quality checks
3. [Training]   ADF stationarity test
4. [Training]   Grid search → best SARIMA parameters (AIC + MAPE)
5. [Training]   Walk-forward validation over last N months
6. [Training]   Train final model + generate forecast
7. [Training]   Artifacts saved automatically to data dashboard/<timestamp>/
8. [Training]   Review comparison vs current production model
9. [Training]   Click "Approve" → latest.txt updated → Dashboard live
10. [Dashboard] Loads active model automatically on startup
11. [Dashboard] Sidebar lets users switch between any historical run
```

No ZIP files. No manual file copying. The training app writes directly to the shared `data dashboard/` folder and the dashboard reads from it automatically.

---

## App 1 — Training (`app_entrenamiento.py`)

### Input data format

Excel files (`.xlsx` / `.xls`) with a sheet named `Hoja1`:

| Column | Description |
|--------|-------------|
| `FECHA-VENTA` | Sale date (parseable by pandas) |
| `MARCA` | Vehicle brand |
| `MODELO3` | Vehicle model name |

### Configurable parameters (UI)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Brand filter | `CHERY` | Brand to filter on |
| Model filter | `TIGGO 2` | Model name to filter on |
| Start date | `2021-01-01` | Ignore sales before this date |
| Forecast horizon | `6` months | How many months ahead to predict (3–12) |
| Max sales limit | `100` units/month | Predictions outside `[0, limit]` are discarded |
| Drop current month | `true` | Remove incomplete current month from training |

### SARIMA model details

- **Algorithm**: SARIMAX (Seasonal ARIMA with Exogenous variables)
- **Exogenous variable**: Monthly sales of other models from the same brand
- **Search space**: `p ∈ {1,2,3,5,7}`, `d=1`, `q ∈ {0,1,2}`, seasonal `(P=1, D=1, Q∈{0,1,2}, m=12)`
- **Selection criterion**: Lowest AIC, MAPE < 100%, predictions within configured range
- **Stationarity check**: ADF test run before grid search; result shown with critical values
- **Confidence intervals**: 95% on all forecast points

### Training steps

| Step | Description |
|------|-------------|
| 1 | Data preparation and temporal aggregation |
| 2 | ADF stationarity test |
| 3 | ACF / PACF plots (saved to run folder) |
| 4 | Grid search — reports valid combinations and discarded ones |
| 5 | Walk-forward validation over last N months |
| 6 | Final model training on full dataset |
| 7 | Auto-save all artifacts to `data dashboard/<timestamp>/` |

### Approval flow

After training, the comparison tab shows metrics side-by-side with the current production model (MAPE, AIC, next-month forecast). The analyst clicks **Approve** to activate the new model — this updates `latest.txt` and the dashboard reflects it immediately. If not approved, the run is preserved in history but production is unchanged.

### Residuals diagnostic

The comparison tab includes a two-panel chart (residual time series + distribution) and summary statistics (mean, std, max absolute residual) to verify model fit.

### Training history tab

Every run is logged to `training_log.json` with full metadata. The **Historial** tab shows:
- Summary table with SARIMA order, AIC, MAPE, valid/discarded combinations
- MAPE evolution chart across runs
- CSV export of the full log

---

## App 2 — Business Dashboard (`app_dashboard_negocio.py`)

### Model version selector

The sidebar shows all available training runs ordered by date. The active production run (pointed by `latest.txt`) is marked with 🟢. Historical runs are marked 🔵 and can be selected for comparison without affecting production.

### Tabs by role

| Tab | Admin | Analyst | Manager | Viewer |
|-----|:-----:|:-------:|:-------:|:------:|
| Dashboard (KPIs + historical chart) | ✅ | ✅ | ✅ | ✅ |
| Predictions (N-month forecast + CI) | ✅ | ✅ | ✅ | ✅ |
| Recommendations (purchase strategy) | — | — | ✅ | — |
| ACF/PACF analysis | ✅ | ✅ | — | — |
| Grid search results | ✅ | ✅ | — | — |
| Walk-forward validation | ✅ | ✅ | — | — |
| Full technical metrics | ✅ | ✅ | — | — |

### Manager recommendations

The Manager tab automatically suggests conservative (+10% over IC upper bound) and aggressive (+20%) purchase quantities, and analyses the 3-month trend relative to the historical average to recommend which strategy to use.

---

## Security Notes

- Passwords and credentials are **not** stored in this repository — managed via `.streamlit/secrets.toml`
- Session timeout enforced by `auth_system` (30 minutes by default)
- All file paths use `pathlib.Path(__file__).parent` — no hardcoded absolute paths
- Do not commit `data dashboard/`, `*.pkl`, `*.xlsx`, or `secrets.toml`

---

## .gitignore (key entries)

```
.streamlit/secrets.toml
data dashboard/
*.pkl
*.xlsx
*.xls
*.csv
__pycache__/
*.pyc
.env
```

---

## License

Internal use only. Not for public distribution.
