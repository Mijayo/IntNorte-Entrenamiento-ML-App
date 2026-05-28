"""
============================================================================
PÁGINA: COMPARATIVA DE MODELOS ML
============================================================================
Compara SARIMAX, Prophet, Regresión Lineal, Random Forest y XGBoost
para predecir el volumen mensual de ventas del Chery Tiggo 2.
Los modelos ML usan lag features + features de calendario.
============================================================================
"""

import io
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')

# ── Constantes de configuración ───────────────────────────────────────────────
RF_N_ESTIMATORS: int = 300
RF_RANDOM_STATE: int = 42
XGB_N_ESTIMATORS: int = 300
XGB_LEARNING_RATE: float = 0.05
XGB_MAX_DEPTH: int = 4
XGB_RANDOM_STATE: int = 42
MIN_OBS_ML_BUFFER: int = 5   # Meses de margen sobre lag_12 + n_test para modelos ML

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

import core.supabase_io as sio
from core.auth_system import (init_session_state, show_login_page, show_user_info,
                              check_session_timeout, has_permission, show_header)

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Comparativa ML", page_icon="🏆", layout="wide")

# CSS global inyectado por show_header(); .winner-box definido en styles.py

# ── Auth ──────────────────────────────────────────────────────────────────────

init_session_state()
if check_session_timeout():
    st.warning("⏱️ Tu sesión ha expirado.")
    st.stop()
if not st.session_state.authenticated:
    show_login_page("🏆 Comparativa de Modelos ML")
    st.stop()
if not has_permission('entrenar_modelos'):
    st.error("❌ No tienes permiso para acceder a esta página")
    st.stop()

show_header("Comparativa de Modelos ML", "¿Qué modelo predice mejor las ventas del Tiggo 2?")
show_user_info()

# ── Paleta de colores por modelo ──────────────────────────────────────────────

COLORES = {
    "SARIMAX":          "#1C7293",
    "Prophet":          "#E84855",
    "Reg. Lineal":      "#2ECC71",
    "Random Forest":    "#F39C12",
    "XGBoost":          "#9B59B6",
}

# ── Feature engineering para modelos ML ──────────────────────────────────────

def crear_features(series: pd.Series) -> pd.DataFrame:
    """
    Transforma la serie mensual en un DataFrame de features para modelos ML.
    Lags: 1, 2, 3, 6, 12 meses.
    Rolling: media y std sobre los últimos 3 y 6 meses (desplazados 1 para no filtrar el futuro).
    Calendario: mes, trimestre.
    Se eliminan las filas con NaN (primeros 12 registros).
    """
    df = pd.DataFrame({"y": series})
    for lag in (1, 2, 3, 6, 12):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["roll_mean_3"] = df["y"].shift(1).rolling(3).mean()
    df["roll_mean_6"] = df["y"].shift(1).rolling(6).mean()
    df["roll_std_3"]  = df["y"].shift(1).rolling(3).std().fillna(0)
    df["mes"]         = df.index.month
    df["trimestre"]   = df.index.quarter
    return df.dropna()


# ── Funciones de métricas y entrenamiento ─────────────────────────────────────

def calc_metrics(real, pred, label):
    real, pred = np.array(real), np.array(pred)
    return {
        "Modelo":    label,
        "MAE":       round(mean_absolute_error(real, pred), 2),
        "RMSE":      round(np.sqrt(mean_squared_error(real, pred)), 2),
        "MAPE (%)":  round(np.mean(np.abs((real - pred) / (real + 0.1))) * 100, 2),
        "R²":        round(r2_score(real, pred), 4),
    }


def entrenar_sarima(train, test_len, order, seasonal_order, exog_train=None, exog_test=None):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    exog=exog_train,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False, maxiter=200, method='lbfgs')
    return np.clip(res.forecast(steps=test_len, exog=exog_test).values, 0, None)


def entrenar_prophet(train, test_len, usar_holidays):
    df_p = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode='multiplicative')
    if usar_holidays:
        m.add_country_holidays(country_name='PE')
    m.fit(df_p)
    future_dates = pd.date_range(
        start=train.index[-1] + pd.offsets.MonthBegin(1),
        periods=test_len, freq='MS'
    )
    forecast = m.predict(pd.DataFrame({"ds": future_dates}))
    return np.clip(forecast["yhat"].values, 0, None)


def entrenar_ml(series, n_test, ModelClass, **kwargs):
    """
    Entrena un modelo ML usando lag features sobre la serie completa.
    El split train/test dentro del DataFrame de features coincide con
    los últimos n_test meses de la serie original (siempre que haya
    al menos lag_12 + n_test + 5 observaciones).
    Devuelve (predicciones, importancias_dict).
    """
    df = crear_features(series)
    feature_cols = [c for c in df.columns if c != "y"]
    n_train = len(df) - n_test
    if n_train < 5:
        raise ValueError(
            f"Solo {n_train} observaciones de entrenamiento tras crear los lags. "
            "Necesitas al menos 17 meses de histórico (12 de lag + 5 de train)."
        )
    X_train, y_train = df[feature_cols].iloc[:n_train], df["y"].iloc[:n_train]
    X_test            = df[feature_cols].iloc[n_train:]

    model = ModelClass(**kwargs)
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, None)

    if hasattr(model, "feature_importances_"):
        imps = dict(zip(feature_cols, model.feature_importances_))
    elif hasattr(model, "coef_"):
        imps = dict(zip(feature_cols, np.abs(model.coef_)))
    else:
        imps = None

    return pred, imps


# ── Gráficas ──────────────────────────────────────────────────────────────────

def plot_predicciones(train, test, predicciones):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.index, y=train.values, name="Histórico (train)",
        line=dict(color="#888888", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=test.index, y=test.values, name="Real (test)",
        line=dict(color="#000000", width=3, dash="dot")
    ))
    for nombre, pred in predicciones.items():
        fig.add_trace(go.Scatter(
            x=test.index, y=pred, name=nombre,
            line=dict(color=COLORES.get(nombre, "#999999"), width=2.5)
        ))
    fig.update_layout(
        title="Predicciones vs Real — período de test",
        xaxis_title="Fecha", yaxis_title="Unidades vendidas",
        template="plotly_white", height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_errores(test, predicciones):
    fig = go.Figure()
    for nombre, pred in predicciones.items():
        fig.add_trace(go.Bar(
            x=test.index, y=np.abs(test.values - pred),
            name=nombre, marker_color=COLORES.get(nombre, "#999999"), opacity=0.82
        ))
    fig.update_layout(
        title="Error absoluto por mes y modelo",
        barmode="group", template="plotly_white", height=360,
        yaxis_title="Error (unidades)", xaxis_title="Fecha"
    )
    return fig


def plot_importancias(importancias):
    fig = go.Figure()
    for nombre, imps in importancias.items():
        sorted_imps = dict(sorted(imps.items(), key=lambda x: x[1], reverse=True))
        fig.add_trace(go.Bar(
            name=nombre,
            x=list(sorted_imps.keys()),
            y=list(sorted_imps.values()),
            marker_color=COLORES.get(nombre, "#999999"), opacity=0.85
        ))
    fig.update_layout(
        title="Importancia de features — modelos ML",
        barmode="group", template="plotly_white", height=360,
        yaxis_title="Importancia", xaxis_title="Feature"
    )
    return fig


# ── UI Principal ──────────────────────────────────────────────────────────────

st.markdown("""
> Compara hasta **5 modelos** sobre el mismo histórico mensual del Tiggo 2.
> SARIMAX y Prophet modelan la serie directamente.
> SARIMAX incorpora además **ventas de otros modelos Chery** como regresor exógeno,
> idéntico al modelo de producción.
> Regresión Lineal, Random Forest y XGBoost usan **lag features** (ventas de meses previos)
> y features de calendario para aprender patrones de compra.
""")

# ── 1. Fuente de datos ────────────────────────────────────────────────────────

st.header("1. Fuente de datos", divider="blue")

fuente = st.radio(
    "Elige cómo cargar el histórico:",
    ["Cargar desde un run guardado en Supabase", "Subir archivo Excel manualmente"],
    horizontal=True
)

ventas_series = None
run_sel = None

if fuente == "Cargar desde un run guardado en Supabase":
    runs = sio.get_available_runs()
    if not runs:
        st.warning("No hay runs guardados. Entrena primero un modelo en la página de Entrenamiento.")
        st.stop()

    run_sel = st.selectbox("Selecciona un run:", runs, format_func=sio.format_run_label)

    if st.button("Cargar histórico", type="primary"):
        with st.spinner("Descargando datos de Supabase..."):
            try:
                metricas_run, _, _, _, hist, exog = sio.load_precargados(run_sel)
                hist = hist.sort_index()
                hist.index = hist.index.to_period('M').to_timestamp()
                st.session_state["ventas_cml"] = hist
                st.session_state["run_metricas_cml"] = metricas_run

                if exog is not None:
                    exog = exog.sort_index()
                    exog.index = exog.index.to_period('M').to_timestamp()
                    st.session_state["exog_cml"] = exog
                    _r = metricas_run.get("variable_exogena", {}).get("pearson_r", "—")
                    st.success(
                        f"✅ Histórico cargado: {len(hist)} meses "
                        f"({hist.index[0].strftime('%b %Y')} → {hist.index[-1].strftime('%b %Y')})"
                        f"  \n✅ **Variable exógena** (`ventas_otros`) disponible "
                        f"— Pearson r = {_r}. SARIMAX usará exactamente los mismos datos que en Entrenamiento."
                    )
                else:
                    st.session_state["exog_cml"] = None
                    st.success(
                        f"✅ Histórico cargado: {len(hist)} meses "
                        f"({hist.index[0].strftime('%b %Y')} → {hist.index[-1].strftime('%b %Y')})"
                    )
                    st.warning(
                        "⚠️ **Variable exógena no disponible** en este run (generado con una versión "
                        "anterior del sistema). SARIMAX correrá sin `ventas_otros`. "
                        "Reentrena el modelo para guardar también la exógena."
                    )
            except Exception as e:
                st.error(f"Error al cargar: {e}")

    if "ventas_cml" in st.session_state:
        ventas_series = st.session_state["ventas_cml"]

else:
    uploaded = st.file_uploader(
        "Excel con columna de fechas y columna de ventas", type=["xlsx", "xls"]
    )
    if uploaded:
        try:
            df_raw = pd.read_excel(uploaded, engine="openpyxl")
            st.dataframe(df_raw.head(), use_container_width=True)
            cols       = df_raw.columns.tolist()
            col_fecha  = st.selectbox(
                "Columna de fecha:", cols,
                help=(
                    "Columna que contiene el mes/año de cada registro. "
                    "Ej: 'Fecha', 'Mes', 'Period'. "
                    "Los valores deben ser fechas reconocibles: '2022-01', 'Ene 2022', '01/2022'."
                )
            )
            col_ventas = st.selectbox(
                "Columna de ventas:", [c for c in cols if c != col_fecha],
                help=(
                    "Columna numérica con las unidades vendidas del Tiggo 2 en cada período. "
                    "Ej: 'Ventas', 'Unidades', 'Total'. "
                    "Si el Excel tiene un registro por venta individual, los valores se sumarán por mes."
                )
            )
            _NONE_EXOG = "— Sin variable exógena —"
            col_exog   = st.selectbox(
                "Variable exógena para SARIMAX (ventas_otros u otra):",
                [_NONE_EXOG] + [c for c in cols if c not in (col_fecha, col_ventas)],
                help="Selecciona ventas_otros u otra variable correlacionada para que SARIMAX la use como regresor exógeno, igual que en Entrenamiento."
            )
            if st.button("Usar estos datos", type="primary"):
                df_raw[col_fecha] = pd.to_datetime(df_raw[col_fecha])
                serie = (
                    df_raw.groupby(col_fecha)[col_ventas]
                    .sum().resample('MS').sum().sort_index()
                )
                st.session_state["ventas_cml"] = serie
                if col_exog != _NONE_EXOG:
                    exog_serie = (
                        df_raw.groupby(col_fecha)[col_exog]
                        .sum().resample('MS').sum().sort_index()
                    )
                    st.session_state["exog_cml"] = exog_serie
                    st.success(f"Serie lista: {len(serie)} meses · exógena: {col_exog}")
                else:
                    st.session_state["exog_cml"] = None
                    st.success(f"Serie lista: {len(serie)} meses")
        except Exception as e:
            st.error(f"Error procesando archivo: {e}")

    if "ventas_cml" in st.session_state:
        ventas_series = st.session_state["ventas_cml"]

if ventas_series is None:
    st.info("Carga o sube el histórico para continuar.")
    st.stop()

# ── 2. Período de análisis ────────────────────────────────────────────────────

st.header("2. Período de análisis", divider="blue")

_all_months  = pd.date_range(ventas_series.index.min(), ventas_series.index.max(), freq='MS')
_month_labels = [d.strftime('%b %Y') for d in _all_months]
_n_total      = len(_all_months)

col_fi, col_ff = st.columns(2)
with col_fi:
    idx_inicio = st.selectbox(
        "Desde:",
        options=list(range(_n_total)),
        format_func=lambda i: _month_labels[i],
        index=0,
        help=f"Primer mes incluido en el análisis. El histórico disponible comienza en {_month_labels[0]}."
    )
with col_ff:
    idx_fin = st.selectbox(
        "Hasta:",
        options=list(range(_n_total)),
        format_func=lambda i: _month_labels[i],
        index=_n_total - 1,
        help=f"Último mes incluido en el análisis. El histórico disponible termina en {_month_labels[-1]}."
    )

if idx_inicio >= idx_fin:
    st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
    st.stop()

_fecha_ini = _all_months[idx_inicio]
_fecha_fin = _all_months[idx_fin]
ventas_series = ventas_series[
    (ventas_series.index >= _fecha_ini) & (ventas_series.index <= _fecha_fin)
]
st.caption(
    f"Período activo: **{_fecha_ini.strftime('%b %Y')} → {_fecha_fin.strftime('%b %Y')}** "
    f"· {len(ventas_series)} meses"
)

# ── 3. Configuración ──────────────────────────────────────────────────────────

st.header("3. Configuración", divider="blue")

col_cfg1, col_cfg2 = st.columns(2)

with col_cfg1:
    st.subheader("Partición train / test")
    n_test = st.slider("Meses para test (hold-out):", min_value=3, max_value=12, value=6)
    st.caption(f"Train: {len(ventas_series) - n_test} meses | Test: {n_test} meses")

    st.subheader("Modelos a comparar")
    usar_sarima  = st.checkbox("SARIMAX",                   value=True)
    usar_prophet = st.checkbox("Prophet",                   value=True)
    usar_lr      = st.checkbox("Regresión Lineal",          value=True)
    usar_rf      = st.checkbox("Random Forest",             value=True)
    usar_xgb     = st.checkbox(
        "XGBoost",
        value=XGBOOST_OK,
        disabled=not XGBOOST_OK,
        help="Instala xgboost (pip install xgboost) para activar este modelo." if not XGBOOST_OK else ""
    )

with col_cfg2:
    if usar_sarima:
        st.subheader("Parámetros SARIMAX")

        # Autocompletar desde el run cargado si está disponible
        _run_met = st.session_state.get("run_metricas_cml")
        if _run_met and "mejor_modelo" in _run_met:
            _ord = _run_met["mejor_modelo"].get("order", [1, 1, 0])
            _sea = _run_met["mejor_modelo"].get("seasonal_order", [1, 0, 2, 12])
            st.caption(
                f"⚡ Parámetros autocompletados desde el run cargado: "
                f"({_ord[0]},{_ord[1]},{_ord[2]})({_sea[0]},{_sea[1]},{_sea[2]})[12]"
            )
        else:
            _ord = [1, 1, 0]
            _sea = [1, 0, 2, 12]

        cs1, cs2 = st.columns(2)
        with cs1:
            p = st.number_input("p (AR)",  min_value=0, max_value=5, value=int(_ord[0]))
            d = st.number_input("d (I)",   min_value=0, max_value=2, value=int(_ord[1]))
            q = st.number_input("q (MA)",  min_value=0, max_value=5, value=int(_ord[2]))
        with cs2:
            P = st.number_input("P (SAR)", min_value=0, max_value=3, value=int(_sea[0]))
            D = st.number_input("D (SI)",  min_value=0, max_value=2, value=int(_sea[1]))
            Q = st.number_input("Q (SMA)", min_value=0, max_value=3, value=int(_sea[2]))
        st.caption("Período estacional fijo: 12 meses")

    usar_holidays = False
    if usar_prophet:
        st.subheader("Opciones Prophet")
        usar_holidays = st.checkbox("Festivos de Perú (PE)", value=True)

# Validación mínima de datos para modelos ML
min_obs_ml = 12 + n_test + MIN_OBS_ML_BUFFER   # lag_12 + test + buffer mínimo de train
hay_suficientes = len(ventas_series) >= min_obs_ml
if (usar_lr or usar_rf or (usar_xgb and XGBOOST_OK)) and not hay_suficientes:
    st.warning(
        f"Los modelos ML necesitan al menos {min_obs_ml} meses de histórico "
        f"(tienes {len(ventas_series)}). Reduce el test o añade más datos."
    )

# ── 3. Ejecutar ───────────────────────────────────────────────────────────────

st.header("4. Ejecutar comparación", divider="blue")

if not any([usar_sarima, usar_prophet, usar_lr, usar_rf, usar_xgb and XGBOOST_OK]):
    st.warning("Selecciona al menos un modelo.")
    st.stop()

if st.button("🏆 Comparar modelos", type="primary", use_container_width=True):

    train = ventas_series.iloc[:-n_test]
    test  = ventas_series.iloc[-n_test:]

    n_activos   = sum([usar_sarima, usar_prophet, usar_lr, usar_rf,
                       usar_xgb and XGBOOST_OK])
    resultados  = []
    predicciones = {}
    importancias = {}
    errores      = {}

    progress = st.progress(0)
    status   = st.empty()
    paso     = [0]

    def avanzar(nombre):
        paso[0] += 1
        progress.progress(paso[0] / n_activos)
        status.text(f"✔ {nombre} completado")

    # ── SARIMAX ───────────────────────────────────────────────────────────────
    if usar_sarima:
        status.text("Entrenando SARIMAX...")
        t0 = time.time()
        try:
            exog_s = st.session_state.get("exog_cml")
            exog_tr = exog_te = None
            if exog_s is not None:
                exog_al = exog_s.reindex(ventas_series.index).ffill().fillna(0)
                exog_tr = exog_al.iloc[:-n_test].values.reshape(-1, 1)
                exog_te = exog_al.iloc[-n_test:].values.reshape(-1, 1)
            pred = entrenar_sarima(train, n_test, (p, d, q), (P, D, Q, 12), exog_tr, exog_te)
            met  = calc_metrics(test.values, pred, "SARIMAX")
            met["Tiempo (s)"] = round(time.time() - t0, 2)
            resultados.append(met)
            predicciones["SARIMAX"] = pred
        except Exception as e:
            errores["SARIMAX"] = str(e)
        avanzar("SARIMAX")

    # ── Prophet ───────────────────────────────────────────────────────────────
    if usar_prophet:
        status.text("Entrenando Prophet...")
        t0 = time.time()
        try:
            pred = entrenar_prophet(train, n_test, usar_holidays)
            met  = calc_metrics(test.values, pred, "Prophet")
            met["Tiempo (s)"] = round(time.time() - t0, 2)
            resultados.append(met)
            predicciones["Prophet"] = pred
        except Exception as e:
            errores["Prophet"] = str(e)
        avanzar("Prophet")

    # ── Regresión Lineal ──────────────────────────────────────────────────────
    if usar_lr:
        status.text("Entrenando Regresión Lineal...")
        t0 = time.time()
        try:
            pred, imps = entrenar_ml(ventas_series, n_test, LinearRegression)
            met = calc_metrics(test.values, pred, "Reg. Lineal")
            met["Tiempo (s)"] = round(time.time() - t0, 2)
            resultados.append(met)
            predicciones["Reg. Lineal"] = pred
            if imps:
                importancias["Reg. Lineal"] = imps
        except Exception as e:
            errores["Reg. Lineal"] = str(e)
        avanzar("Regresión Lineal")

    # ── Random Forest ─────────────────────────────────────────────────────────
    if usar_rf:
        status.text("Entrenando Random Forest...")
        t0 = time.time()
        try:
            pred, imps = entrenar_ml(
                ventas_series, n_test, RandomForestRegressor,
                n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, n_jobs=-1
            )
            met = calc_metrics(test.values, pred, "Random Forest")
            met["Tiempo (s)"] = round(time.time() - t0, 2)
            resultados.append(met)
            predicciones["Random Forest"] = pred
            if imps:
                importancias["Random Forest"] = imps
        except Exception as e:
            errores["Random Forest"] = str(e)
        avanzar("Random Forest")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if usar_xgb and XGBOOST_OK:
        status.text("Entrenando XGBoost...")
        t0 = time.time()
        try:
            pred, imps = entrenar_ml(
                ventas_series, n_test, XGBRegressor,
                n_estimators=XGB_N_ESTIMATORS, learning_rate=XGB_LEARNING_RATE,
                max_depth=XGB_MAX_DEPTH, random_state=XGB_RANDOM_STATE, verbosity=0
            )
            met = calc_metrics(test.values, pred, "XGBoost")
            met["Tiempo (s)"] = round(time.time() - t0, 2)
            resultados.append(met)
            predicciones["XGBoost"] = pred
            if imps:
                importancias["XGBoost"] = imps
        except Exception as e:
            errores["XGBoost"] = str(e)
        avanzar("XGBoost")

    progress.empty()
    status.empty()

    # Mostrar errores individuales
    for nombre, err in errores.items():
        st.warning(f"**{nombre}** no pudo entrenarse: {err}")

    if not resultados:
        st.error("Ningún modelo completó el entrenamiento. Revisa los datos.")
        st.stop()

    # ── 4. Resultados ─────────────────────────────────────────────────────────

    st.header("5. Resultados", divider="blue")

    df_met = pd.DataFrame(resultados).set_index("Modelo")

    # Tabla con celdas resaltadas
    st.dataframe(
        df_met.style
            .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], axis=0, color="#d4edda")
            .highlight_max(subset=["R²"],                       axis=0, color="#d4edda")
            .highlight_min(subset=["Tiempo (s)"],               axis=0, color="#fff3cd"),
        use_container_width=True
    )
    st.caption(
        "\\* MAPE calculado con denominador `(real + 0.1)` para evitar división por cero "
        "en meses con ventas nulas — idéntico al criterio del modelo de producción."
    )

    # Ganador
    mejor = df_met["MAPE (%)"].idxmin()
    mape_mejor = df_met.loc[mejor, "MAPE (%)"]
    ranking = df_met["MAPE (%)"].sort_values()
    diff_pp = round(ranking.iloc[1] - ranking.iloc[0], 1) if len(ranking) > 1 else 0
    segundo = ranking.index[1] if len(ranking) > 1 else ""

    st.markdown(
        f'<div class="winner-box">🏆 <b>Mejor modelo (MAPE):</b> {mejor} — '
        f'{mape_mejor:.1f}%'
        f'{f" · {diff_pp}pp mejor que {segundo}" if segundo else ""}</div>',
        unsafe_allow_html=True
    )

    # Métricas resumen en columnas
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    for col, key, minimize in zip(
        [col_m1, col_m2, col_m3, col_m4],
        ["MAE", "RMSE", "MAPE (%)", "R²"],
        [True,   True,   True,      False]
    ):
        with col:
            best_val = df_met[key].min() if minimize else df_met[key].max()
            best_mod = df_met[key].idxmin() if minimize else df_met[key].idxmax()
            st.metric(label=f"{key} — mejor", value=str(best_val), delta=best_mod,
                      delta_color="off")

    st.divider()

    # Gráfica predicciones
    st.plotly_chart(plot_predicciones(train, test, predicciones), use_container_width=True)

    # Gráfica errores
    st.plotly_chart(plot_errores(test, predicciones), use_container_width=True)

    # Feature importances (solo modelos ML)
    if importancias:
        st.subheader("Importancia de features — modelos ML")
        st.caption(
            "Random Forest / XGBoost: importancia por reducción de impureza (Gini). "
            "Regresión Lineal: valor absoluto del coeficiente."
        )
        st.plotly_chart(plot_importancias(importancias), use_container_width=True)

    # Tabla detallada
    st.subheader("Tabla detallada — período de test")
    df_tabla = pd.DataFrame({
        "Fecha": test.index.strftime("%b %Y"),
        "Real":  test.values.astype(int),
    })
    for nombre, pred in predicciones.items():
        df_tabla[nombre]            = np.round(pred).astype(int)
        df_tabla[f"Err {nombre}"]   = np.abs(test.values - pred).round(1)
    st.dataframe(df_tabla.set_index("Fecha"), use_container_width=True)

    # Descarga CSV
    buf = io.StringIO()
    df_tabla.to_csv(buf)
    st.download_button(
        "⬇ Descargar tabla como CSV",
        data=buf.getvalue().encode("utf-8"),
        file_name="comparativa_modelos_tiggo2.csv",
        mime="text/csv"
    )

    # Interpretación
    mape_sarimax = df_met.loc["SARIMAX",       "MAPE (%)"] if "SARIMAX"       in df_met.index else "—"
    mape_prophet = df_met.loc["Prophet",       "MAPE (%)"] if "Prophet"       in df_met.index else "—"
    mape_lr      = df_met.loc["Reg. Lineal",   "MAPE (%)"] if "Reg. Lineal"   in df_met.index else "—"
    mape_rf      = df_met.loc["Random Forest", "MAPE (%)"] if "Random Forest" in df_met.index else "—"
    mape_xgb     = df_met.loc["XGBoost",       "MAPE (%)"] if "XGBoost"       in df_met.index else "—"

    _exog_usada = st.session_state.get("exog_cml") is not None
    _exog_nota  = ("con variable exógena `ventas_otros`" if _exog_usada
                   else "sin variable exógena (run antiguo — reentrena para activarla)")

    with st.expander("📚 ¿Cómo interpretar estos resultados?"):
        st.markdown(f"""
        ### MAPE (Mean Absolute Percentage Error)
        El MAPE indica cuánto se equivoca cada modelo **en promedio**, expresado
        como porcentaje sobre el valor real de ventas mensual del Tiggo 2.

        | Modelo | MAPE obtenido | Interpretación |
        |--------|--------------|----------------|
        | SARIMAX | {mape_sarimax}% | En promedio, el modelo se equivoca {mape_sarimax}% respecto al valor real |
        | Prophet | {mape_prophet}% | En promedio, se equivoca {mape_prophet}% |
        | Reg. Lineal | {mape_lr}% | En promedio, se equivoca {mape_lr}% |
        | Random Forest | {mape_rf}% | En promedio, se equivoca {mape_rf}% |
        | XGBoost | {mape_xgb}% | En promedio, se equivoca {mape_xgb}% |

        **Referencia práctica:** MAPE < 10% es excelente · < 20% es aceptable en ventas automotrices.

        ---

        ### Todas las métricas
        | Métrica | Significado | Objetivo |
        |---------|-------------|---------|
        | **MAPE (%)** | Error % promedio sobre el valor real | Menor |
        | **MAE** | Error promedio en unidades vendidas | Menor |
        | **RMSE** | Como MAE pero penaliza errores grandes | Menor |
        | **R²** | Proporción de varianza de ventas explicada (1.0 = perfecto) | Mayor |

        > SARIMAX evaluado {_exog_nota}.

        ---

        ### ¿Por qué Prophet puede ganar a SARIMAX?
        1. **Estacionalidad multiplicativa** — si las ventas del Tiggo 2 suben en ciertos
           meses, Prophet escala el efecto proporcionalmente a la tendencia.
           SARIMAX usa diferenciación, que asume efectos aditivos.
        2. **Festivos de Perú** — Prophet añade automáticamente el impacto de días como
           Navidad, Año Nuevo, Fiestas Patrias y Semana Santa sobre el comportamiento de compra.
        3. **Changepoints** — si hubo un cambio de producto, crisis (ej. COVID) o
           ajuste de precio, Prophet detecta y adapta la tendencia. SARIMAX no lo hace
           de forma automática.

        ### ¿Cuándo SARIMAX puede ganar?
        - Series cortas (< 36 meses) donde Prophet no tiene suficiente historia.
        - Cuando la variable exógena `ventas_otros` tiene alta correlación (|r| ≥ 0.3)
          con las ventas del Tiggo 2 — en ese caso SARIMAX captura dinámicas del mercado
          que Prophet no puede ver.
        - Series muy estables sin quiebres de tendencia visibles.

        ### ¿Por qué los modelos ML (LR, RF, XGBoost) pueden ganar?
        - Capturan patrones **no lineales** entre meses anteriores y el mes a predecir.
        - El **lag de 12 meses** les permite aprender directamente la estacionalidad anual
          del Tiggo 2 sin necesidad de especificarla como en SARIMAX.
        - XGBoost y Random Forest son robustos a valores atípicos en el histórico.

        ### ¿Cuándo los modelos ML pueden fallar?
        - Con series cortas (< {min_obs_ml} meses), los lags consumen demasiadas
          observaciones y el train queda con muy pocos ejemplos.
        - Si la serie tiene una tendencia fuerte y creciente, los lags no capturan
          bien la extrapolación hacia el futuro (a diferencia de SARIMAX/Prophet).

        ### ¿Cuál usar en producción?
        El modelo con **menor MAPE** ({mejor} en este run) es la mejor opción para
        planificación de inventario mensual. Si dos modelos tienen MAPE similar (< 2pp
        de diferencia), prioriza el más **rápido** o el más **interpretable** para
        presentar a la dirección.
        """)

    # Guardar en session_state
    st.session_state["cml_resultados"] = {
        "metricas":    df_met,
        "predicciones": predicciones,
        "test":        test,
        "train":       train,
        "ganador":     mejor,
        "fuente_run":  run_sel if fuente == "Cargar desde un run guardado en Supabase" else None,
    }

# ── 5. Publicar en producción ─────────────────────────────────────────────────

st.header("6. Publicar modelo ganador en producción", divider="green")

if "cml_resultados" not in st.session_state:
    st.info("Ejecuta primero la comparación para ver las opciones de publicación.")
else:
    res      = st.session_state["cml_resultados"]
    ganador  = res["ganador"]
    mape_gan = res["metricas"].loc[ganador, "MAPE (%)"]
    fuente_run = res.get("fuente_run")

    st.markdown(
        f"**Modelo ganador:** {ganador} — MAPE {mape_gan:.1f}%"
    )

    if mape_gan > 20:
        st.warning(
            f"⚠️ El modelo ganador ({ganador}) tiene MAPE > 20%. "
            "Considera reentrenar con más datos antes de publicar."
        )

    # Mostrar parámetros recomendados si SARIMAX es el ganador o está entre los evaluados
    if "SARIMAX" in res["metricas"].index:
        mape_sarimax_cml = res["metricas"].loc["SARIMAX", "MAPE (%)"]
        sarimax_order_str = (
            f"`order=({p},{d},{q})` `seasonal_order=({P},{D},{Q},12)`"
            if usar_sarima else "parámetros definidos en la configuración de esta sesión"
        )
        _exog_pub = st.session_state.get("exog_cml") is not None
        with st.expander("⚙️ Configuración SARIMAX evaluada en esta comparativa"):
            st.info(
                f"SARIMAX MAPE: **{mape_sarimax_cml:.1f}%**  \n"
                f"Parámetros usados: {sarimax_order_str}  \n"
                f"Variable exógena: **{'✅ ventas_otros incluida' if _exog_pub else '⚠️ no disponible (run antiguo)'}**  \n\n"
                "Si SARIMAX es el modelo que quieres publicar, el run ya está guardado en Supabase "
                "con todos los artefactos. Actívalo directamente con el botón de abajo."
            )

    # Si la fuente es un run de Supabase, ofrecer activarlo
    if fuente_run:
        st.markdown("---")
        st.subheader("🚀 Activar run de Supabase como producción")

        available_runs = sio.get_available_runs()
        current_latest = sio.get_default_run(available_runs)
        already_active = current_latest == fuente_run

        if already_active:
            st.success(f"✅ El run `{sio.format_run_label(fuente_run)}` ya es el modelo activo en el Dashboard.")
        else:
            if current_latest:
                st.info(
                    f"Dashboard usa actualmente: **{sio.format_run_label(current_latest)}**  \n"
                    f"Activar: **{sio.format_run_label(fuente_run)}** (fuente de esta comparativa)"
                )
            if st.button(
                f"✅ Activar '{sio.format_run_label(fuente_run)}' en el Dashboard",
                type="primary", use_container_width=True
            ):
                sio.approve_model(fuente_run)
                st.success(
                    f"✅ Run `{fuente_run}` activado como modelo de producción. "
                    "El Dashboard ya lo refleja."
                )
    else:
        st.info(
            "ℹ️ Para publicar en producción desde aquí, carga los datos "
            "desde un **run guardado en Supabase** (opción 1 de la fuente de datos). "
            "Si subiste un Excel manual, entrena primero el modelo en la página "
            "**Entrenamiento** y después actívalo desde allí o desde esta sección."
        )
