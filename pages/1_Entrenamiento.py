"""
============================================================================
PÁGINA: ENTRENAMIENTO DE MODELOS SARIMA
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
from datetime import datetime, date, timedelta
import io
import optuna
import warnings

# Suprimir solo las advertencias de convergencia propias de statsmodels/scipy;
# las de NumPy y el resto del runtime permanecen visibles.
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')

from core.logger import get_logger
log = get_logger("entrenamiento")

# ── Constantes de configuración del modelo ────────────────────────────────────
OPTUNA_N_TRIALS: int = 80        # Trials bayesianos TPE (≈ 4× más rápido que el grid exhaustivo de 384)
SARIMA_SEASONAL_PERIOD: int = 12 # Período estacional mensual
WALK_FORWARD_MONTHS: int = 12    # Ventana máxima de validación walk-forward
EXOG_ROLLING_WINDOW: int = 6     # Meses para proyectar ventas_otros en el horizonte

# ── Ventana de entrenamiento por defecto ──────────────────────────────────────
# SARIMAX aprende los patrones del período que le muestras. Si el mercado ha
# cambiado de régimen (ej.: nuevo nivel de demanda post-2024), entrenar con
# datos históricos lejanos ancla el modelo en ese nivel antiguo y produce
# predicciones sistemáticamente bajas.
#
# Regla práctica:
#   · Usa al menos 3 ciclos estacionales completos (36 meses) para que SARIMA
#     aprenda los coeficientes estacionales con fiabilidad estadística.
#   · Excluye períodos que no representen el comportamiento actual del mercado
#     (pandemia 2020, quiebres estructurales, lanzamientos de producto).
#
# Para TIGGO 2 (caso 2026): el nivel de demanda pasó de ~31 uds/mes (2021-2024)
# a ~65 uds/mes (2026). Arrancar en 2024-01-01 da 27+ meses recientes y evita
# que los datos pre-boom arrastren la predicción hacia abajo.
TRAINING_DEFAULT_START: date = date(2024, 1, 1)

import core.supabase_io as sio
from core.auth_system import (init_session_state, show_login_page, show_user_info,
                              check_session_timeout, has_permission, show_header)
from core.utils_validacion import (validate_dataframe, show_validation_results,
                                   preview_data, plot_temporal_distribution,
                                   plot_missing_data)

# ── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Entrenamiento SARIMA", page_icon="🤖", layout="wide")

# CSS global inyectado por show_header() — estilos extra de progreso
st.markdown("""
<style>
.stProgress > div > div > div > div { background-color: #20C997 !important; }
</style>
""", unsafe_allow_html=True)

# ── Auth ─────────────────────────────────────────────────────────────────────

init_session_state()
if check_session_timeout():
    st.warning("⏱️ Tu sesión ha expirado.")
    st.stop()
if not st.session_state.authenticated:
    show_login_page("🤖 Entrenamiento de Modelos")
    st.stop()
if not has_permission('entrenar_modelos'):
    st.error("❌ No tienes permiso para acceder a esta aplicación")
    st.stop()

show_header("Entrenamiento de Modelos SARIMA", "Sistema de Entrenamiento Automatizado")
show_user_info()

# ── Funciones de modelo (sin I/O) ────────────────────────────────────────────

def run_adf_test(series: pd.Series) -> dict:
    """
    Test de Dickey-Fuller Aumentado para estacionariedad.

    Returns
    -------
    dict
        statistic, p_value, lags_used, critical_1/5/10pct, is_stationary
        (is_stationary=True cuando p_value < 0.05).
    """
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'statistic': round(result[0], 4), 'p_value': round(result[1], 4),
        'lags_used': result[2], 'critical_1pct': round(result[4]['1%'], 4),
        'critical_5pct': round(result[4]['5%'], 4),
        'critical_10pct': round(result[4]['10%'], 4),
        'is_stationary': bool(result[1] < 0.05)
    }


def train_sarima_model(ventas: pd.Series, exog_data: pd.DataFrame,
                       order: tuple, seasonal_order: tuple):
    """
    Entrena un modelo SARIMAX con variable exógena.

    Parameters
    ----------
    ventas : pd.Series
        Serie temporal mensual de ventas.
    exog_data : pd.DataFrame
        Variable exógena (ventas_otros) alineada con ventas.
    order : tuple
        (p, d, q) del componente ARIMA.
    seasonal_order : tuple
        (P, D, Q, m) del componente estacional.

    Returns
    -------
    SARIMAXResultsWrapper
        Modelo ajustado listo para forecast.
    """
    model = SARIMAX(ventas, exog=exog_data, order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False, maxiter=200, method='lbfgs')


def perform_optuna_search(train: pd.Series, test: pd.Series,
                          train_exog: pd.DataFrame, test_exog: pd.DataFrame,
                          progress_bar, status_text,
                          max_ventas: int,
                          n_trials: int = OPTUNA_N_TRIALS) -> tuple:
    """
    Búsqueda inteligente de hiperparámetros SARIMA usando Optuna (TPE Bayesiano).

    Espacio de búsqueda: p∈{0-3}, d∈{0-1}, q∈{0-3}, P∈{0-1}, D∈{0-1}, Q∈{0-2}.
    Se descarta automáticamente d=1∧D=1 (sobre-diferenciación en series cortas).
    Criterio de optimización: MAPE mínimo sobre el conjunto de test.

    Parameters
    ----------
    train, test : pd.Series
        Conjuntos de entrenamiento y test de la serie temporal.
    train_exog, test_exog : pd.DataFrame
        Variable exógena (ventas_otros) para cada split.
    progress_bar, status_text : Streamlit widgets
        Para actualizar la barra de progreso y el texto de estado.
    max_ventas : int
        Límite superior de predicciones válidas (unidades/mes).
    n_trials : int
        Número de trials bayesianos (por defecto: OPTUNA_N_TRIALS).

    Returns
    -------
    tuple
        (best_params, best_aic, best_mape, trial_results, n_discarded)
        best_params es None si ningún trial fue válido.
    """
    trial_results = []
    best_state = {'mape': np.inf, 'aic': np.inf, 'params': None}
    failures = [0]

    def objective(trial):
        p = trial.suggest_int("p", 0, 3)
        d = trial.suggest_int("d", 0, 1)
        q = trial.suggest_int("q", 0, 3)
        P = trial.suggest_int("P", 0, 1)
        D = trial.suggest_int("D", 0, 1)
        Q = trial.suggest_int("Q", 0, 2)
        # Evitar doble diferenciación: d=1 y D=1 simultáneos sobrediferencian
        # la serie y producen modelos inestables con series cortas (<80 obs).
        if d == 1 and D == 1:
            failures[0] += 1
            return np.inf
        try:
            model = SARIMAX(train, exog=train_exog, order=(p, d, q),
                            seasonal_order=(P, D, Q, SARIMA_SEASONAL_PERIOD),
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False, maxiter=200, method='lbfgs')
            predictions = results.forecast(steps=len(test), exog=test_exog)
            if predictions.min() < 0 or predictions.max() > max_ventas:
                failures[0] += 1
                return np.inf  # penaliza predicciones fuera del rango lógico
            mape = np.mean(np.abs((test - predictions) / (test + 0.1))) * 100
            trial_results.append({
                'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q, 'm': SARIMA_SEASONAL_PERIOD,
                'mae': mean_absolute_error(test, predictions),
                'rmse': np.sqrt(mean_squared_error(test, predictions)),
                'mape': mape, 'aic': results.aic, 'bic': results.bic
            })
            if mape < best_state['mape']:
                best_state['mape'] = mape
                best_state['aic'] = results.aic
                best_state['params'] = ((p, d, q), (P, D, Q, SARIMA_SEASONAL_PERIOD))
            return mape
        except Exception:
            failures[0] += 1
            return np.inf

    def progress_callback(study, trial):
        pct = (trial.number + 1) / n_trials
        progress_bar.progress(min(0.25 + pct * 0.35, 0.60))
        n_valid = len(trial_results)
        n_disc = failures[0]
        mejor = f"MAPE {best_state['mape']:.2f}%" if best_state['params'] else "buscando..."
        status_text.text(
            f"🔍 Optuna trial {trial.number + 1}/{n_trials} · "
            f"Evaluados: {n_valid + n_disc} · Válidos: {n_valid} · "
            f"Descartados: {n_disc} · Mejor: {mejor}"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])

    return best_state['params'], best_state['aic'], best_state['mape'], trial_results, failures[0]


def perform_walk_forward(ventas: pd.Series, exog_data: pd.DataFrame,
                         best_params: tuple, n_months: int,
                         max_ventas: int) -> list[dict]:
    """
    Validación walk-forward (rolling origin) sobre los últimos n_months meses.

    Por cada mes del período de validación, reentrena el modelo con todos los
    datos anteriores y predice un paso adelante. Solo se registran predicciones
    en el rango [0, max_ventas].

    Returns
    -------
    list[dict]
        Lista de registros con fecha, real, prediccion, error y error_pct.
    """
    results = []
    for i in range(len(ventas) - n_months, len(ventas)):
        try:
            _exog_i = exog_data[:i] if exog_data is not None else None
            model_wf = SARIMAX(ventas[:i], exog=_exog_i,
                               order=best_params[0], seasonal_order=best_params[1],
                               enforce_stationarity=False, enforce_invertibility=False)
            res_wf = model_wf.fit(disp=False, maxiter=200, method='lbfgs')
            pred = res_wf.forecast(steps=1, exog=exog_data[i:i+1] if exog_data is not None else None)
            if 0 <= pred.iloc[0] <= max_ventas:
                real = ventas.iloc[i]
                results.append({
                    'fecha': ventas.index[i], 'real': real,
                    'prediccion': pred.iloc[0],
                    'error': abs(real - pred.iloc[0]),
                    'error_pct': abs(real - pred.iloc[0]) / (real + 0.1) * 100
                })
        except Exception:
            continue
    return results


def plot_residuals(model_results):
    residuals = model_results.resid
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Residuos en el tiempo", "Distribución"),
                        column_widths=[0.65, 0.35])
    fig.add_trace(go.Scatter(x=residuals.index, y=residuals.values,
                             mode='lines', line=dict(color='#1C7293', width=1.5)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_trace(go.Histogram(x=residuals.values, nbinsx=15,
                               marker_color='#1C7293', opacity=0.75), row=1, col=2)
    fig.update_layout(title="Diagnóstico de Residuos", height=350,
                      template='plotly_white', showlegend=False)
    return fig


# ── Tabs ─────────────────────────────────────────────────────────────────────

tabs = st.tabs(["📤 Cargar Datos", "✅ Validación", "🎓 Preparar Datos",
                "🤖 Entrenamiento", "📊 Comparación", "📋 Historial"])

# ── Tab 1: Cargar Datos ───────────────────────────────────────────────────────

with tabs[0]:
    st.header("📤 Carga de Datos", divider='blue')
    st.markdown("""
    **Instrucciones:**
    1. Carga el archivo Excel con el histórico de ventas
    2. El sistema limpiará y validará los datos automáticamente
    3. También puedes cargar varios archivos si el histórico está dividido — se unificarán en uno solo
    """)

    uploaded_files = st.file_uploader(
        "Selecciona el archivo Excel", type=['xlsx', 'xls'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} archivo{'s' if len(uploaded_files) > 1 else ''} cargado{'s' if len(uploaded_files) > 1 else ''}")
        for f in uploaded_files:
            st.markdown(f"- **{f.name}** ({f.size / 1024:.1f} KB)")

        if st.button("🔄 Procesar", type="primary"):
            with st.spinner("Procesando..."):
                dfs_ventas = []
                df_stock_cargado = None
                for f in uploaded_files:
                    try:
                        sheets = pd.ExcelFile(f, engine='openpyxl').sheet_names
                        if 'Hoja1' in sheets:
                            df = pd.read_excel(f, sheet_name='Hoja1', engine='openpyxl')
                            dfs_ventas.append(df)
                            st.success(f"✅ {f.name} → ventas ({len(df):,} filas)")
                        elif 'Stock Actual' in sheets:
                            df_stock_cargado = pd.read_excel(f, sheet_name='Stock Actual', engine='openpyxl')
                            st.success(f"✅ {f.name} → stock ({len(df_stock_cargado):,} filas)")
                        else:
                            st.error(f"❌ {f.name}: hojas encontradas {sheets} — se esperaba 'Hoja1' o 'Stock Actual'")
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")
                if dfs_ventas:
                    df_unified = pd.concat(dfs_ventas, ignore_index=True)
                    n_bruto = len(df_unified)

                    # ── Limpieza automática ───────────────────────────────────
                    # 1. Eliminar filas con MODELO3 nulo
                    n_nulos = int(df_unified['MODELO3'].isna().sum()) if 'MODELO3' in df_unified.columns else 0
                    if n_nulos > 0:
                        df_unified = df_unified.dropna(subset=['MODELO3'])

                    # 2. Eliminar duplicados por CHASIS (conservar registro más reciente)
                    n_antes_dedup = len(df_unified)
                    if 'CHASIS' in df_unified.columns and 'FECHA-VENTA' in df_unified.columns:
                        df_unified['FECHA-VENTA'] = pd.to_datetime(df_unified['FECHA-VENTA'], errors='coerce')
                        df_unified = (df_unified
                                      .sort_values('FECHA-VENTA')
                                      .drop_duplicates(subset=['CHASIS'], keep='last')
                                      .reset_index(drop=True))
                    n_dupl = n_antes_dedup - len(df_unified)
                    # ─────────────────────────────────────────────────────────

                    st.session_state['df_raw'] = df_unified
                    st.success(f"✅ {n_bruto:,} registros brutos → **{len(df_unified):,} limpios**")
                    if n_dupl > 0 or n_nulos > 0:
                        st.warning(f"🧹 Limpieza: {n_dupl} duplicados por CHASIS eliminados · "
                                   f"{n_nulos} filas sin MODELO3 eliminadas")
                if df_stock_cargado is not None:
                    st.session_state['df_stock'] = df_stock_cargado
                if dfs_ventas or df_stock_cargado is not None:
                    st.rerun()

# ── Tab 2: Validación ─────────────────────────────────────────────────────────

with tabs[1]:
    st.header("✅ Validación de Datos", divider='green')

    if 'df_raw' not in st.session_state:
        st.info("👈 Primero carga los datos en la pestaña **Cargar Datos**")
    else:
        df_raw = st.session_state['df_raw']
        is_valid, results, warnings_val, errors = validate_dataframe(df_raw, "Datos Unificados")
        show_validation_results(results, warnings_val, errors)
        st.markdown("---")
        if st.checkbox("👁️ Ver Preview", value=True):
            preview_data(df_raw)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Distribución Temporal")
            plot_temporal_distribution(df_raw)
        with col2:
            st.subheader("❌ Datos Faltantes")
            plot_missing_data(df_raw)
        st.session_state['validation_passed'] = is_valid
        st.session_state['df_validated'] = df_raw

# ── Tab 3: Preparar Datos (académico) ─────────────────────────────────────────

with tabs[2]:
    st.header("🎓 ¿Cómo se construye el Excel de entrenamiento?", divider='violet')

    if 'df_validated' not in st.session_state:
        st.info("👈 Primero carga y valida los datos en las pestañas anteriores.")
    else:
        df_ac = st.session_state['df_validated'].copy()
        df_ac['FECHA-VENTA'] = pd.to_datetime(df_ac['FECHA-VENTA'], errors='coerce')

        st.markdown(
            "Esta pestaña muestra paso a paso cómo se transforma el histórico de ventas "
            "en la **serie temporal mensual** que alimenta el modelo SARIMA. "
            "Configura los filtros y observa cómo evoluciona el dataset en cada etapa."
        )

        col1, col2 = st.columns(2)
        with col1:
            ac_marca  = st.text_input("Marca",  value="CHERY",   key="ac_marca")
            ac_modelo = st.text_input("Modelo", value="TIGGO 2", key="ac_modelo")
        with col2:
            ac_fecha_ini = st.date_input("Fecha inicio", value=TRAINING_DEFAULT_START, key="ac_fi")
            ac_fecha_fin = st.date_input("Fecha fin",    value=date.today(),      key="ac_ff")

        # Cálculo del límite superior exclusivo
        if ac_fecha_fin.month == 12:
            ac_fecha_fin_excl = pd.Timestamp(ac_fecha_fin.year + 1, 1, 1)
        else:
            ac_fecha_fin_excl = pd.Timestamp(ac_fecha_fin.year, ac_fecha_fin.month + 1, 1)

        st.markdown("---")

        # ── Paso 1: datos brutos ──────────────────────────────────────────────
        with st.expander("📋 Paso 1 — Datos brutos (muestra)", expanded=True):
            st.markdown(
                "El Excel de entrada contiene **una fila por venta individual**. "
                "Cada registro tiene al menos: fecha de venta, marca y modelo del vehículo."
            )
            st.dataframe(df_ac.head(10), use_container_width=True, hide_index=True)
            st.caption(f"Total en memoria: **{len(df_ac):,} filas** · columnas: {df_ac.columns.tolist()}")

        # ── Paso 2: filtrar marca ─────────────────────────────────────────────
        df_p2 = df_ac[df_ac['MARCA'] == ac_marca]
        with st.expander(f"🏷️ Paso 2 — Filtrar por marca: «{ac_marca}»"):
            st.markdown(
                f"Se descartan todas las filas que no pertenezcan a la marca **{ac_marca}**. "
                "Esto reduce el conjunto a un fabricante concreto."
            )
            st.dataframe(df_p2.head(10), use_container_width=True, hide_index=True)
            pct = len(df_p2) / len(df_ac) * 100 if len(df_ac) > 0 else 0
            st.caption(f"Resultado: **{len(df_p2):,} filas** ({pct:.1f}% del total)")

        # ── Paso 3: filtrar modelo ────────────────────────────────────────────
        df_p3 = df_p2[df_p2['MODELO3'] == ac_modelo]
        with st.expander(f"🚗 Paso 3 — Filtrar por modelo: «{ac_modelo}»"):
            st.markdown(
                f"Del subconjunto anterior se conservan solo las ventas de **{ac_modelo}**. "
                "Esta será la serie objetivo (y) que el modelo aprenderá a predecir."
            )
            st.dataframe(df_p3.head(10), use_container_width=True, hide_index=True)
            st.caption(f"Resultado: **{len(df_p3):,} filas**")

        # ── Paso 4: filtrar rango de fechas ────────────────────────────────────
        df_p4 = df_p3[
            (df_p3['FECHA-VENTA'] >= pd.Timestamp(ac_fecha_ini)) &
            (df_p3['FECHA-VENTA'] <  ac_fecha_fin_excl)
        ]
        with st.expander(f"📅 Paso 4 — Filtrar rango de fechas: {ac_fecha_ini} → {ac_fecha_fin}"):
            st.markdown(
                "Se recorta el histórico al rango seleccionado. Esto define la "
                "**ventana de entrenamiento**: el período cuyos patrones de nivel, "
                "tendencia y estacionalidad aprenderá el modelo.\n\n"
                "**Principio clave:** incluir únicamente los meses que representen "
                "el comportamiento *actual* del mercado. Datos de un régimen de demanda "
                "anterior (pre-pandemia, pre-lanzamiento, pre-cambio de precio) "
                "sesgan el modelo hacia ese nivel histórico y producen predicciones "
                "sistemáticamente alejadas de la realidad presente."
            )
            st.dataframe(df_p4.head(10), use_container_width=True, hide_index=True)
            if len(df_p4) > 0:
                st.caption(
                    f"Resultado: **{len(df_p4):,} filas** · "
                    f"{df_p4['FECHA-VENTA'].min().strftime('%Y-%m')} → "
                    f"{df_p4['FECHA-VENTA'].max().strftime('%Y-%m')}"
                )
            else:
                st.warning("⚠️ Sin datos para los filtros seleccionados.")

        if len(df_p4) > 0:
            # ── Paso 5: resample mensual ──────────────────────────────────────
            ventas_ac = df_p4.set_index('FECHA-VENTA').resample('ME').size().rename('ventas')
            df_mensual = ventas_ac.reset_index().rename(columns={'FECHA-VENTA': 'Mes', 'ventas': 'Ventas'})

            with st.expander("📊 Paso 5 — Agregar por mes (resample 'ME')", expanded=True):
                st.markdown(
                    "Se **cuenta el número de ventas por mes** (`resample('ME').size()`). "
                    "Esto convierte las filas individuales en una serie temporal de frecuencia mensual, "
                    "que es el formato que requiere SARIMA."
                )
                col_t, col_g = st.columns([1, 2])
                with col_t:
                    st.dataframe(df_mensual, use_container_width=True, hide_index=True)
                with col_g:
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=df_mensual['Mes'], y=df_mensual['Ventas'],
                        mode='lines+markers',
                        line=dict(color='#1C7293', width=2),
                        marker=dict(size=6),
                        name='Ventas'
                    ))
                    fig_ts.update_layout(
                        title=f"Serie temporal mensual — {ac_modelo}",
                        xaxis_title="Mes", yaxis_title="Unidades vendidas",
                        template='plotly_white', height=300
                    )
                    st.plotly_chart(fig_ts, use_container_width=True, config={'displayModeBar': False})
                st.caption(
                    f"**{len(df_mensual)} meses** · min={df_mensual['Ventas'].min()} · "
                    f"max={df_mensual['Ventas'].max()} · media={df_mensual['Ventas'].mean():.1f}"
                )

            # ── Paso 6: variable exógena ──────────────────────────────────────
            df_otros_ac = df_p2[
                (df_p2['MODELO3'] != ac_modelo) &
                (df_p2['FECHA-VENTA'] >= pd.Timestamp(ac_fecha_ini)) &
                (df_p2['FECHA-VENTA'] <  ac_fecha_fin_excl)
            ]
            ventas_otros_ac = (df_otros_ac.set_index('FECHA-VENTA')
                               .resample('ME').size().rename('ventas_otros'))
            exog_ac = (pd.DataFrame({'ventas_otros': ventas_otros_ac})
                       .reindex(ventas_ac.index, fill_value=0))

            with st.expander("📐 Paso 6 — Variable exógena (ventas otros modelos de la marca)"):
                st.markdown(
                    "SARIMA admite una variable explicativa externa (**exog**). "
                    "Se usa la **suma mensual de ventas de todos los demás modelos de la misma marca** "
                    "como proxy de la dinámica general del mercado. "
                    "Cuando la marca vende más en conjunto, probablemente también sube la demanda del modelo objetivo."
                )
                df_exog_show = pd.DataFrame({
                    'Mes': ventas_ac.index.strftime('%Y-%m'),
                    f'{ac_modelo} (y)': ventas_ac.values,
                    'Otros modelos (exog)': exog_ac['ventas_otros'].values
                })
                st.dataframe(df_exog_show, use_container_width=True, hide_index=True)

            # ── Resultado final ───────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📥 DataFrame final que recibe SARIMA")
            st.markdown(
                "- **Índice**: `DatetimeIndex` con frecuencia mensual (`ME`)  \n"
                "- **`ventas`**: variable objetivo **(y)** — lo que el modelo predice  \n"
                "- **`ventas_otros`**: variable exógena **(X)** — ayuda al modelo a captar tendencias del mercado"
            )

            df_final_ac = pd.DataFrame({
                'Mes': ventas_ac.index.strftime('%Y-%m'),
                'ventas (y)': ventas_ac.values,
                'ventas_otros (exog)': exog_ac['ventas_otros'].values
            })
            st.dataframe(df_final_ac, use_container_width=True, hide_index=True)

            # Descarga Excel
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                df_final_ac.to_excel(writer, index=False, sheet_name='Serie_SARIMA')
                df_mensual.to_excel(writer, index=False, sheet_name='Ventas_Mensuales')
                df_exog_show.to_excel(writer, index=False, sheet_name='Comparativa')
            nombre_xlsx = (
                f"datos_sarima_{ac_modelo.replace(' ','_')}"
                f"_{ac_fecha_ini}_{ac_fecha_fin}.xlsx"
            )
            st.download_button(
                "📥 Descargar Excel de entrenamiento",
                data=buf.getvalue(),
                file_name=nombre_xlsx,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

# ── Tab 4: Entrenamiento ──────────────────────────────────────────────────────

with tabs[3]:
    st.header("🤖 Entrenamiento del Modelo", divider='orange')

    if 'df_validated' not in st.session_state:
        st.info("👈 Primero valida los datos en la pestaña **Validación**")
    elif not st.session_state.get('validation_passed', False):
        st.error("❌ Los datos no pasaron la validación.")
    else:
        df = st.session_state['df_validated']

        st.subheader("⚙️ Configuración")
        col1, col2 = st.columns(2)
        with col1:
            modelo_filtro = st.text_input("Filtro Modelo (MODELO3)", value="TIGGO 2")
            marca_filtro = st.text_input("Filtro Marca", value="CHERY")
        with col2:
            fecha_inicio = st.date_input(
                "Fecha de inicio del entrenamiento",
                value=TRAINING_DEFAULT_START,
                help=(
                    "**Ventana de entrenamiento** — el modelo aprende exclusivamente "
                    "del período comprendido entre esta fecha y la fecha fin.\n\n"
                    "**¿Por qué importa?** SARIMA ajusta sus coeficientes al nivel "
                    "medio y la estacionalidad del período que ve. Si el mercado cambió "
                    "de régimen (nuevo precio, nueva competencia, post-pandemia), incluir "
                    "datos del régimen anterior introduce sesgo sistemático y el modelo "
                    "subestima o sobreestima de forma persistente.\n\n"
                    "**Regla práctica:** usa al menos 36 meses (3 ciclos estacionales) "
                    "del período que mejor represente el comportamiento actual. "
                    f"Valor por defecto: {TRAINING_DEFAULT_START.strftime('%d/%m/%Y')}."
                )
            )
            fecha_fin = st.date_input(
                "Fecha fin de datos",
                value=date.today(),
                help=(
                    "Límite superior del histórico usado para entrenar. "
                    "Si 'Eliminar mes actual' también está marcado, "
                    "se aplica el corte más conservador de los dos."
                )
            )

        col3, col4, col5 = st.columns(3)
        with col3:
            horizonte = st.slider("Horizonte (meses)", 3, 12, 6)
        with col4:
            max_ventas = st.number_input("Límite máximo ventas/mes",
                                         min_value=10, max_value=10000, value=100, step=10)
        with col5:
            eliminar_mes_actual = st.checkbox("Eliminar mes actual", value=True)

        # ── Documentación funcional: ventana de entrenamiento ─────────────────
        n_meses_ventana = (
            (fecha_fin.year - fecha_inicio.year) * 12
            + (fecha_fin.month - fecha_inicio.month)
        )
        with st.expander("📅 ¿Cómo elegir la ventana de entrenamiento?", expanded=False):
            st.markdown(f"""
**Ventana seleccionada:** `{fecha_inicio.strftime('%d/%m/%Y')}` → `{fecha_fin.strftime('%d/%m/%Y')}`
({n_meses_ventana} meses aprox.)

---

#### Concepto — Quiebre estructural (*structural break*)

Un quiebre estructural ocurre cuando las condiciones de mercado cambian de forma
permanente y la serie adopta un **nuevo nivel base, tendencia o patrón estacional**.
Incluir datos anteriores al quiebre introduce un sesgo sistemático porque el modelo
intenta reconciliar dos comportamientos incompatibles.

**Indicadores de quiebre estructural:**
- El modelo predice consistentemente **por debajo** de los valores reales durante
  varios meses seguidos (sesgo negativo persistente).
- La media de los últimos 12 meses es **más del 30% superior** a la media de los
  3 años anteriores.
- Ocurrió un evento conocido: cambio de precio, nuevo modelo, apertura de concesionario,
  crisis o recuperación de demanda.

#### Guía de configuración

| Situación | Fecha inicio recomendada |
|-----------|--------------------------|
| Mercado estable, sin cambios de tendencia | Máximo histórico disponible (2017+) |
| Recuperación post-pandemia capturada | 2021-01-01 |
| Nuevo nivel de demanda desde 2024 | **2024-01-01** ← caso actual TIGGO 2 |
| Lanzamiento de versión nueva del modelo | Fecha del lanzamiento |

#### Mínimo estadístico

SARIMA necesita al menos **36 meses** (3 ciclos estacionales completos) para
estimar los coeficientes estacionales con fiabilidad. Con menos datos, los
intervalos de confianza son muy amplios y el modelo tiende a sobreajustarse.
            """)
            if n_meses_ventana < 36:
                st.warning(
                    f"⚠️ La ventana actual tiene ~{n_meses_ventana} meses — por debajo del "
                    "mínimo recomendado de 36. Considera ampliar la fecha de inicio."
                )
            elif n_meses_ventana < 48:
                st.info(
                    f"ℹ️ Ventana de ~{n_meses_ventana} meses — funcional, aunque con "
                    "48+ meses los coeficientes estacionales serán más robustos."
                )
            else:
                st.success(f"✅ Ventana de ~{n_meses_ventana} meses — tamaño adecuado.")

        st.markdown("---")

        if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Paso 0: Validación anticipada de configuración
                # Necesitamos preparar la serie primero para poder validar max_ventas.
                # Esta comprobación rápida usa los datos ya validados en session_state.
                _df_check = st.session_state['df_validated']
                _df_check['FECHA-VENTA'] = pd.to_datetime(_df_check['FECHA-VENTA'], errors='coerce')
                _ventas_check = (
                    _df_check[
                        (_df_check['MARCA'] == marca_filtro) &
                        (_df_check['MODELO3'] == modelo_filtro)
                    ]
                    .set_index('FECHA-VENTA')
                    .resample('ME')
                    .size()
                )
                if len(_ventas_check) > 0:
                    pico_historico = int(_ventas_check.max())
                    if max_ventas < pico_historico:
                        st.error(
                            f"❌ El límite máximo de ventas configurado ({max_ventas} uds/mes) "
                            f"es menor que el pico histórico del modelo ({pico_historico} uds/mes). "
                            "Todos los trials de Optuna serán rechazados. "
                            f"Aumenta el límite a al menos **{pico_historico + 10}** antes de entrenar."
                        )
                        st.stop()

                # Paso 1: Preparar datos
                status_text.text("📊 Preparando datos...")
                progress_bar.progress(0.05)
                df['FECHA-VENTA'] = pd.to_datetime(df['FECHA-VENTA'], errors='coerce')
                df_chery = df[df['MARCA'] == marca_filtro].copy()
                df_modelo = df_chery[df_chery['MODELO3'] == modelo_filtro].copy()

                # Calcular límite superior exclusivo a partir de fecha_fin
                if fecha_fin.month == 12:
                    fecha_fin_excl = datetime(fecha_fin.year + 1, 1, 1)
                else:
                    fecha_fin_excl = datetime(fecha_fin.year, fecha_fin.month + 1, 1)

                if eliminar_mes_actual:
                    fecha_mes_actual = datetime.now().replace(
                        day=1, hour=0, minute=0, second=0, microsecond=0)
                    fecha_limite = min(fecha_fin_excl, fecha_mes_actual)
                else:
                    fecha_limite = fecha_fin_excl

                df_modelo = df_modelo[df_modelo['FECHA-VENTA'] < fecha_limite]
                st.info(f"✅ Datos hasta: {(fecha_limite - timedelta(days=1)).strftime('%Y-%m-%d')}")

                fecha_inicio_str = fecha_inicio.strftime('%Y-%m-%d')
                df_modelo = df_modelo[df_modelo['FECHA-VENTA'] >= fecha_inicio_str]
                ventas_modelo = df_modelo.set_index('FECHA-VENTA').resample('ME').size()

                st.success(f"✅ {len(ventas_modelo)} meses · {len(df_modelo):,} ventas")

                df_otros = df_chery[df_chery['MODELO3'] != modelo_filtro].copy()
                df_otros = df_otros[df_otros['FECHA-VENTA'] < fecha_limite]
                df_otros = df_otros[df_otros['FECHA-VENTA'] >= fecha_inicio_str]
                ventas_otros = df_otros.set_index('FECHA-VENTA').resample('ME').size()
                exog_data = pd.DataFrame({'ventas_otros': ventas_otros}) \
                              .reindex(ventas_modelo.index, fill_value=0)

                # ── Filtro de correlación: descartar exógeno si es ruido ────────
                _r_exog = float(np.corrcoef(ventas_modelo.values,
                                            exog_data['ventas_otros'].values)[0, 1])
                if abs(_r_exog) >= 0.3:
                    st.info(
                        f"ℹ️ Variable exógena incluida — Pearson r = {_r_exog:.2f} "
                        f"(|r| ≥ 0.3 · señal suficiente)"
                    )
                else:
                    st.warning(
                        f"⚠️ Variable exógena descartada — Pearson r = {_r_exog:.2f} "
                        f"(|r| < 0.3 · introduce ruido). Se entrena SARIMA puro."
                    )
                    exog_data = None

                progress_bar.progress(0.10)

                # Paso 2: ADF
                status_text.text("📐 Test de estacionariedad ADF...")
                adf = run_adf_test(ventas_modelo)
                with st.expander("📐 Resultado Test ADF", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Estadístico ADF", f"{adf['statistic']:.4f}")
                    c2.metric("p-valor", f"{adf['p_value']:.4f}")
                    c3.metric("¿Estacionaria?", "✅ Sí" if adf['is_stationary'] else "⚠️ No")
                    st.caption(f"Valores críticos — 1%: {adf['critical_1pct']} · "
                               f"5%: {adf['critical_5pct']} · 10%: {adf['critical_10pct']}")
                    if not adf['is_stationary']:
                        st.info("No estacionaria: el parámetro d=1 aplica diferenciación.")
                progress_bar.progress(0.15)

                # Paso 3: ACF/PACF
                status_text.text("📊 Generando ACF/PACF...")
                fig_acf, ax_acf = plt.subplots(figsize=(12, 4))
                plot_acf(ventas_modelo, lags=24, ax=ax_acf)
                ax_acf.set_title('ACF - Autocorrelación')
                plt.tight_layout()

                fig_pacf, ax_pacf = plt.subplots(figsize=(12, 4))
                plot_pacf(ventas_modelo, lags=24, ax=ax_pacf, method='ywm')
                ax_pacf.set_title('PACF - Autocorrelación Parcial')
                plt.tight_layout()
                progress_bar.progress(0.25)

                # Paso 4: Búsqueda con Optuna (TPE Bayesiano)
                status_text.text("🔍 Iniciando búsqueda Optuna...")
                train_size = len(ventas_modelo) - horizonte
                train = ventas_modelo[:train_size]
                test = ventas_modelo[train_size:]
                train_exog = exog_data[:train_size] if exog_data is not None else None
                test_exog = exog_data[train_size:] if exog_data is not None else None

                best_params, best_aic, best_mape, grid_results, failures = perform_optuna_search(
                    train, test, train_exog, test_exog, progress_bar, status_text, max_ventas
                )

                if best_params is None:
                    st.error(f"❌ Ningún trial produjo predicciones en [0, {max_ventas}].")
                    st.info("Aumenta el límite máximo de ventas en la configuración.")
                    st.stop()

                df_grid = pd.DataFrame(grid_results).sort_values('mape')

                # ── Resultado Optuna con explicación clara ──────────────────
                n_valid = len(grid_results)
                n_total = n_valid + failures
                st.success(
                    f"✅ Mejor modelo encontrado: **SARIMA{best_params[0]}{best_params[1]}** · "
                    f"MAPE: {best_mape:.2f}% · AIC: {best_aic:.2f}"
                )
                with st.expander("📊 Detalle de la búsqueda Optuna — ¿qué son válidos y descartados?"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Trials evaluados", n_total,
                              help="Combinaciones de (p,d,q)(P,D,Q) que Optuna probó")
                    c2.metric("✅ Válidos", n_valid,
                              help="El modelo ajustó correctamente Y sus predicciones están dentro del rango [0, max_ventas]")
                    c3.metric("❌ Descartados", failures,
                              help="Trials rechazados por: predicciones negativas, superiores al límite máximo, o errores numéricos al ajustar")
                    st.markdown("""
**¿Por qué se descarta un trial?**
- **Predicciones negativas** — el modelo predice ventas < 0, lo cual no tiene sentido físico.
- **Predicciones fuera del límite** — superan el `max_ventas` configurado (posible sobreajuste).
- **Error numérico** — la combinación (p,d,q)(P,D,Q) no converge con los datos disponibles.

**¿Por qué Optuna es mejor que Grid Search?**
Con Grid Search se evalúan **384 combinaciones fijas**. Optuna usa **TPE (Tree-structured Parzen Estimator)**: aprende qué zonas del espacio dan buenos resultados y enfoca la búsqueda ahí, logrando igual o mejor calidad con solo **80 trials** (~4× más rápido).
""")
                progress_bar.progress(0.60)

                # Paso 5: Walk-Forward
                # Se validan hasta 12 meses para una estimación robusta del MAPE
                # (mínimo: el horizonte configurado; requiere al menos 2 meses de entrenamiento)
                status_text.text(f"🔄 Walk-forward validation ({WALK_FORWARD_MONTHS} meses)...")
                n_wf = min(WALK_FORWARD_MONTHS, len(ventas_modelo) - 2)
                n_wf = max(n_wf, horizonte)
                wf_results = perform_walk_forward(ventas_modelo, exog_data, best_params,
                                                   n_wf, max_ventas)
                df_wf = pd.DataFrame(wf_results)

                if df_wf.empty:
                    st.error("❌ Walk-forward sin resultados válidos. Revisa el límite máximo.")
                    st.stop()

                mape_wf = df_wf['error_pct'].mean()
                st.success(f"✅ MAPE walk-forward: {mape_wf:.2f}% · {len(df_wf)}/{n_wf} meses validados")

                # ── Diagnóstico automático cuando MAPE > 15% ─────────────────
                if mape_wf > 15:
                    with st.expander("⚠️ MAPE elevado: causas probables y acciones recomendadas",
                                     expanded=True):
                        st.warning(
                            f"El MAPE walk-forward es **{mape_wf:.1f}%** (objetivo: ≤ 15%). "
                            "Esto indica que, en promedio, las predicciones se desvían más de un "
                            "15% del valor real. Antes de activar este modelo en producción, "
                            "revisa las siguientes causas:"
                        )
                        st.markdown("""
**Causas más frecuentes y su solución:**

| # | Causa probable | Señal de alerta | Solución |
|---|---------------|-----------------|----------|
| 1 | **Serie demasiado corta** — SARIMA necesita al menos 3 ciclos estacionales completos | < 48 meses en el histórico | Ampliar el rango de fechas de entrenamiento |
| 2 | **Alta variabilidad intrínseca** — meses con ventas cero o picos extremos distorsionan el modelo | `max/mean > 4` en la serie mensual | Aplicar suavizado o considerar Prophet, más robusto ante quiebres |
| 3 | **Mes actual incluido** — datos incompletos inflan el error del último mes | Opción "Eliminar mes actual" desactivada | Activar "Eliminar mes actual" en la configuración |
| 4 | **Variable exógena poco informativa** — si `ventas_otros` tiene baja correlación con `ventas_tiggo2`, puede añadir ruido | Correlación < 0.3 entre las dos series | Entrenar sin variable exógena (SARIMA puro) y comparar MAPE |
| 5 | **Quiebre estructural** — el mercado adoptó un nuevo nivel base que el modelo no vio (ej.: demanda TIGGO 2 pasó de ~31 uds/mes en 2021-2024 a ~65 uds/mes en 2026, produciendo MAPE = 42%) | Sesgo negativo persistente varios meses seguidos; media últimos 12m > 30% superior a la media histórica | Cambiar «Fecha de inicio» a la fecha del quiebre (ej.: 2024-01-01) y reentrenar — ver expander «¿Cómo elegir la ventana de entrenamiento?» |
""")
                        st.info(
                            "**Próximos pasos recomendados:**  \n"
                            "1. Ve a la pestaña **Comparativa ML** y compara con Prophet y Random Forest.  \n"
                            "2. Si Prophet obtiene MAPE < 15%, considera usarlo como modelo de producción.  \n"
                            "3. Si ningún modelo logra MAPE < 15%, documenta la limitación de los datos "
                            "antes de presentar el sistema."
                        )
                elif mape_wf > 10:
                    st.warning(
                        f"⚠️ MAPE walk-forward: **{mape_wf:.1f}%** — aceptable pero mejorable. "
                        "Objetivo: <15%. Considera comparar con Prophet en la pestaña **Comparativa ML**."
                    )

                progress_bar.progress(0.80)

                # Paso 6: Modelo final
                status_text.text("🤖 Entrenando modelo final...")
                model_final = train_sarima_model(ventas_modelo, exog_data,
                                                  best_params[0], best_params[1])

                if exog_data is not None:
                    _n_trend = min(EXOG_ROLLING_WINDOW * 2, len(exog_data))
                    _recent = exog_data['ventas_otros'].values[-_n_trend:]
                    _slope, _intercept = np.polyfit(np.arange(_n_trend), _recent, 1)
                    _future_x = np.arange(_n_trend, _n_trend + horizonte)
                    _exog_proj = np.clip(_intercept + _slope * _future_x, 0, None).round(0)
                    exog_future = pd.DataFrame({'ventas_otros': _exog_proj})
                    _dir = "↗ creciente" if _slope > 1 else ("↘ decreciente" if _slope < -1 else "→ estable")
                    st.info(
                        f"ℹ️ **Proyección exógena (tendencia lineal):** últimos {_n_trend} meses · "
                        f"pendiente {_slope:+.1f} uds/mes ({_dir}) · "
                        f"rango proyectado: {int(_exog_proj.min())}–{int(_exog_proj.max())} uds/mes."
                    )
                else:
                    exog_future = None
                forecast = model_final.forecast(steps=horizonte, exog=exog_future)
                conf_int = model_final.get_forecast(steps=horizonte, exog=exog_future).conf_int()
                fechas_futuras = pd.date_range(
                    start=ventas_modelo.index[-1], periods=horizonte + 1, freq='ME'
                )[1:]

                predicciones = pd.DataFrame({
                    'Fecha': fechas_futuras,
                    'Mes': fechas_futuras.strftime('%B %Y'),
                    'Predicción': forecast.values.round(1),
                    'IC_Inferior': conf_int.iloc[:, 0].values.round(1),
                    'IC_Superior': conf_int.iloc[:, 1].values.round(1)
                })

                metricas = {
                    'fecha_entrenamiento': datetime.now().strftime('%Y%m%d_%H%M%S'),
                    'usuario': st.session_state.username,
                    'configuracion': {
                        'modelo_filtro': modelo_filtro, 'marca_filtro': marca_filtro,
                        'fecha_inicio': fecha_inicio_str,
                        'fecha_fin': fecha_fin.strftime('%Y-%m-%d'),
                        'horizonte': horizonte,
                        'max_ventas': int(max_ventas),
                        # Meses incluidos en el entrenamiento: permite auditar
                        # qué ventana temporal aprendió el modelo y reproducirlo.
                        'meses_ventana': n_meses_ventana
                    },
                    'datos_limpios': {
                        'total_ventas': len(df_modelo), 'meses_datos': len(ventas_modelo),
                        'periodo': (f"{ventas_modelo.index.min().strftime('%Y-%m')} a "
                                    f"{ventas_modelo.index.max().strftime('%Y-%m')}")
                    },
                    'mejor_modelo': {
                        'order': list(best_params[0]), 'seasonal_order': list(best_params[1]),
                        'aic': float(best_aic), 'bic': float(model_final.bic),
                        'combinaciones_validas': len(grid_results),
                        'combinaciones_descartadas': failures
                    },
                    'adf_test': adf,
                    'variable_exogena': {'usada': exog_data is not None, 'pearson_r': round(_r_exog, 3)},
                    'walk_forward_validation': {'mape': float(mape_wf), 'meses_evaluados': len(df_wf)},
                    'predicciones_futuras': {'proximo_mes': float(predicciones['Predicción'].iloc[0])}
                }

                # Guardar en Supabase
                run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
                with st.spinner("Guardando en Supabase..."):
                    sio.save_to_dashboard(run_name, model_final, predicciones, df_grid,
                                          df_wf, ventas_modelo, metricas, fig_acf, fig_pacf)

                progress_bar.progress(1.0)
                status_text.text("✅ ¡Completado!")

                # Guardar en session state
                st.session_state.update({
                    'new_model': model_final, 'new_predictions': predicciones,
                    'new_grid': df_grid, 'new_walkforward': df_wf,
                    'new_historico': ventas_modelo, 'new_metrics': metricas,
                    'new_acf_fig': fig_acf, 'new_pacf_fig': fig_pacf,
                    'current_run_name': run_name, 'training_complete': True
                })

                sio.save_training_log({
                    'timestamp': datetime.now().isoformat(), 'run_name': run_name,
                    'usuario': st.session_state.username,
                    'modelo': modelo_filtro, 'marca': marca_filtro,
                    'fecha_inicio': fecha_inicio_str, 'horizonte': horizonte,
                    'max_ventas': int(max_ventas),
                    'sarima_order': list(best_params[0]),
                    'sarima_seasonal': list(best_params[1]),
                    'aic': float(best_aic), 'mape_wf': float(mape_wf),
                    'meses_datos': len(ventas_modelo),
                    'combinaciones_validas': len(grid_results),
                    'combinaciones_descartadas': failures
                })

                st.success(f"✅ Guardado en Supabase como `{run_name}`. "
                           "Ve a **Comparación** para activarlo.")

            except Exception as e:
                log.exception("Error inesperado durante el entrenamiento")
                st.error(
                    "❌ Se produjo un error durante el entrenamiento. "
                    "Comprueba la configuración (marca, modelo, fechas y límite de ventas) "
                    "y vuelve a intentarlo. Si el error persiste, revisa los logs de la aplicación."
                )
                with st.expander("Detalle del error (para soporte técnico)"):
                    st.code(str(e))

# ── Tab 5: Comparación ────────────────────────────────────────────────────────

with tabs[4]:
    st.header("📊 Comparación: Nuevo vs Actual", divider='green')

    if 'training_complete' not in st.session_state:
        st.info("👈 Primero entrena un modelo.")
    else:
        current_model = sio.load_current_model()
        new_metrics = st.session_state['new_metrics']

        if current_model:
            st.subheader("📊 Tabla Comparativa")
            mape_delta = (new_metrics['walk_forward_validation']['mape']
                          - current_model['walk_forward_validation']['mape'])
            aic_delta = (new_metrics['mejor_modelo']['aic']
                         - current_model['mejor_modelo']['aic'])
            pred_delta = (new_metrics['predicciones_futuras']['proximo_mes']
                          - current_model['predicciones_futuras']['proximo_mes'])

            actual_orden = f"{current_model['mejor_modelo']['order']}{current_model['mejor_modelo']['seasonal_order']}"
            nuevo_orden = f"{new_metrics['mejor_modelo']['order']}{new_metrics['mejor_modelo']['seasonal_order']}"

            df_comp = pd.DataFrame({
                'Métrica': ['MAPE Walk-Forward', 'AIC', 'Predicción Próximo Mes', 'Modelo'],
                'Actual': [f"{current_model['walk_forward_validation']['mape']:.2f}%",
                           f"{current_model['mejor_modelo']['aic']:.2f}",
                           f"{current_model['predicciones_futuras']['proximo_mes']:.0f}",
                           actual_orden],
                'Nuevo':  [f"{new_metrics['walk_forward_validation']['mape']:.2f}%",
                           f"{new_metrics['mejor_modelo']['aic']:.2f}",
                           f"{new_metrics['predicciones_futuras']['proximo_mes']:.0f}",
                           nuevo_orden],
                'Δ': [f"{mape_delta:+.2f}%", f"{aic_delta:+.2f}",
                      f"{pred_delta:+.0f}",
                      "Cambió" if actual_orden != nuevo_orden else "="]
            })
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            st.subheader("💡 Recomendación")
            if mape_delta < 0 and aic_delta < 0:
                st.markdown('<div class="success-box">✅ <strong>APROBAR</strong> — Mejora MAPE y AIC.</div>',
                            unsafe_allow_html=True)
            elif mape_delta < 0 or aic_delta < 0:
                st.markdown('<div class="warning-box">⚠️ <strong>REVISAR</strong> — Mejora en algunas métricas.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="comparison-worse">❌ <strong>NO APROBAR</strong> — Empeora las métricas.</div>',
                            unsafe_allow_html=True)
        else:
            st.info("No hay modelo previo. Este será el primero en producción.")

        # KPIs nuevo modelo
        st.subheader("📋 Métricas del Nuevo Modelo")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAPE", f"{new_metrics['walk_forward_validation']['mape']:.2f}%")
        col2.metric("AIC", f"{new_metrics['mejor_modelo']['aic']:.2f}")
        col3.metric("Próximo Mes", f"{new_metrics['predicciones_futuras']['proximo_mes']:.0f}")
        col4.metric("Meses Datos", new_metrics['datos_limpios']['meses_datos'])

        st.subheader("🔮 Predicciones")
        st.dataframe(st.session_state['new_predictions'], use_container_width=True, hide_index=True)

        # Residuos
        st.subheader("🔬 Diagnóstico de Residuos")
        fig_res = plot_residuals(st.session_state['new_model'])
        st.plotly_chart(fig_res, use_container_width=True, config={'displayModeBar': False})
        resid = st.session_state['new_model'].resid
        c1, c2, c3 = st.columns(3)
        c1.metric("Media residuos", f"{resid.mean():.4f}")
        c2.metric("Desv. estándar", f"{resid.std():.4f}")
        c3.metric("Residuo máx. abs.", f"{resid.abs().max():.4f}")

        # Aprobar
        st.subheader("🚀 Activar en Dashboard")
        run_name = st.session_state.get('current_run_name', '')
        if run_name:
            st.info(f"Run guardado: `{run_name}`")
            available = sio.get_available_runs()
            current_latest = sio.get_default_run(available)
            already_active = current_latest == run_name

            if already_active:
                st.success("✅ Este modelo ya está activo en el Dashboard.")
            else:
                if current_latest:
                    st.warning(f"Dashboard usa actualmente: `{current_latest}`")
                if st.button("✅ Aprobar y activar en Dashboard",
                             type="primary", use_container_width=True):
                    sio.approve_model(run_name)
                    st.success(f"✅ Modelo `{run_name}` activado. El Dashboard ya lo usa.")
                    st.rerun()

# ── Tab 6: Historial ──────────────────────────────────────────────────────────

with tabs[5]:
    st.header("📋 Historial de Entrenamientos", divider='gray')

    historial = sio.load_training_log()

    if not historial:
        st.info("No hay entrenamientos registrados todavía.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total ejecuciones", len(historial))
        col2.metric("Último MAPE", f"{historial[-1]['mape_wf']:.2f}%")
        col3.metric("Mejor MAPE", f"{min(e['mape_wf'] for e in historial):.2f}%")

        st.markdown("---")

        df_log = pd.DataFrame([{
            'Fecha': e['timestamp'][:16].replace('T', ' '),
            'Usuario': e['usuario'], 'Modelo': e['modelo'],
            'SARIMA': f"{e['sarima_order']}{e['sarima_seasonal']}",
            'AIC': round(e['aic'], 2), 'MAPE WF': f"{e['mape_wf']:.2f}%",
            'Horizonte': e.get('horizonte', 6),
            'Trials válidos/total': f"{e['combinaciones_validas']}/{e['combinaciones_validas'] + e['combinaciones_descartadas']}"
        } for e in reversed(historial)])

        st.dataframe(df_log, use_container_width=True, hide_index=True)

        if len(historial) > 1:
            st.subheader("📈 Evolución del MAPE")
            fig_log = go.Figure()
            fig_log.add_trace(go.Scatter(
                x=[e['timestamp'][:16].replace('T', ' ') for e in historial],
                y=[e['mape_wf'] for e in historial],
                mode='lines+markers', line=dict(color='#1C7293', width=2),
                marker=dict(size=8)
            ))
            fig_log.update_layout(xaxis_title='Entrenamiento', yaxis_title='MAPE (%)',
                                   template='plotly_white', height=350)
            st.plotly_chart(fig_log, use_container_width=True, config={'displayModeBar': False})

        csv_log = df_log.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Exportar historial CSV", csv_log,
                           f"historial_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'><strong>App de Entrenamiento SARIMA</strong></div>",
            unsafe_allow_html=True)
