"""
============================================================================
APP STREAMLIT: ENTRENAMIENTO DE MODELOS SARIMA
Para Admin y Analista de Datos
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
from datetime import datetime, date
from pathlib import Path
import itertools
import io
import zipfile
import warnings
warnings.filterwarnings('ignore')

# Directorio base: siempre relativo a este fichero, en cualquier máquina
DATA_DIR = Path(__file__).parent / "data dashboard"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Importar módulos propios
from auth_system import (init_session_state, show_login_page, show_user_info,
                         check_session_timeout, has_permission)
from utils_validacion import (validate_dataframe, show_validation_results,
                              preview_data, plot_temporal_distribution,
                              plot_missing_data, get_dataset_summary)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="Entrenamiento SARIMA - TIGGO 2",
    page_icon="🤖",
    layout="wide"
)

# ============================================================================
# ESTILOS CSS
# ============================================================================

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stProgress > div > div > div > div {background-color: #1C7293;}
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .comparison-better {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
    }
    .comparison-worse {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
    }
    .log-entry {
        background-color: white;
        border-left: 4px solid #1C7293;
        padding: 12px;
        border-radius: 5px;
        margin: 8px 0;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

init_session_state()

if check_session_timeout():
    st.warning("⏱️ Tu sesión ha expirado. Por favor inicia sesión nuevamente.")
    st.stop()

# ============================================================================
# AUTENTICACIÓN
# ============================================================================

if not st.session_state.authenticated:
    show_login_page("🤖 Entrenamiento de Modelos")
    st.stop()

if not has_permission('entrenar_modelos'):
    st.error("❌ No tienes permiso para acceder a esta aplicación")
    st.info("Esta aplicación es solo para **Administradores** y **Analistas de Datos**")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.title("🤖 Entrenamiento de Modelos SARIMA")
st.markdown("**Sistema de Entrenamiento Automatizado**")

show_user_info()

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_current_model():
    """Cargar métricas del modelo actual si existe"""
    try:
        with open(DATA_DIR / 'metricas_mejoradas.json', 'r') as f:
            return json.load(f)
    except Exception:
        return None


def save_training_log(entry):
    """Añadir entrada al historial de entrenamientos"""
    log_path = DATA_DIR / 'training_log.json'
    try:
        log = []
        if log_path.exists():
            with open(log_path, 'r') as f:
                log = json.load(f)
        log.append(entry)
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"No se pudo guardar el historial: {e}")


def load_training_log():
    """Cargar historial de entrenamientos"""
    log_path = DATA_DIR / 'training_log.json'
    try:
        if log_path.exists():
            with open(log_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def run_adf_test(series):
    """Test de Dickey-Fuller Aumentado para estacionariedad"""
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'statistic': round(result[0], 4),
        'p_value': round(result[1], 4),
        'lags_used': result[2],
        'critical_1pct': round(result[4]['1%'], 4),
        'critical_5pct': round(result[4]['5%'], 4),
        'critical_10pct': round(result[4]['10%'], 4),
        'is_stationary': result[1] < 0.05
    }


def train_sarima_model(ventas, exog_data, order, seasonal_order):
    """Entrenar modelo SARIMA con todos los datos disponibles"""
    model = SARIMAX(
        ventas,
        exog=exog_data,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False, maxiter=200, method='lbfgs')


def perform_grid_search(train, test, train_exog, test_exog,
                        progress_bar, status_text, max_ventas):
    """Grid search de parámetros SARIMA con conteo de fallos"""

    param_combinations = list(itertools.product(
        [1, 2, 3, 5, 7],  # p
        [1],              # d
        [0, 1, 2],        # q
        [1],              # P
        [1],              # D
        [0, 1, 2]         # Q
    ))

    best_aic = np.inf
    best_params = None
    best_mape = np.inf
    grid_results = []
    failures = 0
    total = len(param_combinations)

    for i, (p, d, q, P, D, Q) in enumerate(param_combinations):
        try:
            order = (p, d, q)
            seasonal_order = (P, D, Q, 12)

            model = SARIMAX(
                train,
                exog=train_exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            results = model.fit(disp=False, maxiter=200, method='lbfgs')
            predictions = results.forecast(steps=len(test), exog=test_exog)

            if predictions.min() < 0 or predictions.max() > max_ventas:
                failures += 1
                progress_bar.progress((i + 1) / total)
                status_text.text(
                    f"Evaluando {i+1}/{total} | "
                    f"Válidos: {len(grid_results)} | "
                    f"Fuera de rango: {failures}"
                )
                continue

            mae = mean_absolute_error(test, predictions)
            rmse = np.sqrt(mean_squared_error(test, predictions))
            mape = np.mean(np.abs((test - predictions) / (test + 0.1))) * 100
            aic = results.aic
            bic = results.bic

            grid_results.append({
                'p': p, 'd': d, 'q': q,
                'P': P, 'D': D, 'Q': Q, 'm': 12,
                'mae': mae, 'rmse': rmse, 'mape': mape,
                'aic': aic, 'bic': bic
            })

            if aic < best_aic and mape < 100:
                best_aic = aic
                best_mape = mape
                best_params = (order, seasonal_order)

        except Exception:
            failures += 1

        progress_bar.progress((i + 1) / total)
        status_text.text(
            f"Evaluando {i+1}/{total} | "
            f"Válidos: {len(grid_results)} | "
            f"Fuera de rango: {failures}"
        )

    return best_params, best_aic, best_mape, grid_results, failures


def perform_walk_forward(ventas, exog_data, best_params, n_months, max_ventas):
    """Walk-forward validation sobre los últimos n_months meses"""

    walk_forward_results = []

    for i in range(len(ventas) - n_months, len(ventas)):
        train_wf = ventas[:i]
        test_wf = ventas[i:i+1]
        train_exog_wf = exog_data[:i]
        test_exog_wf = exog_data[i:i+1]

        try:
            model_wf = SARIMAX(
                train_wf,
                exog=train_exog_wf,
                order=best_params[0],
                seasonal_order=best_params[1],
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            results_wf = model_wf.fit(disp=False, maxiter=200, method='lbfgs')
            pred_wf = results_wf.forecast(steps=1, exog=test_exog_wf)

            if pred_wf.iloc[0] < 0 or pred_wf.iloc[0] > max_ventas:
                continue

            error_pct = abs(test_wf.iloc[0] - pred_wf.iloc[0]) / (test_wf.iloc[0] + 0.1) * 100

            walk_forward_results.append({
                'fecha': test_wf.index[0],
                'real': test_wf.iloc[0],
                'prediccion': pred_wf.iloc[0],
                'error': abs(test_wf.iloc[0] - pred_wf.iloc[0]),
                'error_pct': error_pct
            })

        except Exception:
            continue

    return walk_forward_results


def plot_residuals(model_results):
    """Gráfico de diagnóstico de residuos (serie + distribución)"""
    residuals = model_results.resid

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Residuos en el tiempo", "Distribución de residuos"),
        column_widths=[0.65, 0.35]
    )

    fig.add_trace(
        go.Scatter(
            x=residuals.index,
            y=residuals.values,
            mode='lines',
            name='Residuo',
            line=dict(color='#1C7293', width=1.5)
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(
        go.Histogram(
            x=residuals.values,
            nbinsx=15,
            name='Distribución',
            marker_color='#1C7293',
            opacity=0.75
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="Diagnóstico de Residuos del Modelo",
        height=350,
        template='plotly_white',
        showlegend=False
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)

    return fig


def create_download_package(modelo, predicciones, grid_results, walk_forward,
                            historico, metricas, acf_fig, pacf_fig):
    """Crear paquete ZIP con todos los archivos para el dashboard"""

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        model_buffer = io.BytesIO()
        pickle.dump(modelo, model_buffer)
        zip_file.writestr('modelo_total_mejorado.pkl', model_buffer.getvalue())

        pred_buffer = io.BytesIO()
        predicciones.to_excel(pred_buffer, index=False, engine='openpyxl')
        zip_file.writestr('prediccion_total_mejorada.xlsx', pred_buffer.getvalue())

        grid_buffer = io.BytesIO()
        grid_results.to_excel(grid_buffer, index=False, engine='openpyxl')
        zip_file.writestr('grid_search_results.xlsx', grid_buffer.getvalue())

        wf_buffer = io.BytesIO()
        walk_forward.to_excel(wf_buffer, index=False, engine='openpyxl')
        zip_file.writestr('walk_forward_validation.xlsx', wf_buffer.getvalue())

        hist_buffer = io.BytesIO()
        historico.to_excel(hist_buffer, engine='openpyxl')
        zip_file.writestr('historico_total_mejorado.xlsx', hist_buffer.getvalue())

        zip_file.writestr('metricas_mejoradas.json', json.dumps(metricas, indent=2))

        acf_img_buffer = io.BytesIO()
        acf_fig.savefig(acf_img_buffer, format='png', dpi=150, bbox_inches='tight')
        zip_file.writestr('acf_plot.png', acf_img_buffer.getvalue())

        pacf_img_buffer = io.BytesIO()
        pacf_fig.savefig(pacf_img_buffer, format='png', dpi=150, bbox_inches='tight')
        zip_file.writestr('pacf_plot.png', pacf_img_buffer.getvalue())

    return zip_buffer.getvalue()


# ============================================================================
# MAIN APP
# ============================================================================

tabs = st.tabs(["📤 Cargar Datos", "✅ Validación", "🤖 Entrenamiento",
                "📊 Comparación", "📋 Historial"])

# ============================================================================
# TAB 1: CARGAR DATOS
# ============================================================================

with tabs[0]:
    st.header("📤 Carga de Datos", divider='blue')

    st.markdown("""
    **Instrucciones:**
    1. Carga uno o varios archivos Excel con datos de ventas
    2. Los archivos se unificarán automáticamente
    3. El sistema validará la calidad de los datos
    4. Podrás proceder al entrenamiento si todo es válido
    """)

    uploaded_files = st.file_uploader(
        "Selecciona archivo(s) Excel",
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        help="Puedes cargar múltiples archivos. Se unificarán automáticamente."
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} archivo(s) cargado(s)")

        for file in uploaded_files:
            st.markdown(f"- **{file.name}** ({file.size / 1024:.1f} KB)")

        if st.button("🔄 Procesar Archivos", type="primary"):
            with st.spinner("Procesando archivos..."):
                dfs = []

                for file in uploaded_files:
                    try:
                        df = pd.read_excel(file, sheet_name='Hoja1', engine='openpyxl')
                        dfs.append(df)
                        st.success(f"✅ {file.name}: {len(df):,} filas")
                    except Exception as e:
                        st.error(f"❌ Error en {file.name}: {str(e)}")

                if dfs:
                    df_unified = pd.concat(dfs, ignore_index=True)
                    st.session_state['df_raw'] = df_unified
                    st.session_state['files_loaded'] = True
                    st.success(f"✅ Datos unificados: {len(df_unified):,} registros totales")
                    st.rerun()

# ============================================================================
# TAB 2: VALIDACIÓN
# ============================================================================

with tabs[1]:
    st.header("✅ Validación de Datos", divider='green')

    if 'df_raw' not in st.session_state:
        st.info("👈 Primero carga los datos en la pestaña **Cargar Datos**")
    else:
        df_raw = st.session_state['df_raw']

        is_valid, results, warnings_val, errors = validate_dataframe(df_raw, "Datos Unificados")
        show_validation_results(results, warnings_val, errors)

        st.markdown("---")

        if st.checkbox("👁️ Ver Preview de Datos", value=True):
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

# ============================================================================
# TAB 3: ENTRENAMIENTO
# ============================================================================

with tabs[2]:
    st.header("🤖 Entrenamiento del Modelo", divider='orange')

    if 'df_validated' not in st.session_state:
        st.info("👈 Primero valida los datos en la pestaña **Validación**")
    elif not st.session_state.get('validation_passed', False):
        st.error("❌ Los datos no pasaron la validación. Corrige los errores antes de entrenar.")
    else:
        df = st.session_state['df_validated']

        # ---- Configuración ----
        st.subheader("⚙️ Configuración")

        col1, col2 = st.columns(2)
        with col1:
            modelo_filtro = st.text_input("Filtro Modelo (MODELO3)", value="TIGGO 2")
            marca_filtro = st.text_input("Filtro Marca", value="CHERY")
        with col2:
            fecha_inicio = st.date_input(
                "Fecha de inicio de la serie",
                value=date(2021, 1, 1),
                help="Se ignorarán ventas anteriores a esta fecha."
            )
            horizonte = st.slider(
                "Horizonte de predicción (meses)",
                min_value=3, max_value=12, value=6
            )

        col3, col4 = st.columns(2)
        with col3:
            max_ventas = st.number_input(
                "Límite máximo de ventas por mes",
                min_value=10, max_value=10000, value=100, step=10,
                help="Predicciones fuera del rango [0, límite] se descartan del grid search."
            )
        with col4:
            eliminar_mes_actual = st.checkbox(
                "Eliminar mes actual (incompleto)", value=True
            )

        st.markdown("---")

        if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # ================================================
                # PASO 1: Preparar datos
                # ================================================
                status_text.text("📊 Preparando datos...")
                progress_bar.progress(0.05)

                df['FECHA-VENTA'] = pd.to_datetime(df['FECHA-VENTA'], errors='coerce')

                df_chery = df[df['MARCA'] == marca_filtro].copy()
                df_modelo = df_chery[df_chery['MODELO3'] == modelo_filtro].copy()

                if eliminar_mes_actual:
                    fecha_limite = datetime.now().replace(day=1)
                    df_modelo = df_modelo[df_modelo['FECHA-VENTA'] < fecha_limite]
                    st.info(f"✅ Mes actual eliminado. Datos hasta: {fecha_limite.strftime('%Y-%m-%d')}")

                fecha_inicio_str = fecha_inicio.strftime('%Y-%m-%d')
                df_modelo = df_modelo[df_modelo['FECHA-VENTA'] >= fecha_inicio_str]

                df_modelo_temporal = df_modelo.set_index('FECHA-VENTA')
                ventas_modelo = df_modelo_temporal.resample('ME').size()

                st.success(
                    f"✅ Serie temporal: {len(ventas_modelo)} meses · "
                    f"{len(df_modelo):,} ventas · "
                    f"desde {fecha_inicio_str}"
                )

                # Variable exógena
                df_otros = df_chery[df_chery['MODELO3'] != modelo_filtro].copy()
                if eliminar_mes_actual:
                    df_otros = df_otros[df_otros['FECHA-VENTA'] < fecha_limite]
                df_otros = df_otros[df_otros['FECHA-VENTA'] >= fecha_inicio_str]

                ventas_otros = df_otros.set_index('FECHA-VENTA').resample('ME').size()
                exog_data = pd.DataFrame({
                    'ventas_otros': ventas_otros
                }).reindex(ventas_modelo.index, fill_value=0)

                progress_bar.progress(0.10)

                # ================================================
                # PASO 2: Test de estacionariedad (ADF)
                # ================================================
                status_text.text("📐 Test de estacionariedad ADF...")

                adf = run_adf_test(ventas_modelo)

                with st.expander("📐 Resultado Test ADF (Augmented Dickey-Fuller)", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Estadístico ADF", f"{adf['statistic']:.4f}")
                    c2.metric("p-valor", f"{adf['p_value']:.4f}")
                    c3.metric(
                        "¿Estacionaria?",
                        "✅ Sí" if adf['is_stationary'] else "⚠️ No",
                        delta="p < 0.05" if adf['is_stationary'] else "p ≥ 0.05",
                        delta_color="normal" if adf['is_stationary'] else "inverse"
                    )
                    st.caption(
                        f"Valores críticos — 1%: {adf['critical_1pct']} · "
                        f"5%: {adf['critical_5pct']} · "
                        f"10%: {adf['critical_10pct']}"
                    )
                    if not adf['is_stationary']:
                        st.info(
                            "La serie no es estacionaria (p ≥ 0.05). "
                            "El parámetro d=1 en SARIMA aplica diferenciación para corregirlo."
                        )

                progress_bar.progress(0.15)

                # ================================================
                # PASO 3: ACF / PACF
                # ================================================
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

                # ================================================
                # PASO 4: Grid Search
                # ================================================
                status_text.text("🔍 Grid search de parámetros...")

                train_size = len(ventas_modelo) - horizonte
                train = ventas_modelo[:train_size]
                test = ventas_modelo[train_size:]
                train_exog = exog_data[:train_size]
                test_exog = exog_data[train_size:]

                best_params, best_aic, best_mape, grid_results, failures = perform_grid_search(
                    train, test, train_exog, test_exog,
                    progress_bar, status_text, max_ventas
                )

                # Validar que grid search encontró un modelo válido
                if best_params is None:
                    st.error(
                        f"❌ Grid search completado pero ninguna combinación produjo "
                        f"predicciones válidas en el rango [0, {max_ventas}]."
                    )
                    st.info(
                        f"Se evaluaron {len(grid_results) + failures} combinaciones. "
                        f"Intenta aumentar el **Límite máximo de ventas** en la configuración."
                    )
                    st.stop()

                df_grid = pd.DataFrame(grid_results).sort_values('aic')

                st.success(
                    f"✅ Mejor modelo: SARIMA{best_params[0]}{best_params[1]} · "
                    f"AIC: {best_aic:.2f} · "
                    f"Combinaciones evaluadas: {len(grid_results)} válidas / {failures} descartadas"
                )

                progress_bar.progress(0.60)

                # ================================================
                # PASO 5: Walk-Forward Validation
                # ================================================
                status_text.text("🔄 Walk-forward validation...")

                walk_forward_results = perform_walk_forward(
                    ventas_modelo, exog_data, best_params, horizonte, max_ventas
                )
                df_wf = pd.DataFrame(walk_forward_results)

                if df_wf.empty:
                    st.error(
                        "❌ Walk-forward no produjo resultados válidos. "
                        f"Todas las predicciones cayeron fuera del rango [0, {max_ventas}]. "
                        "Revisa el límite máximo de ventas."
                    )
                    st.stop()

                mape_wf = df_wf['error_pct'].mean()
                st.success(
                    f"✅ Walk-forward completado — "
                    f"MAPE: {mape_wf:.2f}% · "
                    f"Meses evaluados: {len(df_wf)}/{horizonte}"
                )

                progress_bar.progress(0.80)

                # ================================================
                # PASO 6: Modelo Final
                # ================================================
                status_text.text("🤖 Entrenando modelo final...")

                model_final = train_sarima_model(
                    ventas_modelo, exog_data, best_params[0], best_params[1]
                )

                # Predicciones futuras
                exog_future = pd.DataFrame({
                    'ventas_otros': [
                        exog_data['ventas_otros'].rolling(6).mean().iloc[-1]
                    ] * horizonte
                })

                forecast = model_final.forecast(steps=horizonte, exog=exog_future)
                forecast_obj = model_final.get_forecast(steps=horizonte, exog=exog_future)
                conf_int = forecast_obj.conf_int()

                last_date = ventas_modelo.index[-1]
                fechas_futuras = pd.date_range(
                    start=last_date, periods=horizonte + 1, freq='ME'
                )[1:]

                predicciones = pd.DataFrame({
                    'Fecha': fechas_futuras,
                    'Mes': fechas_futuras.strftime('%B %Y'),
                    'Predicción': forecast.values.round(1),
                    'IC_Inferior': conf_int.iloc[:, 0].values.round(1),
                    'IC_Superior': conf_int.iloc[:, 1].values.round(1)
                })

                # Métricas
                metricas = {
                    'fecha_entrenamiento': datetime.now().strftime('%Y%m%d_%H%M%S'),
                    'usuario': st.session_state.username,
                    'configuracion': {
                        'modelo_filtro': modelo_filtro,
                        'marca_filtro': marca_filtro,
                        'fecha_inicio': fecha_inicio_str,
                        'horizonte': horizonte,
                        'max_ventas': int(max_ventas)
                    },
                    'datos_limpios': {
                        'total_ventas': len(df_modelo),
                        'meses_datos': len(ventas_modelo),
                        'periodo': (
                            f"{ventas_modelo.index.min().strftime('%Y-%m')} a "
                            f"{ventas_modelo.index.max().strftime('%Y-%m')}"
                        )
                    },
                    'mejor_modelo': {
                        'order': list(best_params[0]),
                        'seasonal_order': list(best_params[1]),
                        'aic': float(best_aic),
                        'bic': float(model_final.bic),
                        'combinaciones_validas': len(grid_results),
                        'combinaciones_descartadas': failures
                    },
                    'adf_test': adf,
                    'walk_forward_validation': {
                        'mape': float(mape_wf),
                        'meses_evaluados': len(df_wf)
                    },
                    'predicciones_futuras': {
                        'proximo_mes': float(predicciones['Predicción'].iloc[0])
                    }
                }

                progress_bar.progress(1.0)
                status_text.text("✅ ¡Entrenamiento completado!")

                # Guardar en session state
                st.session_state['new_model'] = model_final
                st.session_state['new_predictions'] = predicciones
                st.session_state['new_grid'] = df_grid
                st.session_state['new_walkforward'] = df_wf
                st.session_state['new_historico'] = ventas_modelo
                st.session_state['new_metrics'] = metricas
                st.session_state['new_acf_fig'] = fig_acf
                st.session_state['new_pacf_fig'] = fig_pacf
                st.session_state['training_complete'] = True

                # Guardar en historial
                save_training_log({
                    'timestamp': datetime.now().isoformat(),
                    'usuario': st.session_state.username,
                    'modelo': modelo_filtro,
                    'marca': marca_filtro,
                    'fecha_inicio': fecha_inicio_str,
                    'horizonte': horizonte,
                    'max_ventas': int(max_ventas),
                    'sarima_order': list(best_params[0]),
                    'sarima_seasonal': list(best_params[1]),
                    'aic': float(best_aic),
                    'mape_wf': float(mape_wf),
                    'meses_datos': len(ventas_modelo),
                    'combinaciones_validas': len(grid_results),
                    'combinaciones_descartadas': failures
                })

                st.success(
                    "✅ Modelo entrenado y guardado en historial. "
                    "Ve a la pestaña **Comparación** para ver resultados."
                )

            except Exception as e:
                st.error(f"❌ Error durante entrenamiento: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# TAB 4: COMPARACIÓN
# ============================================================================

with tabs[3]:
    st.header("📊 Comparación: Nuevo vs Actual", divider='green')

    if 'training_complete' not in st.session_state:
        st.info("👈 Primero entrena un modelo en la pestaña **Entrenamiento**")
    else:
        current_model = load_current_model()
        new_metrics = st.session_state['new_metrics']

        st.subheader("📊 Tabla Comparativa")

        if current_model:
            comp_data = {
                'Métrica': ['MAPE Walk-Forward', 'AIC', 'Predicción Próximo Mes', 'Modelo'],
                'Actual': [
                    f"{current_model['walk_forward_validation']['mape']:.2f}%",
                    f"{current_model['mejor_modelo']['aic']:.2f}",
                    f"{current_model['predicciones_futuras']['proximo_mes']:.0f}",
                    f"{current_model['mejor_modelo']['order']}{current_model['mejor_modelo']['seasonal_order']}"
                ],
                'Nuevo': [
                    f"{new_metrics['walk_forward_validation']['mape']:.2f}%",
                    f"{new_metrics['mejor_modelo']['aic']:.2f}",
                    f"{new_metrics['predicciones_futuras']['proximo_mes']:.0f}",
                    f"{new_metrics['mejor_modelo']['order']}{new_metrics['mejor_modelo']['seasonal_order']}"
                ]
            }

            mape_delta = (
                new_metrics['walk_forward_validation']['mape']
                - current_model['walk_forward_validation']['mape']
            )
            aic_delta = (
                new_metrics['mejor_modelo']['aic']
                - current_model['mejor_modelo']['aic']
            )
            pred_delta = (
                new_metrics['predicciones_futuras']['proximo_mes']
                - current_model['predicciones_futuras']['proximo_mes']
            )

            comp_data['Δ'] = [
                f"{mape_delta:+.2f}%",
                f"{aic_delta:+.2f}",
                f"{pred_delta:+.0f}",
                "Cambió" if comp_data['Actual'][3] != comp_data['Nuevo'][3] else "="
            ]

            df_comp = pd.DataFrame(comp_data)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            st.subheader("💡 Recomendación")

            mejora_mape = mape_delta < 0
            mejora_aic = aic_delta < 0

            if mejora_mape and mejora_aic:
                st.markdown(
                    '<div class="success-box">✅ <strong>RECOMENDACIÓN: APROBAR</strong>'
                    '<br>El nuevo modelo mejora tanto MAPE como AIC. Se recomienda implementarlo.</div>',
                    unsafe_allow_html=True
                )
            elif mejora_mape or mejora_aic:
                st.markdown(
                    '<div class="warning-box">⚠️ <strong>RECOMENDACIÓN: REVISAR</strong>'
                    '<br>El nuevo modelo mejora en algunas métricas pero empeora en otras. Revisar con detalle.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="comparison-worse">❌ <strong>RECOMENDACIÓN: NO APROBAR</strong>'
                    '<br>El nuevo modelo empeora las métricas. No se recomienda implementarlo.</div>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("No hay modelo actual para comparar. Este será el primer modelo.")

        # Métricas del nuevo modelo
        st.subheader("📋 Métricas del Nuevo Modelo")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAPE", f"{new_metrics['walk_forward_validation']['mape']:.2f}%")
        with col2:
            st.metric("AIC", f"{new_metrics['mejor_modelo']['aic']:.2f}")
        with col3:
            st.metric("Próximo Mes", f"{new_metrics['predicciones_futuras']['proximo_mes']:.0f}")
        with col4:
            st.metric("Meses Datos", new_metrics['datos_limpios']['meses_datos'])

        # Preview predicciones
        st.subheader("🔮 Preview Predicciones")
        st.dataframe(
            st.session_state['new_predictions'],
            use_container_width=True,
            hide_index=True
        )

        # Diagnóstico de residuos
        st.subheader("🔬 Diagnóstico de Residuos")
        with st.spinner("Generando diagnóstico..."):
            fig_res = plot_residuals(st.session_state['new_model'])
            st.plotly_chart(fig_res, use_container_width=True,
                            config={'displayModeBar': False})

            resid = st.session_state['new_model'].resid
            col1, col2, col3 = st.columns(3)
            col1.metric("Media residuos", f"{resid.mean():.4f}",
                        help="Debe ser cercana a 0")
            col2.metric("Desv. estándar", f"{resid.std():.4f}")
            col3.metric("Residuo máx. abs.", f"{resid.abs().max():.4f}")

        # Descargar paquete
        st.subheader("📥 Descargar Paquete para Dashboard")

        st.info("""
        **El paquete incluye:**
        - Modelo entrenado (.pkl)
        - Predicciones futuras (.xlsx)
        - Resultados grid search (.xlsx)
        - Walk-forward validation (.xlsx)
        - Histórico (.xlsx)
        - Métricas (.json)
        - Gráficos ACF/PACF (.png)
        """)

        if st.button("📦 Generar Paquete ZIP", type="primary", use_container_width=True):
            with st.spinner("Generando paquete..."):
                zip_data = create_download_package(
                    st.session_state['new_model'],
                    st.session_state['new_predictions'],
                    st.session_state['new_grid'],
                    st.session_state['new_walkforward'],
                    st.session_state['new_historico'],
                    st.session_state['new_metrics'],
                    st.session_state['new_acf_fig'],
                    st.session_state['new_pacf_fig']
                )

                st.download_button(
                    label="⬇️ Descargar Paquete Dashboard",
                    data=zip_data,
                    file_name=f"modelo_tiggo2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )

                st.success(
                    "✅ Paquete generado. "
                    "Descarga el ZIP y extrae los archivos en la carpeta 'data dashboard/'."
                )

# ============================================================================
# TAB 5: HISTORIAL
# ============================================================================

with tabs[4]:
    st.header("📋 Historial de Entrenamientos", divider='gray')

    log = load_training_log()

    if not log:
        st.info("No hay entrenamientos registrados todavía.")
    else:
        # Métricas resumen
        col1, col2, col3 = st.columns(3)
        col1.metric("Total ejecuciones", len(log))
        col2.metric("Último MAPE", f"{log[-1]['mape_wf']:.2f}%")
        col3.metric("Mejor MAPE histórico",
                    f"{min(e['mape_wf'] for e in log):.2f}%")

        st.markdown("---")

        # Tabla resumen
        df_log = pd.DataFrame([{
            'Fecha': e['timestamp'][:16].replace('T', ' '),
            'Usuario': e['usuario'],
            'Modelo': e['modelo'],
            'SARIMA': f"{e['sarima_order']}{e['sarima_seasonal']}",
            'AIC': round(e['aic'], 2),
            'MAPE WF': f"{e['mape_wf']:.2f}%",
            'Horizonte': e.get('horizonte', 6),
            'Válidas/Total': f"{e['combinaciones_validas']}/{e['combinaciones_validas'] + e['combinaciones_descartadas']}"
        } for e in reversed(log)])

        st.dataframe(
            df_log.style.background_gradient(
                subset=['AIC'], cmap='Greens_r'
            ),
            use_container_width=True,
            hide_index=True
        )

        # Evolución del MAPE a lo largo del tiempo
        if len(log) > 1:
            st.subheader("📈 Evolución del MAPE")
            fig_log = go.Figure()
            fig_log.add_trace(go.Scatter(
                x=[e['timestamp'][:16].replace('T', ' ') for e in log],
                y=[e['mape_wf'] for e in log],
                mode='lines+markers',
                line=dict(color='#1C7293', width=2),
                marker=dict(size=8)
            ))
            fig_log.update_layout(
                xaxis_title='Entrenamiento',
                yaxis_title='MAPE (%)',
                template='plotly_white',
                height=350
            )
            st.plotly_chart(fig_log, use_container_width=True,
                            config={'displayModeBar': False})

        # Exportar historial
        csv_log = df_log.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exportar historial CSV",
            data=csv_log,
            file_name=f"historial_entrenamientos_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>App de Entrenamiento SARIMA</strong></p>
</div>
""", unsafe_allow_html=True)
