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
from datetime import datetime, date
import itertools
import warnings
warnings.filterwarnings('ignore')

import supabase_io as sio
from auth_system import (init_session_state, show_login_page, show_user_info,
                         check_session_timeout, has_permission, show_header)
from utils_validacion import (validate_dataframe, show_validation_results,
                              preview_data, plot_temporal_distribution,
                              plot_missing_data)

# ── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Entrenamiento SARIMA", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stProgress > div > div > div > div {background-color: #1C7293;}
    .success-box {background-color:#d4edda;border-left:4px solid #28a745;padding:15px;border-radius:5px;margin:10px 0;}
    .warning-box {background-color:#fff3cd;border-left:4px solid #ffc107;padding:15px;border-radius:5px;margin:10px 0;}
    .comparison-worse {background-color:#f8d7da;padding:10px;border-radius:5px;}
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

def run_adf_test(series):
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'statistic': round(result[0], 4), 'p_value': round(result[1], 4),
        'lags_used': result[2], 'critical_1pct': round(result[4]['1%'], 4),
        'critical_5pct': round(result[4]['5%'], 4),
        'critical_10pct': round(result[4]['10%'], 4),
        'is_stationary': result[1] < 0.05
    }


def train_sarima_model(ventas, exog_data, order, seasonal_order):
    model = SARIMAX(ventas, exog=exog_data, order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False, maxiter=200, method='lbfgs')


def perform_grid_search(train, test, train_exog, test_exog,
                        progress_bar, status_text, max_ventas):
    param_combinations = list(itertools.product(
        [1, 2, 3, 5, 7], [1], [0, 1, 2], [1], [1], [0, 1, 2]
    ))
    best_aic, best_params, best_mape = np.inf, None, np.inf
    grid_results, failures = [], 0
    total = len(param_combinations)

    for i, (p, d, q, P, D, Q) in enumerate(param_combinations):
        try:
            order = (p, d, q)
            seasonal_order = (P, D, Q, 12)
            model = SARIMAX(train, exog=train_exog, order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False, maxiter=200, method='lbfgs')
            predictions = results.forecast(steps=len(test), exog=test_exog)

            if predictions.min() < 0 or predictions.max() > max_ventas:
                failures += 1
            else:
                mae = mean_absolute_error(test, predictions)
                rmse = np.sqrt(mean_squared_error(test, predictions))
                mape = np.mean(np.abs((test - predictions) / (test + 0.1))) * 100
                aic, bic = results.aic, results.bic
                grid_results.append({'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q,
                                     'm': 12, 'mae': mae, 'rmse': rmse,
                                     'mape': mape, 'aic': aic, 'bic': bic})
                if aic < best_aic and mape < 100:
                    best_aic, best_mape, best_params = aic, mape, (order, seasonal_order)
        except Exception:
            failures += 1

        progress_bar.progress((i + 1) / total)
        status_text.text(f"Evaluando {i+1}/{total} | Válidos: {len(grid_results)} | Descartados: {failures}")

    return best_params, best_aic, best_mape, grid_results, failures


def perform_walk_forward(ventas, exog_data, best_params, n_months, max_ventas):
    results = []
    for i in range(len(ventas) - n_months, len(ventas)):
        try:
            model_wf = SARIMAX(ventas[:i], exog=exog_data[:i],
                               order=best_params[0], seasonal_order=best_params[1],
                               enforce_stationarity=False, enforce_invertibility=False)
            res_wf = model_wf.fit(disp=False, maxiter=200, method='lbfgs')
            pred = res_wf.forecast(steps=1, exog=exog_data[i:i+1])
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

tabs = st.tabs(["📤 Cargar Datos", "✅ Validación",
                "🤖 Entrenamiento", "📊 Comparación", "📋 Historial"])

# ── Tab 1: Cargar Datos ───────────────────────────────────────────────────────

with tabs[0]:
    st.header("📤 Carga de Datos", divider='blue')
    st.markdown("""
    **Instrucciones:**
    1. Carga uno o varios archivos Excel con datos de ventas
    2. Los archivos se unificarán automáticamente
    3. El sistema validará la calidad de los datos
    """)

    uploaded_files = st.file_uploader(
        "Selecciona archivo(s) Excel", type=['xlsx', 'xls'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} archivo(s) cargado(s)")
        for f in uploaded_files:
            st.markdown(f"- **{f.name}** ({f.size / 1024:.1f} KB)")

        if st.button("🔄 Procesar Archivos", type="primary"):
            with st.spinner("Procesando..."):
                dfs = []
                for f in uploaded_files:
                    try:
                        df = pd.read_excel(f, sheet_name='Hoja1', engine='openpyxl')
                        dfs.append(df)
                        st.success(f"✅ {f.name}: {len(df):,} filas")
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")
                if dfs:
                    df_unified = pd.concat(dfs, ignore_index=True)
                    st.session_state['df_raw'] = df_unified
                    st.success(f"✅ {len(df_unified):,} registros totales")
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

# ── Tab 3: Entrenamiento ──────────────────────────────────────────────────────

with tabs[2]:
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
            fecha_inicio = st.date_input("Fecha de inicio", value=date(2021, 1, 1))
            horizonte = st.slider("Horizonte (meses)", 3, 12, 6)
        col3, col4 = st.columns(2)
        with col3:
            max_ventas = st.number_input("Límite máximo ventas/mes",
                                         min_value=10, max_value=10000, value=100, step=10)
        with col4:
            eliminar_mes_actual = st.checkbox("Eliminar mes actual", value=True)

        st.markdown("---")

        if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Paso 1: Preparar datos
                status_text.text("📊 Preparando datos...")
                progress_bar.progress(0.05)
                df['FECHA-VENTA'] = pd.to_datetime(df['FECHA-VENTA'], errors='coerce')
                df_chery = df[df['MARCA'] == marca_filtro].copy()
                df_modelo = df_chery[df_chery['MODELO3'] == modelo_filtro].copy()

                if eliminar_mes_actual:
                    fecha_limite = datetime.now().replace(day=1)
                    df_modelo = df_modelo[df_modelo['FECHA-VENTA'] < fecha_limite]
                    st.info(f"✅ Datos hasta: {fecha_limite.strftime('%Y-%m-%d')}")

                fecha_inicio_str = fecha_inicio.strftime('%Y-%m-%d')
                df_modelo = df_modelo[df_modelo['FECHA-VENTA'] >= fecha_inicio_str]
                ventas_modelo = df_modelo.set_index('FECHA-VENTA').resample('ME').size()

                st.success(f"✅ {len(ventas_modelo)} meses · {len(df_modelo):,} ventas")

                df_otros = df_chery[df_chery['MODELO3'] != modelo_filtro].copy()
                if eliminar_mes_actual:
                    df_otros = df_otros[df_otros['FECHA-VENTA'] < fecha_limite]
                df_otros = df_otros[df_otros['FECHA-VENTA'] >= fecha_inicio_str]
                ventas_otros = df_otros.set_index('FECHA-VENTA').resample('ME').size()
                exog_data = pd.DataFrame({'ventas_otros': ventas_otros}) \
                              .reindex(ventas_modelo.index, fill_value=0)

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

                # Paso 4: Grid Search
                status_text.text("🔍 Grid search...")
                train_size = len(ventas_modelo) - horizonte
                train = ventas_modelo[:train_size]
                test = ventas_modelo[train_size:]
                train_exog = exog_data[:train_size]
                test_exog = exog_data[train_size:]

                best_params, best_aic, best_mape, grid_results, failures = perform_grid_search(
                    train, test, train_exog, test_exog, progress_bar, status_text, max_ventas
                )

                if best_params is None:
                    st.error(f"❌ Ninguna combinación produjo predicciones en [0, {max_ventas}].")
                    st.info("Aumenta el límite máximo de ventas.")
                    st.stop()

                df_grid = pd.DataFrame(grid_results).sort_values('aic')
                st.success(f"✅ SARIMA{best_params[0]}{best_params[1]} · AIC: {best_aic:.2f} · "
                           f"{len(grid_results)} válidas / {failures} descartadas")
                progress_bar.progress(0.60)

                # Paso 5: Walk-Forward
                status_text.text("🔄 Walk-forward validation...")
                wf_results = perform_walk_forward(ventas_modelo, exog_data, best_params,
                                                   horizonte, max_ventas)
                df_wf = pd.DataFrame(wf_results)

                if df_wf.empty:
                    st.error("❌ Walk-forward sin resultados válidos. Revisa el límite máximo.")
                    st.stop()

                mape_wf = df_wf['error_pct'].mean()
                st.success(f"✅ MAPE walk-forward: {mape_wf:.2f}% · {len(df_wf)}/{horizonte} meses")
                progress_bar.progress(0.80)

                # Paso 6: Modelo final
                status_text.text("🤖 Entrenando modelo final...")
                model_final = train_sarima_model(ventas_modelo, exog_data,
                                                  best_params[0], best_params[1])

                exog_future = pd.DataFrame({
                    'ventas_otros': [exog_data['ventas_otros'].rolling(6).mean().iloc[-1]] * horizonte
                })
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
                        'fecha_inicio': fecha_inicio_str, 'horizonte': horizonte,
                        'max_ventas': int(max_ventas)
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
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ── Tab 4: Comparación ────────────────────────────────────────────────────────

with tabs[3]:
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

# ── Tab 5: Historial ──────────────────────────────────────────────────────────

with tabs[4]:
    st.header("📋 Historial de Entrenamientos", divider='gray')

    log = sio.load_training_log()

    if not log:
        st.info("No hay entrenamientos registrados todavía.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total ejecuciones", len(log))
        col2.metric("Último MAPE", f"{log[-1]['mape_wf']:.2f}%")
        col3.metric("Mejor MAPE", f"{min(e['mape_wf'] for e in log):.2f}%")

        st.markdown("---")

        df_log = pd.DataFrame([{
            'Fecha': e['timestamp'][:16].replace('T', ' '),
            'Usuario': e['usuario'], 'Modelo': e['modelo'],
            'SARIMA': f"{e['sarima_order']}{e['sarima_seasonal']}",
            'AIC': round(e['aic'], 2), 'MAPE WF': f"{e['mape_wf']:.2f}%",
            'Horizonte': e.get('horizonte', 6),
            'Válidas/Total': f"{e['combinaciones_validas']}/{e['combinaciones_validas'] + e['combinaciones_descartadas']}"
        } for e in reversed(log)])

        st.dataframe(df_log, use_container_width=True, hide_index=True)

        if len(log) > 1:
            st.subheader("📈 Evolución del MAPE")
            fig_log = go.Figure()
            fig_log.add_trace(go.Scatter(
                x=[e['timestamp'][:16].replace('T', ' ') for e in log],
                y=[e['mape_wf'] for e in log],
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
