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
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import json
from datetime import datetime
from pathlib import Path
import itertools
import io
import zipfile
import warnings
warnings.filterwarnings('ignore')

# Directorio base: siempre relativo a este fichero, en cualquier máquina
DATA_DIR = Path(__file__).parent / "data dashboard"

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
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

init_session_state()

# Verificar timeout de sesión
if check_session_timeout():
    st.warning("⏱️ Tu sesión ha expirado. Por favor inicia sesión nuevamente.")
    st.stop()

# ============================================================================
# AUTENTICACIÓN
# ============================================================================

if not st.session_state.authenticated:
    show_login_page("🤖 Entrenamiento de Modelos")
    st.stop()

# Verificar permisos (solo Admin y Analista)
if not has_permission('entrenar_modelos'):
    st.error("❌ No tienes permiso para acceder a esta aplicación")
    st.info("Esta aplicación es solo para **Administradores** y **Analistas de Datos**")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.title("🤖 Entrenamiento de Modelos SARIMA")
st.markdown("**Sistema de Entrenamiento Automatizado** | Post-Reunión Pedro León Pita")

show_user_info()

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_current_model():
    """Cargar modelo actual si existe"""
    try:
        # Intentar cargar métricas del modelo actual
        with open(DATA_DIR / 'metricas_mejoradas.json', 'r') as f:
            current_metrics = json.load(f)
        return current_metrics
    except:
        return None

def train_sarima_model(ventas, exog_data, order, seasonal_order):
    """Entrenar modelo SARIMA"""
    model = SARIMAX(
        ventas,
        exog=exog_data,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results = model.fit(disp=False, maxiter=200, method='lbfgs')
    return results

def perform_grid_search(train, test, train_exog, test_exog, progress_bar, status_text):
    """Grid search de parámetros SARIMA"""
    
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
            
            # Validar predicciones razonables
            if predictions.min() < 0 or predictions.max() > 100:
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
            
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Evaluando combinación {i+1}/{total}...")
        
        except Exception as e:
            continue
    
    return best_params, best_aic, best_mape, grid_results

def perform_walk_forward(ventas, exog_data, best_params, n_months=6):
    """Walk-forward validation"""
    
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
            
            if pred_wf.iloc[0] < 0 or pred_wf.iloc[0] > 100:
                continue
            
            error_pct = abs(test_wf.iloc[0] - pred_wf.iloc[0]) / (test_wf.iloc[0] + 0.1) * 100
            
            walk_forward_results.append({
                'fecha': test_wf.index[0],
                'real': test_wf.iloc[0],
                'prediccion': pred_wf.iloc[0],
                'error': abs(test_wf.iloc[0] - pred_wf.iloc[0]),
                'error_pct': error_pct
            })
        
        except Exception as e:
            continue
    
    return walk_forward_results

def create_download_package(modelo, predicciones, grid_results, walk_forward, 
                            historico, metricas, acf_fig, pacf_fig):
    """Crear paquete ZIP con todos los archivos para el dashboard"""
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Modelo PKL
        model_buffer = io.BytesIO()
        pickle.dump(modelo, model_buffer)
        zip_file.writestr('modelo_total_mejorado.pkl', model_buffer.getvalue())
        
        # Excel files
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
        
        # JSON
        zip_file.writestr('metricas_mejoradas.json', json.dumps(metricas, indent=2))
        
        # PNG files
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

tabs = st.tabs(["📤 Cargar Datos", "✅ Validación", "🤖 Entrenamiento", "📊 Comparación"])

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
        
        # Mostrar archivos cargados
        for file in uploaded_files:
            st.markdown(f"- **{file.name}** ({file.size / 1024:.1f} KB)")
        
        # Cargar y unificar
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
                    # Unificar
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
        
        # Ejecutar validación
        is_valid, results, warnings, errors = validate_dataframe(df_raw, "Datos Unificados")
        
        # Mostrar resultados
        show_validation_results(results, warnings, errors)
        
        st.markdown("---")
        
        # Preview de datos
        if st.checkbox("👁️ Ver Preview de Datos", value=True):
            preview_data(df_raw)
        
        # Gráficos de diagnóstico
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribución Temporal")
            plot_temporal_distribution(df_raw)
        
        with col2:
            st.subheader("❌ Datos Faltantes")
            plot_missing_data(df_raw)
        
        # Guardar validación
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
        
        # Configuración de entrenamiento
        st.subheader("⚙️ Configuración")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            modelo_filtro = st.text_input("Filtro Modelo (MODELO3)", value="TIGGO 2")
        with col2:
            marca_filtro = st.text_input("Filtro Marca", value="CHERY")
        with col3:
            eliminar_mes_actual = st.checkbox("Eliminar mes actual (incompleto)", value=True)
        
        # Botón de entrenamiento
        if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
            
            # Barra de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # ========================================
                # PASO 1: Preparar datos
                # ========================================
                status_text.text("📊 Preparando datos...")
                progress_bar.progress(0.1)
                
                df['FECHA-VENTA'] = pd.to_datetime(df['FECHA-VENTA'], errors='coerce')
                
                # Filtrar
                df_chery = df[df['MARCA'] == marca_filtro].copy()
                df_modelo = df_chery[df_chery['MODELO3'] == modelo_filtro].copy()
                
                # Eliminar mes actual si está seleccionado
                if eliminar_mes_actual:
                    fecha_limite = datetime.now().replace(day=1)
                    df_modelo = df_modelo[df_modelo['FECHA-VENTA'] < fecha_limite]
                    st.info(f"✅ Mes actual eliminado. Datos hasta: {fecha_limite.strftime('%Y-%m-%d')}")
                
                # Filtrar período 2021-2026
                df_modelo = df_modelo[
                    (df_modelo['FECHA-VENTA'] >= '2021-01-01')
                ]
                
                # Serie temporal
                df_modelo_temporal = df_modelo.set_index('FECHA-VENTA')
                ventas_modelo = df_modelo_temporal.resample('ME').size()
                
                st.success(f"✅ Serie temporal: {len(ventas_modelo)} meses, {len(df_modelo):,} ventas")
                
                # Variable exógena
                df_otros = df_chery[df_chery['MODELO3'] != modelo_filtro].copy()
                if eliminar_mes_actual:
                    df_otros = df_otros[df_otros['FECHA-VENTA'] < fecha_limite]
                df_otros = df_otros[df_otros['FECHA-VENTA'] >= '2021-01-01']
                
                ventas_otros = df_otros.set_index('FECHA-VENTA').resample('ME').size()
                exog_data = pd.DataFrame({
                    'ventas_otros': ventas_otros
                }).reindex(ventas_modelo.index, fill_value=0)
                
                progress_bar.progress(0.2)
                
                # ========================================
                # PASO 2: ACF/PACF
                # ========================================
                status_text.text("📊 Generando ACF/PACF...")
                
                fig_acf, ax_acf = plt.subplots(figsize=(12, 4))
                plot_acf(ventas_modelo, lags=24, ax=ax_acf)
                ax_acf.set_title('ACF - Autocorrelación')
                plt.tight_layout()
                
                fig_pacf, ax_pacf = plt.subplots(figsize=(12, 4))
                plot_pacf(ventas_modelo, lags=24, ax=ax_pacf, method='ywm')
                ax_pacf.set_title('PACF - Autocorrelación Parcial')
                plt.tight_layout()
                
                progress_bar.progress(0.3)
                
                # ========================================
                # PASO 3: Grid Search
                # ========================================
                status_text.text("🔍 Grid search de parámetros...")
                
                train_size = len(ventas_modelo) - 6
                train = ventas_modelo[:train_size]
                test = ventas_modelo[train_size:]
                train_exog = exog_data[:train_size]
                test_exog = exog_data[train_size:]
                
                best_params, best_aic, best_mape, grid_results = perform_grid_search(
                    train, test, train_exog, test_exog, progress_bar, status_text
                )
                
                df_grid = pd.DataFrame(grid_results).sort_values('aic')
                
                st.success(f"✅ Mejor modelo: SARIMA{best_params[0]}{best_params[1]} (AIC: {best_aic:.2f})")
                
                progress_bar.progress(0.6)
                
                # ========================================
                # PASO 4: Walk-Forward
                # ========================================
                status_text.text("🔄 Walk-forward validation...")
                
                walk_forward_results = perform_walk_forward(ventas_modelo, exog_data, best_params)
                df_wf = pd.DataFrame(walk_forward_results)
                mape_wf = df_wf['error_pct'].mean()
                
                st.success(f"✅ MAPE walk-forward: {mape_wf:.2f}%")
                
                progress_bar.progress(0.8)
                
                # ========================================
                # PASO 5: Modelo Final
                # ========================================
                status_text.text("🤖 Entrenando modelo final...")
                
                model_final = train_sarima_model(ventas_modelo, exog_data, best_params[0], best_params[1])
                
                # Predicciones
                exog_future = pd.DataFrame({
                    'ventas_otros': [exog_data['ventas_otros'].rolling(6).mean().iloc[-1]] * 6
                })
                
                forecast = model_final.forecast(steps=6, exog=exog_future)
                forecast_obj = model_final.get_forecast(steps=6, exog=exog_future)
                conf_int = forecast_obj.conf_int()
                
                last_date = ventas_modelo.index[-1]
                fechas_futuras = pd.date_range(start=last_date, periods=7, freq='ME')[1:]
                
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
                    'datos_limpios': {
                        'total_ventas': len(df_modelo),
                        'meses_datos': len(ventas_modelo),
                        'periodo': f"{ventas_modelo.index.min().strftime('%Y-%m')} a {ventas_modelo.index.max().strftime('%Y-%m')}"
                    },
                    'mejor_modelo': {
                        'order': best_params[0],
                        'seasonal_order': best_params[1],
                        'aic': float(best_aic),
                        'bic': float(model_final.bic)
                    },
                    'walk_forward_validation': {
                        'mape': float(mape_wf)
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
                
                st.success("✅ Modelo entrenado exitosamente. Ve a la pestaña **Comparación** para ver resultados.")
                
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
        # Cargar modelo actual
        current_model = load_current_model()
        new_metrics = st.session_state['new_metrics']
        
        st.subheader("📊 Tabla Comparativa")
        
        # Comparación
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
            
            # Calcular deltas
            mape_delta = new_metrics['walk_forward_validation']['mape'] - current_model['walk_forward_validation']['mape']
            aic_delta = new_metrics['mejor_modelo']['aic'] - current_model['mejor_modelo']['aic']
            pred_delta = new_metrics['predicciones_futuras']['proximo_mes'] - current_model['predicciones_futuras']['proximo_mes']
            
            comp_data['Δ'] = [
                f"{mape_delta:+.2f}%",
                f"{aic_delta:+.2f}",
                f"{pred_delta:+.0f}",
                "Cambió" if comp_data['Actual'][3] != comp_data['Nuevo'][3] else "="
            ]
            
            df_comp = pd.DataFrame(comp_data)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
            
            # Recomendación
            st.subheader("💡 Recomendación")
            
            mejora_mape = mape_delta < 0
            mejora_aic = aic_delta < 0
            
            if mejora_mape and mejora_aic:
                st.markdown('<div class="success-box">✅ <strong>RECOMENDACIÓN: APROBAR</strong><br>El nuevo modelo mejora tanto MAPE como AIC. Se recomienda implementarlo.</div>', unsafe_allow_html=True)
            elif mejora_mape or mejora_aic:
                st.markdown('<div class="warning-box">⚠️ <strong>RECOMENDACIÓN: REVISAR</strong><br>El nuevo modelo mejora en algunas métricas pero empeora en otras. Revisar con detalle.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="comparison-worse">❌ <strong>RECOMENDACIÓN: NO APROBAR</strong><br>El nuevo modelo empeora las métricas. No se recomienda implementarlo.</div>', unsafe_allow_html=True)
        
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
        st.dataframe(st.session_state['new_predictions'], use_container_width=True, hide_index=True)
        
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
                
                st.success("✅ Paquete generado. Descarga el ZIP y carga los archivos en el Dashboard de Negocio.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>App de Entrenamiento SARIMA</strong> | Interamericana Norte</p>
    <p>Equipo Mira Murati - ISDI Master 2026</p>
</div>
""", unsafe_allow_html=True)
