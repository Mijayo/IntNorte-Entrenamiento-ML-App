"""
============================================================================
PÁGINA: DASHBOARD DE NEGOCIO
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from google import genai

import supabase_io as sio
from auth_system import (init_session_state, show_login_page, show_user_info,
                         check_session_timeout, has_permission, show_header)

# ── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dashboard TIGGO 2", page_icon="🚗",
    layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stMetric {background-color:white;padding:15px;border-radius:10px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}
    h1 {color: #065A82;} h2 {color: #1C7293;}
    .role-badge {display:inline-block;padding:5px 15px;border-radius:15px;font-weight:bold;margin:5px 0;}
    .admin-badge   {background-color:#ffd700;color:#000;}
    .manager-badge {background-color:#4CAF50;color:white;}
    .analyst-badge {background-color:#2196F3;color:white;}
    .viewer-badge  {background-color:#9E9E9E;color:white;}
    </style>
""", unsafe_allow_html=True)

# ── Auth ─────────────────────────────────────────────────────────────────────

init_session_state()

if 'cache_llm_tiggo' not in st.session_state:
    st.session_state.cache_llm_tiggo = {}

if check_session_timeout():
    st.warning("⏱️ Tu sesión ha expirado.")
    st.stop()
if not st.session_state.authenticated:
    show_login_page("🚗 Dashboard TIGGO 2")
    st.stop()

# ── Selector de versión (sidebar) ────────────────────────────────────────────

available_runs = sio.get_available_runs()

if not available_runs:
    st.error("❌ No hay modelos entrenados. Ejecuta primero la app de **Entrenamiento**.")
    st.stop()

default_run = sio.get_default_run(available_runs)

selected_run = st.sidebar.selectbox(
    "📦 Versión del modelo",
    options=available_runs,
    index=available_runs.index(default_run) if default_run in available_runs else 0,
    format_func=sio.format_run_label,
    help="Selecciona qué ejecución quieres visualizar."
)

is_latest = sio.get_default_run(available_runs) == selected_run
st.sidebar.caption("🟢 Activo en producción" if is_latest else "🔵 Versión histórica")

# ── Datos de concesionarios (sidebar) ────────────────────────────────────────

st.sidebar.markdown("---")
with st.sidebar.expander("📂 Datos de Concesionarios", expanded=False):
    con_file = st.file_uploader(
        "Excel histórico de ventas", type=['xlsx', 'xls'], key="con_uploader",
        help="Archivo con columnas MARCA, MODELO2/MODELO3, FECHA_VENTA/FECHA-VENTA, DET_CC, AGE"
    )
    if con_file:
        with st.spinner("Procesando..."):
            df_con_raw = pd.read_excel(con_file, engine='openpyxl')
            df_con_raw.columns = [str(c).strip() for c in df_con_raw.columns]
            # Saltar fila de descripciones si la primera fila es todo texto
            if df_con_raw.iloc[0].apply(lambda x: isinstance(x, str)).all():
                df_con_raw = df_con_raw.iloc[1:].reset_index(drop=True)
            # Normalizar columna de fecha
            for _fc in ['FECHA_VENTA', 'FECHA-VENTA', 'FECHA VENTA']:
                if _fc in df_con_raw.columns:
                    df_con_raw[_fc] = pd.to_datetime(df_con_raw[_fc], errors='coerce')
                    if _fc != 'FECHA_VENTA':
                        df_con_raw = df_con_raw.rename(columns={_fc: 'FECHA_VENTA'})
                    break
            # Normalizar columna de modelo
            for _mc in ['MODELO2', 'MODELO3', 'MODELO']:
                if _mc in df_con_raw.columns:
                    if _mc != 'MODELO_NORM':
                        df_con_raw = df_con_raw.rename(columns={_mc: 'MODELO_NORM'})
                    break
            st.session_state['df_concesionarios'] = df_con_raw
            n_chery = len(df_con_raw[df_con_raw['MARCA'] == 'CHERY']) if 'MARCA' in df_con_raw.columns else len(df_con_raw)
            st.success(f"✅ {len(df_con_raw):,} registros · {n_chery:,} CHERY")

# ── Cargar datos ──────────────────────────────────────────────────────────────

with st.spinner('Cargando datos...'):
    metricas, pred_total, grid_search, walk_forward, hist_total = sio.load_precargados(selected_run)

# ── Header ───────────────────────────────────────────────────────────────────

show_header(
    "Dashboard Predicción TIGGO 2",
    f"Sistema de Predicción de Demanda  |  Modelo: {sio.format_run_label(selected_run)} {'🟢' if is_latest else '🔵'}"
)

role_badges = {
    'admin':   '<span class="role-badge admin-badge">👑 ADMIN</span>',
    'manager': '<span class="role-badge manager-badge">💼 GERENTE</span>',
    'analyst': '<span class="role-badge analyst-badge">📊 ANALISTA</span>',
    'viewer':  '<span class="role-badge viewer-badge">👁️ VIEWER</span>'
}
st.markdown(role_badges.get(st.session_state.role, ''), unsafe_allow_html=True)
show_user_info()

# ── Conexión Gemini ───────────────────────────────────────────────────────────

GEMINI_MODEL = 'gemini-2.5-flash'

try:
    gemini = genai.Client(api_key=st.secrets['GENAI_API_KEY'])
except Exception as _e:
    st.sidebar.warning(f"⚠️ Asistente IA no disponible: {_e}")
    gemini = None

# ── Variables contextuales para LLM ──────────────────────────────────────────

_orden     = metricas['mejor_modelo']['order']
_orden_est = metricas['mejor_modelo']['seasonal_order']
_mape      = metricas['walk_forward_validation']['mape']
_proximo   = pred_total['Predicción'].iloc[0]
_ic_inf    = pred_total['IC_Inferior'].iloc[0]
_ic_sup    = pred_total['IC_Superior'].iloc[0]
_prom_hist = hist_total.mean()
_ultimos_3 = hist_total.iloc[-3:].mean()
_tendencia_pct = ((_ultimos_3 - _prom_hist) / _prom_hist) * 100
_cfg       = metricas.get('configuracion', {})

context_tiggo = (
    f"Modelo SARIMA{_orden}{_orden_est}\n"
    f"AIC: {metricas['mejor_modelo']['aic']:.2f}  |  BIC: {metricas['mejor_modelo']['bic']:.2f}\n"
    f"MAPE (walk-forward): {_mape:.2f}%\n"
    f"Predicción próximo mes: {_proximo:.0f} uds  (IC 95%: {_ic_inf:.0f}–{_ic_sup:.0f})\n"
    f"Predicción total horizonte ({_cfg.get('horizonte', 6)} meses): {pred_total['Predicción'].sum():.0f} uds\n"
    f"Tendencia últimos 3 meses vs histórico: {_tendencia_pct:+.1f}%\n"
    f"Promedio histórico mensual: {_prom_hist:.1f}  |  Total ventas: {metricas['datos_limpios']['total_ventas']:,}\n"
    f"Período de datos: {metricas['datos_limpios']['periodo']}  |  Meses: {metricas['datos_limpios']['meses_datos']}"
)

# ── Tabs según rol ────────────────────────────────────────────────────────────

if st.session_state.role in ['admin', 'analyst']:
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones", "🔬 ACF/PACF",
                    "🔍 Grid Search", "🔄 Walk-Forward", "📋 Métricas Técnicas",
                    "🤖 Asistente IA", "🏪 Concesionarios"])
elif st.session_state.role == 'manager':
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones", "💼 Recomendaciones",
                    "🤖 Asistente IA", "🏪 Concesionarios"])
else:
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones"])

# ── Tab 1: Dashboard ──────────────────────────────────────────────────────────

with tabs[0]:
    st.header("📊 Dashboard General", divider='blue')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ventas", f"{metricas['datos_limpios']['total_ventas']:,}")
    col2.metric("Meses de Datos", metricas['datos_limpios']['meses_datos'])
    mape = metricas['walk_forward_validation']['mape']
    col3.metric("MAPE", f"{mape:.2f}%")
    col4.metric("Próximo Mes", f"{int(metricas['predicciones_futuras']['proximo_mes'])}")

    st.subheader("Serie Temporal Histórica")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist_total.index, y=hist_total.values,
        mode='lines+markers', name='Ventas Mensuales',
        line=dict(color='#065A82', width=3), marker=dict(size=6)
    ))
    fig_hist.add_hline(y=hist_total.mean(), line_dash="dash", line_color="red",
                       annotation_text=f"Media: {hist_total.mean():.1f}",
                       annotation_position="right")
    fig_hist.update_layout(title='Ventas Mensuales TIGGO 2',
                           xaxis_title='Fecha', yaxis_title='Ventas',
                           template='plotly_white', height=500, hovermode='x unified')
    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

    if st.session_state.role in ['admin', 'analyst']:
        with st.expander("📊 Estadísticas Descriptivas"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Ventas Mensuales:**")
                st.code(f"Promedio:  {hist_total.mean():.1f}\n"
                        f"Mediana:   {hist_total.median():.1f}\n"
                        f"Mínimo:    {hist_total.min():.0f}\n"
                        f"Máximo:    {hist_total.max():.0f}\n"
                        f"Desv.Est:  {hist_total.std():.1f}")
            with col2:
                st.markdown("**Información del Modelo:**")
                orden = metricas['mejor_modelo']['order']
                orden_est = metricas['mejor_modelo']['seasonal_order']
                st.code(f"Modelo: SARIMA{orden}{orden_est}\n"
                        f"AIC: {metricas['mejor_modelo']['aic']:.2f}\n"
                        f"BIC: {metricas['mejor_modelo']['bic']:.2f}")

# ── Tab 2: Predicciones ───────────────────────────────────────────────────────

with tabs[1]:
    st.header("🔮 Predicciones Futuras", divider='green')

    col1, col2, col3 = st.columns(3)
    col1.metric("Próximo Mes", f"{pred_total['Predicción'].iloc[0]:.0f} ventas")
    col2.metric("Total Horizonte", f"{pred_total['Predicción'].sum():.0f} ventas")
    col3.metric("Promedio Mensual", f"{pred_total['Predicción'].mean():.1f} ventas")

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=hist_total.index, y=hist_total.values,
                                   mode='lines', name='Histórico',
                                   line=dict(color='#065A82', width=2)))
    fig_pred.add_trace(go.Scatter(x=pred_total['Fecha'], y=pred_total['Predicción'],
                                   mode='lines+markers', name='Predicción',
                                   line=dict(color='red', width=3),
                                   marker=dict(size=10, symbol='star')))
    fig_pred.add_trace(go.Scatter(
        x=pred_total['Fecha'].tolist() + pred_total['Fecha'].tolist()[::-1],
        y=pred_total['IC_Superior'].tolist() + pred_total['IC_Inferior'].tolist()[::-1],
        fill='toself', fillcolor='rgba(255,0,0,0.15)',
        line=dict(color='rgba(255,0,0,0)'), name='IC 95%'
    ))
    fig_pred.add_shape(type="line",
                       x0=hist_total.index[-1], x1=hist_total.index[-1],
                       y0=0, y1=1, yref="paper",
                       line=dict(color="gray", width=2, dash="dash"))
    fig_pred.update_layout(title='Histórico + Predicción',
                           xaxis_title='Fecha', yaxis_title='Ventas',
                           template='plotly_white', height=600, hovermode='x unified')
    st.plotly_chart(fig_pred, use_container_width=True, config={'displayModeBar': False})

    st.subheader("📋 Tabla de Predicciones")
    st.dataframe(pred_total[['Mes', 'Predicción', 'IC_Inferior', 'IC_Superior']],
                 use_container_width=True, hide_index=True)

    if has_permission('exportar'):
        csv = pred_total.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Exportar CSV", csv,
                           f"predicciones_{datetime.now().strftime('%Y%m%d')}.csv",
                           "text/csv")

# ── Tab 3: Recomendaciones (manager) ─────────────────────────────────────────

if st.session_state.role == 'manager':
    with tabs[2]:
        st.header("💼 Recomendaciones de Compra", divider='orange')

        proximo = pred_total['Predicción'].iloc[0]
        ic_inf  = pred_total['IC_Inferior'].iloc[0]
        ic_sup  = pred_total['IC_Superior'].iloc[0]

        st.markdown(f"### 📊 Análisis para el próximo mes\n"
                    f"**Predicción:** {proximo:.0f} unidades  \n"
                    f"**Rango IC 95%:** {ic_inf:.0f} – {ic_sup:.0f} unidades")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**📉 Estrategia Conservadora:**\n\n"
                    f"Comprar: **{ic_sup * 1.1:.0f} unidades**\n\n"
                    f"- IC superior + 10%\n- Minimiza sobrestock")
        with col2:
            st.success(f"**📈 Estrategia Agresiva:**\n\n"
                       f"Comprar: **{ic_sup * 1.2:.0f} unidades**\n\n"
                       f"- IC superior + 20%\n- Maximiza cobertura")

        st.subheader("📈 Análisis de Tendencia")
        ultimos_3 = hist_total.iloc[-3:].mean()
        prom_hist = hist_total.mean()
        tendencia_pct = ((ultimos_3 - prom_hist) / prom_hist) * 100

        if tendencia_pct > 10:
            st.success(f"✅ **Tendencia CRECIENTE** — últimos 3 meses: {ultimos_3:.1f} "
                       f"(+{tendencia_pct:.1f}% vs histórico)")
            st.info("💡 Considera la **estrategia agresiva**")
        elif tendencia_pct < -10:
            st.warning(f"⚠️ **Tendencia DECRECIENTE** — últimos 3 meses: {ultimos_3:.1f} "
                       f"({tendencia_pct:.1f}% vs histórico)")
            st.info("💡 Considera la **estrategia conservadora**")
        else:
            st.info(f"📊 **Tendencia ESTABLE** — últimos 3 meses: {ultimos_3:.1f} "
                    f"({tendencia_pct:+.1f}% vs histórico)")
            st.info(f"💡 Usa la predicción directa: **{proximo:.0f} unidades**")

        if abs(proximo - prom_hist) / prom_hist > 0.3:
            st.warning(f"⚠️ La predicción ({proximo:.0f}) difiere >30% del promedio histórico "
                       f"({prom_hist:.1f}). Revisa factores externos.")

# ── Tab LLM (manager) ────────────────────────────────────────────────────────

if st.session_state.role == 'manager':
    with tabs[3]:
        st.header("🤖 Asistente IA", divider='violet')
        st.markdown(
            "Consulta al asistente sobre las predicciones, el modelo o las recomendaciones de compra. "
            "Las respuestas se basan únicamente en los datos del modelo SARIMA entrenado."
        )

        if gemini is None:
            st.error("⚠️ Configura `GENAI_API_KEY` en `.streamlit/secrets.toml` para usar el asistente.")
        else:
            with st.form(key='form_llm_tiggo_manager', border=False):
                question_m = st.text_input(
                    placeholder='Ej: ¿Cuántas unidades debería pedir para el próximo trimestre?',
                    key='input_llm_tiggo_manager', label='', label_visibility='collapsed'
                )
                btn_m = st.form_submit_button('Consultar al asistente')

            if btn_m and question_m:
                if question_m not in st.session_state.cache_llm_tiggo:
                    try:
                        prompt_tiggo = (
                            'Actúa como un Senior Analyst experto en predicción de demanda automotriz '
                            'y gestión de inventario de concesionarios.\n\n'
                            '## OBJETIVO:\n'
                            'Responder de forma precisa y accionable a la consulta del usuario '
                            'sobre el sistema de predicción TIGGO 2.\n\n'
                            f'## CONTEXTO DEL MODELO:\n{context_tiggo}\n\n'
                            f'## SOLICITUD:\n{question_m}\n\n'
                            '## INSTRUCCIONES OBLIGATORIAS:\n'
                            '1. Basa tu respuesta únicamente en los datos del contexto proporcionado.\n'
                            '2. Sé conciso y accionable; prioriza recomendaciones claras.\n'
                            '3. Si la pregunta está fuera del alcance de los datos, indícalo.\n\n'
                            '## FORMATO DE RESPUESTA:\n'
                            '- Máximo 3-4 líneas. Si hay una recomendación numérica, resáltala.'
                        )
                        with st.spinner('El asistente está procesando tu consulta...'):
                            response_m = gemini.models.generate_content(
                                model=GEMINI_MODEL, contents=prompt_tiggo
                            )
                            st.session_state.cache_llm_tiggo[question_m] = response_m.text
                    except Exception as e:
                        st.error(f'Error al consultar el asistente: {e}')

            if question_m in st.session_state.cache_llm_tiggo:
                st.success('Análisis completado')
                st.markdown(st.session_state.cache_llm_tiggo[question_m])

# ── Tabs técnicos (admin / analyst) ──────────────────────────────────────────

if st.session_state.role in ['admin', 'analyst']:

    # ACF / PACF
    with tabs[2]:
        st.header("🔬 Análisis ACF/PACF", divider='blue')
        acf_bytes, pacf_bytes = sio.load_acf_pacf_images(selected_run)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ACF - Autocorrelación")
            if acf_bytes:
                st.image(acf_bytes, use_container_width=True)
            else:
                st.warning("Imagen ACF no disponible")
        with col2:
            st.subheader("PACF - Autocorrelación Parcial")
            if pacf_bytes:
                st.image(pacf_bytes, use_container_width=True)
            else:
                st.warning("Imagen PACF no disponible")

    # Grid Search
    with tabs[3]:
        st.header("🔍 Grid Search de Parámetros", divider='blue')
        col1, col2, col3 = st.columns(3)
        col1.metric("Combinaciones Evaluadas", len(grid_search))
        col2.metric("Mejor MAPE", f"{grid_search['mape'].min():.2f}%")
        col3.metric("AIC del modelo seleccionado", f"{grid_search.loc[grid_search['mape'].idxmin(), 'aic']:.2f}")

        st.subheader("Top 10 Modelos por MAPE")
        top10 = grid_search.nsmallest(10, 'mape')[
            ['p', 'd', 'q', 'P', 'D', 'Q', 'mape', 'mae', 'rmse', 'aic', 'bic']
        ]
        st.dataframe(
            top10.style
                 .background_gradient(subset=['mape'], cmap='RdYlGn_r')
                 .background_gradient(subset=['aic'], cmap='Greens_r')
                 .format({'aic': '{:.2f}', 'bic': '{:.2f}',
                          'mape': '{:.2f}%', 'mae': '{:.2f}', 'rmse': '{:.2f}'}),
            use_container_width=True, hide_index=True
        )

        fig_grid = px.scatter(grid_search, x='aic', y='mape', color='p', size='mae',
                              hover_data=['p', 'd', 'q', 'P', 'D', 'Q'],
                              title='Grid Search: AIC vs MAPE')
        fig_grid.update_layout(height=500, template='plotly_white')
        st.plotly_chart(fig_grid, use_container_width=True, config={'displayModeBar': False})

    # Walk-Forward
    with tabs[4]:
        st.header("🔄 Walk-Forward Validation", divider='blue')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAPE Promedio", f"{walk_forward['error_pct'].mean():.2f}%")
        col2.metric("Mejor Mes", f"{walk_forward['error_pct'].min():.2f}%")
        col3.metric("Peor Mes", f"{walk_forward['error_pct'].max():.2f}%")
        col4.metric("Meses Evaluados", len(walk_forward))

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Scatter(x=walk_forward['fecha'], y=walk_forward['real'],
                                     mode='lines+markers', name='Real',
                                     line=dict(color='green', width=3)))
        fig_wf.add_trace(go.Scatter(x=walk_forward['fecha'], y=walk_forward['prediccion'],
                                     mode='lines+markers', name='Predicción',
                                     line=dict(color='red', width=3, dash='dash')))
        fig_wf.update_layout(title='Walk-Forward: Real vs Predicción',
                              xaxis_title='Fecha', yaxis_title='Ventas',
                              template='plotly_white', height=500)
        st.plotly_chart(fig_wf, use_container_width=True, config={'displayModeBar': False})

        wf_display = walk_forward.copy()
        wf_display['fecha'] = wf_display['fecha'].dt.strftime('%B %Y')
        wf_display.columns = ['Mes', 'Real', 'Predicción', 'Error Abs', 'Error %']
        st.dataframe(
            wf_display.style
                      .background_gradient(subset=['Error %'], cmap='RdYlGn_r')
                      .format({'Real': '{:.0f}', 'Predicción': '{:.1f}',
                               'Error Abs': '{:.2f}', 'Error %': '{:.2f}%'}),
            use_container_width=True, hide_index=True
        )

    # Métricas Técnicas
    with tabs[5]:
        st.header("📋 Métricas Técnicas Completas", divider='gray')
        col1, col2 = st.columns(2)
        orden = metricas['mejor_modelo']['order']
        orden_est = metricas['mejor_modelo']['seasonal_order']
        with col1:
            st.markdown("**Parámetros SARIMA:**")
            st.code(f"order = ({orden[0]}, {orden[1]}, {orden[2]})\n"
                    f"seasonal_order = ({orden_est[0]}, {orden_est[1]}, {orden_est[2]}, 12)\n\n"
                    f"p={orden[0]} (AR)  d={orden[1]} (dif)  q={orden[2]} (MA)\n"
                    f"P={orden_est[0]} (AR_s)  D={orden_est[1]} (dif_s)  "
                    f"Q={orden_est[2]} (MA_s)  m=12")
        with col2:
            st.markdown("**Métricas de Ajuste:**")
            cfg = metricas.get('configuracion', {})
            st.code(f"AIC: {metricas['mejor_modelo']['aic']:.2f}\n"
                    f"BIC: {metricas['mejor_modelo']['bic']:.2f}\n\n"
                    f"MAPE (walk-forward): {metricas['walk_forward_validation']['mape']:.2f}%\n\n"
                    f"Ventas: {metricas['datos_limpios']['total_ventas']:,}\n"
                    f"Meses:  {metricas['datos_limpios']['meses_datos']}\n"
                    f"Período: {metricas['datos_limpios']['periodo']}\n"
                    f"Horizonte: {cfg.get('horizonte', 6)} meses")

# ── Tab LLM (admin / analyst) ────────────────────────────────────────────────

if st.session_state.role in ['admin', 'analyst']:
    with tabs[6]:
        st.header("🤖 Asistente IA", divider='violet')
        st.markdown(
            "Consulta al asistente sobre las predicciones, el modelo SARIMA o las métricas de validación. "
            "Las respuestas se basan únicamente en los datos del modelo entrenado."
        )

        if gemini is None:
            st.error("⚠️ Configura `GENAI_API_KEY` en `.streamlit/secrets.toml` para usar el asistente.")
        else:
            with st.form(key='form_llm_tiggo_analyst', border=False):
                question_a = st.text_input(
                    placeholder='Ej: ¿Qué significa el MAPE obtenido? ¿Es fiable la predicción?',
                    key='input_llm_tiggo_analyst', label='', label_visibility='collapsed'
                )
                btn_a = st.form_submit_button('Consultar al asistente')

            if btn_a and question_a:
                if question_a not in st.session_state.cache_llm_tiggo:
                    try:
                        prompt_tiggo_a = (
                            'Actúa como un Senior Data Scientist experto en series temporales '
                            'y modelos SARIMA aplicados a predicción de demanda automotriz.\n\n'
                            '## OBJETIVO:\n'
                            'Responder de forma técnica y precisa a la consulta del analista '
                            'sobre el modelo o sus métricas de validación.\n\n'
                            f'## CONTEXTO DEL MODELO:\n{context_tiggo}\n\n'
                            f'## SOLICITUD:\n{question_a}\n\n'
                            '## INSTRUCCIONES OBLIGATORIAS:\n'
                            '1. Basa tu respuesta únicamente en los datos del contexto proporcionado.\n'
                            '2. Puedes usar terminología técnica (AIC, MAPE, walk-forward, etc.).\n'
                            '3. Si la pregunta está fuera del alcance de los datos, indícalo.\n\n'
                            '## FORMATO DE RESPUESTA:\n'
                            '- Conciso y técnico. Máximo 4-5 líneas.'
                        )
                        with st.spinner('El asistente está procesando tu consulta...'):
                            response_a = gemini.models.generate_content(
                                model=GEMINI_MODEL, contents=prompt_tiggo_a
                            )
                            st.session_state.cache_llm_tiggo[question_a] = response_a.text
                    except Exception as e:
                        st.error(f'Error al consultar el asistente: {e}')

            if question_a in st.session_state.cache_llm_tiggo:
                st.success('Análisis completado')
                st.markdown(st.session_state.cache_llm_tiggo[question_a])

# ── Tab Concesionarios (admin / analyst / manager) ────────────────────────────

if st.session_state.role in ['admin', 'analyst', 'manager']:
    con_idx = 7 if st.session_state.role in ['admin', 'analyst'] else 4

    with tabs[con_idx]:
        st.header("🏪 Ventas CHERY por Concesionario", divider='violet')

        if 'df_concesionarios' not in st.session_state:
            st.info(
                "👈 Carga el Excel histórico de ventas desde el panel lateral "
                "(**📂 Datos de Concesionarios**) para ver este análisis."
            )
        else:
            df_c = st.session_state['df_concesionarios'].copy()

            # Filtrar CHERY
            if 'MARCA' in df_c.columns:
                df_c = df_c[df_c['MARCA'] == 'CHERY']

            # Detectar columnas
            conc_col   = next((c for c in ['DET_CC', 'AGE', 'SUCURSAL', 'CONCESIONARIO']
                               if c in df_c.columns), None)
            ciudad_col = next((c for c in ['AGE', 'CIUDAD', 'REGION']
                               if c in df_c.columns), None)
            modelo_col = ('MODELO_NORM' if 'MODELO_NORM' in df_c.columns
                          else next((c for c in ['MODELO2', 'MODELO3', 'MODELO']
                                     if c in df_c.columns), None))
            fecha_col  = 'FECHA_VENTA' if 'FECHA_VENTA' in df_c.columns else None

            # Si DET_CC disponible, usarlo como concesionario y AGE como ciudad
            if conc_col == 'AGE' and 'DET_CC' in df_c.columns:
                conc_col   = 'DET_CC'
                ciudad_col = 'AGE'

            if not conc_col or len(df_c) == 0:
                st.warning("⚠️ No se encontró columna de concesionario (DET_CC / AGE) "
                           "o no hay registros CHERY en el archivo.")
            else:
                # ── Filtros ───────────────────────────────────────────────────
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    if fecha_col:
                        years_all = sorted(
                            df_c[fecha_col].dt.year.dropna().unique().astype(int),
                            reverse=True
                        )
                        years_sel = st.multiselect("Año", years_all, default=years_all)
                        if years_sel:
                            df_c = df_c[df_c[fecha_col].dt.year.isin(years_sel)]
                with col_f2:
                    if modelo_col:
                        modelos_all = ['Todos'] + sorted(df_c[modelo_col].dropna().unique())
                        modelo_sel = st.selectbox("Modelo", modelos_all)
                        if modelo_sel != 'Todos':
                            df_c = df_c[df_c[modelo_col] == modelo_sel]
                with col_f3:
                    if ciudad_col and ciudad_col != conc_col:
                        ciudades_all = ['Todas'] + sorted(df_c[ciudad_col].dropna().unique())
                        ciudad_sel = st.selectbox("Ciudad", ciudades_all)
                        if ciudad_sel != 'Todas':
                            df_c = df_c[df_c[ciudad_col] == ciudad_sel]

                if len(df_c) == 0:
                    st.warning("No hay datos con los filtros seleccionados.")
                else:
                    ventas_por_conc = df_c.groupby(conc_col).size().sort_values(ascending=False)
                    top_conc   = ventas_por_conc.index[0]
                    top_modelo = df_c[modelo_col].value_counts().index[0] if modelo_col else '—'

                    # ── KPIs ─────────────────────────────────────────────────
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total Ventas CHERY", f"{len(df_c):,}")
                    k2.metric("Concesionarios", len(ventas_por_conc))
                    k3.metric("Top Concesionario", top_conc)
                    k4.metric("Modelo más vendido", top_modelo)

                    st.markdown("---")

                    # ── Gráfico 1: barras horizontales por concesionario ──────
                    st.subheader("📊 Ventas totales por concesionario")
                    df_bar = ventas_por_conc.reset_index()
                    df_bar.columns = ['Concesionario', 'Ventas']
                    if ciudad_col and ciudad_col != conc_col:
                        df_bar['Ciudad'] = df_bar['Concesionario'].map(
                            df_c.groupby(conc_col)[ciudad_col].first()
                        )
                        fig_bar = px.bar(
                            df_bar, x='Ventas', y='Concesionario', color='Ciudad',
                            orientation='h', text='Ventas',
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )
                    else:
                        fig_bar = px.bar(
                            df_bar, x='Ventas', y='Concesionario',
                            orientation='h', text='Ventas',
                            color_discrete_sequence=['#1C7293']
                        )
                    fig_bar.update_traces(textposition='outside')
                    fig_bar.update_layout(
                        template='plotly_white',
                        height=max(350, 60 + len(df_bar) * 35),
                        yaxis={'categoryorder': 'total ascending'},
                        margin=dict(r=80), showlegend=True
                    )
                    st.plotly_chart(fig_bar, use_container_width=True,
                                    config={'displayModeBar': False})

                    # ── Gráfico 2: evolución mensual por concesionario ────────
                    if fecha_col:
                        st.subheader("📈 Evolución mensual por concesionario")
                        concs_disp = sorted(df_c[conc_col].dropna().unique())
                        concs_sel = st.multiselect(
                            "Selecciona concesionarios",
                            concs_disp,
                            default=concs_disp[:min(5, len(concs_disp))],
                            key="conc_ts_sel"
                        )
                        if concs_sel:
                            df_ts = (
                                df_c[df_c[conc_col].isin(concs_sel)]
                                .groupby([pd.Grouper(key=fecha_col, freq='ME'), conc_col])
                                .size().reset_index(name='Ventas')
                            )
                            fig_ts = px.line(
                                df_ts, x=fecha_col, y='Ventas', color=conc_col,
                                markers=True, template='plotly_white',
                                color_discrete_sequence=px.colors.qualitative.Plotly
                            )
                            fig_ts.update_layout(
                                height=420, hovermode='x unified',
                                xaxis_title='Mes', yaxis_title='Unidades',
                                legend_title='Concesionario'
                            )
                            st.plotly_chart(fig_ts, use_container_width=True,
                                            config={'displayModeBar': False})

                    # ── Gráfico 3: modelos por concesionario ──────────────────
                    if modelo_col:
                        st.subheader("🚗 Distribución de modelos por concesionario")
                        df_mod = (df_c.groupby([conc_col, modelo_col])
                                  .size().reset_index(name='Ventas'))
                        fig_mod = px.bar(
                            df_mod, x=conc_col, y='Ventas', color=modelo_col,
                            barmode='stack', template='plotly_white',
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_mod.update_layout(
                            height=450, xaxis_tickangle=-30,
                            xaxis_title='', yaxis_title='Unidades',
                            legend_title='Modelo'
                        )
                        st.plotly_chart(fig_mod, use_container_width=True,
                                        config={'displayModeBar': False})

                    # ── Tabla resumen ─────────────────────────────────────────
                    st.subheader("📋 Ranking de concesionarios")
                    group_cols = [conc_col]
                    if ciudad_col and ciudad_col != conc_col:
                        group_cols.insert(0, ciudad_col)
                    df_tabla = (df_c.groupby(group_cols)
                                .size().reset_index(name='Ventas')
                                .sort_values('Ventas', ascending=False))
                    df_tabla['% Total'] = (df_tabla['Ventas'] / df_tabla['Ventas'].sum() * 100).round(1)
                    df_tabla['Acumulado %'] = df_tabla['% Total'].cumsum().round(1)
                    st.dataframe(
                        df_tabla.style
                                .background_gradient(subset=['Ventas'], cmap='Blues')
                                .format({'% Total': '{:.1f}%', 'Acumulado %': '{:.1f}%'}),
                        use_container_width=True, hide_index=True
                    )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'><strong>Dashboard TIGGO 2</strong></div>",
            unsafe_allow_html=True)
