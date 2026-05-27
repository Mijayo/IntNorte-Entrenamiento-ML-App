"""
============================================================================
PÁGINA: DASHBOARD DE NEGOCIO
============================================================================
Tabs disponibles según rol:
  admin / analyst : Dashboard · Predicciones · ACF/PACF · Grid Search ·
                    Walk-Forward · Métricas Técnicas · Asistente IA · Concesionarios
  manager         : Dashboard · Predicciones · Recomendaciones · Asistente IA ·
                    Concesionarios
  viewer          : Dashboard · Predicciones

La proyección financiera (💰 Proyección Ingresos) se trasladó a la página
independiente pages/3_Proyeccion_Ingresos.py (2026-05-27).
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

import core.supabase_io as sio
from core.auth_system import (init_session_state, show_login_page, show_user_info,
                              check_session_timeout, has_permission, show_header)
from core.styles import kpi_card, section_header, apply_chart_theme, COLORS

# ── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dashboard TIGGO 2", page_icon="🚗",
    layout="wide", initial_sidebar_state="expanded"
)

# CSS inyectado por show_header() vía styles.get_global_css()

# ── Auth ─────────────────────────────────────────────────────────────────────

init_session_state()

if 'cache_llm_tiggo' not in st.session_state:
    st.session_state.cache_llm_tiggo = {}
if 'cache_llm_run' not in st.session_state:
    st.session_state.cache_llm_run = None

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

# Inicializar referencia de runs conocidos para el watcher realtime
if '_known_runs' not in st.session_state:
    st.session_state['_known_runs'] = list(available_runs)

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

# ── Realtime: watcher de nuevos entrenamientos (polling cada 30 s) ────────────

@st.fragment(run_every=30)
def _live_watcher() -> None:
    """Comprueba cada 30 s si hay un nuevo run en Supabase y notifica con toast."""
    current = sio.get_available_runs()
    prev    = st.session_state.get('_known_runs', [])
    new     = [r for r in current if r not in prev]
    if new:
        for r in new:
            st.toast(f"Nuevo modelo entrenado: {sio.format_run_label(r)}", icon="🔔")
        st.session_state['_known_runs'] = current
        st.rerun()

_live_watcher()

# ── Sidebar: indicador de datos de concesionarios ────────────────────────────

st.sidebar.markdown("---")
if 'df_concesionarios' in st.session_state:
    n_con = len(st.session_state['df_concesionarios'])
    st.sidebar.caption(f"🏪 Concesionarios: {n_con:,} registros cargados")
else:
    st.sidebar.caption("🏪 Concesionarios: sin datos — carga en la pestaña 🏪")

# ── Caché LLM: cargar desde Supabase si cambia el run seleccionado ───────────

if st.session_state.cache_llm_run != selected_run:
    st.session_state.cache_llm_tiggo = sio.load_llm_cache(selected_run)
    st.session_state.cache_llm_run = selected_run

# ── Cargar datos ──────────────────────────────────────────────────────────────

with st.spinner('Cargando datos...'):
    metricas, pred_total, grid_search, walk_forward, hist_total, _exog = sio.load_precargados(selected_run)

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
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones",
                    "🔬 ACF/PACF", "🔍 Grid Search", "🔄 Walk-Forward",
                    "📋 Métricas Técnicas", "🤖 Asistente IA", "🏪 Concesionarios"])
elif st.session_state.role == 'manager':
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones",
                    "💼 Recomendaciones", "🤖 Asistente IA", "🏪 Concesionarios"])
else:
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones"])

# ── Tab 1: Dashboard ──────────────────────────────────────────────────────────

with tabs[0]:
    st.markdown(section_header("Dashboard General", "📊"), unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    mape = metricas['walk_forward_validation']['mape']
    mape_color = "red" if mape > 15 else ("amber" if mape > 10 else "")
    col1.markdown(kpi_card("Total Ventas",    f"{metricas['datos_limpios']['total_ventas']:,}", "📦"), unsafe_allow_html=True)
    col2.markdown(kpi_card("Meses de Datos",  metricas['datos_limpios']['meses_datos'],         "📅", "blue"), unsafe_allow_html=True)
    col3.markdown(kpi_card("MAPE",            f"{mape:.2f}%",                                   "🎯", mape_color), unsafe_allow_html=True)
    col4.markdown(kpi_card("Próximo Mes",     f"{int(metricas['predicciones_futuras']['proximo_mes'])} uds", "🔮"), unsafe_allow_html=True)

    if mape > 15:
        st.error(
            f"⚠️ **MAPE {mape:.1f}% — Modelo de baja fiabilidad (umbral: 15%).** "
            "Las predicciones tienen un error medio superior al 15% sobre el valor real. "
            "Usa los valores del intervalo de confianza con precaución y considera "
            "reentrenar el modelo con datos más recientes o ampliar el histórico."
        )
    elif mape > 10:
        st.warning(
            f"ℹ️ **MAPE {mape:.1f}% — Precisión aceptable (10–15%).** "
            "Adecuado para planificación de rango, menos fiable para compromisos exactos."
        )

    st.markdown(section_header("Serie Temporal Histórica"), unsafe_allow_html=True)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist_total.index, y=hist_total.values,
        mode='lines+markers', name='Ventas Mensuales',
        line=dict(color=COLORS['primary'], width=2.5),
        marker=dict(size=5, color=COLORS['primary']),
        fill='tozeroy', fillcolor='rgba(32,201,151,0.06)',
    ))
    fig_hist.add_hline(
        y=hist_total.mean(), line_dash="dot", line_color=COLORS['accent'],
        annotation_text=f"Media: {hist_total.mean():.1f}",
        annotation_position="top right",
        annotation_font_color=COLORS['accent'],
    )
    apply_chart_theme(fig_hist, height=480, title='Ventas Mensuales — TIGGO 2')
    fig_hist.update_layout(hovermode='x unified', xaxis_title='Fecha', yaxis_title='Unidades')
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
    st.markdown(section_header("Predicciones Futuras", "🔮"), unsafe_allow_html=True)

    # ── Storytelling banner ───────────────────────────────────────────────────
    st.markdown("""
<div style="background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.25);
            border-radius:10px;padding:16px 20px;margin-bottom:18px;">
<span style="font-size:1.05rem;font-weight:600;color:#A78BFA;">Dos conceptos clave — no confundir</span><br>
<span style="color:#94A3B8;font-size:0.92rem;">
<strong style="color:#C9D8E6;">① Predicción mes a mes:</strong>
el modelo genera una estimación independiente para <em>cada mes</em> del horizonte.
No es un agregado — cada fila de la tabla es una predicción propia con su intervalo de confianza.<br><br>
<strong style="color:#C9D8E6;">② Horizonte de 6 meses:</strong>
es la ventana de visibilidad hacia adelante. En operación real, el equipo actualiza el histórico
cada mes con las ventas cerradas y relanza la predicción; la validación walk-forward (zona
<strong style="color:#A78BFA;">violeta</strong>) simula exactamente ese proceso —
el modelo predijo cada mes <strong>un solo paso adelante</strong>, con todos los datos anteriores disponibles.
Es la estimación más honesta del error real del sistema.
</span>
</div>
""", unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    mape_wf = walk_forward['error_pct'].mean()
    mape_color_wf = "red" if mape_wf > 15 else ("amber" if mape_wf > 10 else "")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(kpi_card("Próximo Mes",         f"{pred_total['Predicción'].iloc[0]:.0f} uds", "🔮"), unsafe_allow_html=True)
    col2.markdown(kpi_card("Total Horizonte",     f"{pred_total['Predicción'].sum():.0f} uds",   "📦", "blue"), unsafe_allow_html=True)
    col3.markdown(kpi_card("Promedio Mensual",    f"{pred_total['Predicción'].mean():.1f} uds",  "📊", "amber"), unsafe_allow_html=True)
    col4.markdown(kpi_card("MAPE real (1 mes)",   f"{mape_wf:.1f}%",                             "🎯", mape_color_wf), unsafe_allow_html=True)

    # ── Gráfico principal ─────────────────────────────────────────────────────
    fig_pred = go.Figure()

    # Región de walk-forward (fondo sombreado)
    if not walk_forward.empty:
        wf_x0 = walk_forward['fecha'].iloc[0]
        wf_x1 = walk_forward['fecha'].iloc[-1]
        fig_pred.add_shape(
            type="rect",
            x0=wf_x0, x1=wf_x1, y0=0, y1=1, yref="paper",
            fillcolor="rgba(167,139,250,0.06)",
            line=dict(width=0),
            layer="below",
        )
        fig_pred.add_annotation(
            x=wf_x0, y=1, yref="paper",
            text="◀ Validación walk-forward ▶",
            showarrow=False, xanchor="left",
            font=dict(color="#A78BFA", size=11, family="Rajdhani, sans-serif"),
            bgcolor="rgba(167,139,250,0.12)", borderpad=4,
        )

    # Histórico
    fig_pred.add_trace(go.Scatter(
        x=hist_total.index, y=hist_total.values,
        mode='lines', name='Histórico',
        line=dict(color=COLORS['primary'], width=2),
    ))

    # Walk-forward: predicciones del modelo (violeta)
    if not walk_forward.empty:
        fig_pred.add_trace(go.Scatter(
            x=walk_forward['fecha'], y=walk_forward['prediccion'],
            mode='lines+markers', name='Predicción walk-forward (1 mes)',
            line=dict(color=COLORS['purple'], width=2, dash='dot'),
            marker=dict(size=8, symbol='diamond', color=COLORS['purple'],
                        line=dict(color='#080D18', width=1.5)),
            customdata=walk_forward['error_pct'].values,
            hovertemplate='%{y:.0f} uds<br>Error: %{customdata:.1f}%<extra>WF predicción</extra>',
        ))

    # Predicción futura
    fig_pred.add_trace(go.Scatter(
        x=pred_total['Fecha'], y=pred_total['Predicción'],
        mode='lines+markers', name='Predicción futura',
        line=dict(color=COLORS['accent'], width=2.5),
        marker=dict(size=9, symbol='circle', color=COLORS['accent'],
                    line=dict(color='#080D18', width=1.5)),
    ))

    # Banda IC 95%
    fig_pred.add_trace(go.Scatter(
        x=pred_total['Fecha'].tolist() + pred_total['Fecha'].tolist()[::-1],
        y=pred_total['IC_Superior'].tolist() + pred_total['IC_Inferior'].tolist()[::-1],
        fill='toself', fillcolor='rgba(255,58,92,0.08)',
        line=dict(color='rgba(0,0,0,0)'), name='IC 95%',
    ))

    # Línea vertical: inicio de predicción futura
    fig_pred.add_shape(
        type="line",
        x0=hist_total.index[-1], x1=hist_total.index[-1],
        y0=0, y1=1, yref="paper",
        line=dict(color='rgba(100,116,139,0.6)', width=1.5, dash="dot"),
    )

    apply_chart_theme(fig_pred, height=580, title='Histórico · Validación Walk-Forward · Predicción — TIGGO 2')
    fig_pred.update_layout(hovermode='x unified', xaxis_title='Fecha', yaxis_title='Unidades')
    st.plotly_chart(fig_pred, use_container_width=True, config={'displayModeBar': False})

    if mape_wf > 15:
        st.error(
            f"⚠️ **Atención:** MAPE walk-forward = {mape_wf:.1f}% (objetivo: <15%). "
            "Para decisiones de compra, usa el **IC inferior** como referencia conservadora."
        )
    elif mape_wf > 10:
        st.warning(
            f"ℹ️ MAPE walk-forward = {mape_wf:.1f}% — aceptable, cerca del objetivo <15%."
        )
    else:
        st.success(f"✅ MAPE walk-forward = {mape_wf:.1f}% — por debajo del objetivo del 15%.")

    # ── Tablas ────────────────────────────────────────────────────────────────
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("📋 Predicción futura")
        st.dataframe(pred_total[['Mes', 'Predicción', 'IC_Inferior', 'IC_Superior']],
                     use_container_width=True, hide_index=True)
    with col_t2:
        st.subheader("🔄 Walk-forward (caso de uso real)")
        wf_show = walk_forward.copy()
        wf_show['fecha'] = wf_show['fecha'].dt.strftime('%B %Y')
        wf_show.columns = ['Mes', 'Real', 'Predicción', 'Error Abs', 'Error %']
        st.dataframe(
            wf_show.style
                   .background_gradient(subset=['Error %'], cmap='RdYlGn_r')
                   .format({'Real': '{:.0f}', 'Predicción': '{:.1f}',
                            'Error Abs': '{:.2f}', 'Error %': '{:.2f}%'}),
            use_container_width=True, hide_index=True
        )

    if has_permission('exportar'):
        csv = pred_total.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Exportar CSV predicciones", csv,
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
            "Consulta al asistente sobre las predicciones, el modelo activo o las recomendaciones de compra. "
            "Las respuestas se basan en los datos del modelo SARIMA entrenado. "
            "Para comparar con otros modelos (Prophet, XGBoost, Random Forest), usa la página **Comparativa ML**."
        )

        if gemini is None:
            st.error("⚠️ Configura `GENAI_API_KEY` en `.streamlit/secrets.toml` para usar el asistente.")
        else:
            with st.form(key='form_llm_tiggo_manager', border=False):
                question_m = st.text_input(
                    placeholder='Ej: ¿Cuántas unidades debería pedir para el próximo trimestre?',
                    key='input_llm_tiggo_manager', label='', label_visibility='collapsed',
                    max_chars=500
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
                            sio.save_llm_cache(selected_run, st.session_state.cache_llm_tiggo)
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
        st.markdown(section_header("Grid Search de Parámetros", "🔍"), unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.markdown(kpi_card("Combinaciones", len(grid_search), "🔢"), unsafe_allow_html=True)
        col2.markdown(kpi_card("Mejor MAPE", f"{grid_search['mape'].min():.2f}%", "🎯", "amber"), unsafe_allow_html=True)
        col3.markdown(kpi_card("AIC seleccionado", f"{grid_search.loc[grid_search['mape'].idxmin(), 'aic']:.0f}", "📐", "blue"), unsafe_allow_html=True)

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
                              color_continuous_scale='Teal')
        apply_chart_theme(fig_grid, height=480, title='Grid Search — AIC vs MAPE')
        st.plotly_chart(fig_grid, use_container_width=True, config={'displayModeBar': False})

    # Walk-Forward
    with tabs[4]:
        st.markdown(section_header("Walk-Forward Validation", "🔄"), unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(kpi_card("MAPE Promedio",   f"{walk_forward['error_pct'].mean():.2f}%", "📊", "amber"), unsafe_allow_html=True)
        col2.markdown(kpi_card("Mejor Mes",       f"{walk_forward['error_pct'].min():.2f}%",  "✅"), unsafe_allow_html=True)
        col3.markdown(kpi_card("Peor Mes",        f"{walk_forward['error_pct'].max():.2f}%",  "⚠️", "red"), unsafe_allow_html=True)
        col4.markdown(kpi_card("Meses Evaluados", len(walk_forward),                           "📅", "blue"), unsafe_allow_html=True)

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Scatter(
            x=walk_forward['fecha'], y=walk_forward['real'],
            mode='lines+markers', name='Real',
            line=dict(color=COLORS['primary'], width=2.5),
            marker=dict(size=7, color=COLORS['primary']),
        ))
        fig_wf.add_trace(go.Scatter(
            x=walk_forward['fecha'], y=walk_forward['prediccion'],
            mode='lines+markers', name='Predicción',
            line=dict(color=COLORS['accent'], width=2.5, dash='dot'),
            marker=dict(size=7, color=COLORS['accent'], symbol='diamond'),
        ))
        apply_chart_theme(fig_wf, height=480, title='Walk-Forward — Real vs Predicción')
        fig_wf.update_layout(hovermode='x unified', xaxis_title='Fecha', yaxis_title='Unidades')
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
            "Las respuestas se basan en los datos del modelo activo. "
            "Para comparar SARIMA con Prophet, XGBoost y Random Forest, usa la página **Comparativa ML**."
        )

        if gemini is None:
            st.error("⚠️ Configura `GENAI_API_KEY` en `.streamlit/secrets.toml` para usar el asistente.")
        else:
            with st.form(key='form_llm_tiggo_analyst', border=False):
                question_a = st.text_input(
                    placeholder='Ej: ¿Qué significa el MAPE obtenido? ¿Es fiable la predicción?',
                    key='input_llm_tiggo_analyst', label='', label_visibility='collapsed',
                    max_chars=500
                )
                btn_a = st.form_submit_button('Consultar al asistente')

            if btn_a and question_a:
                if question_a not in st.session_state.cache_llm_tiggo:
                    try:
                        prompt_tiggo_a = (
                            'Actúa como un Senior Data Scientist experto en series temporales '
                            'y modelos de predicción de demanda automotriz (SARIMA, Prophet, '
                            'Random Forest, XGBoost, Regresión Lineal).\n\n'
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
                            sio.save_llm_cache(selected_run, st.session_state.cache_llm_tiggo)
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

        # ── Uploader de datos ─────────────────────────────────────────────────
        with st.expander("📂 Cargar datos de concesionarios", expanded='df_concesionarios' not in st.session_state):
            st.caption("Columnas mínimas: MARCA · MODELO3 · FECHA-VENTA · CONCESIONARIO · CLI-DPTO/CLI-PROV")
            if 'df_concesionarios' in st.session_state:
                if st.button("🗑 Limpiar datos y cargar nuevo archivo", key="con_reset_btn"):
                    del st.session_state['df_concesionarios']
                    st.rerun()
            con_file = st.file_uploader(
                "Excel histórico de ventas", type=['xlsx', 'xls'], key="con_uploader_tab",
            )
            if con_file and 'df_concesionarios' not in st.session_state:
                with st.spinner("Validando y procesando..."):
                    try:
                        df_con_raw = pd.read_excel(con_file, engine='openpyxl')
                        df_con_raw.columns = [str(c).strip() for c in df_con_raw.columns]
                        if len(df_con_raw) > 0 and df_con_raw.iloc[0].apply(lambda x: isinstance(x, str)).all():
                            df_con_raw = df_con_raw.iloc[1:].reset_index(drop=True)

                        cols_raw = df_con_raw.columns.tolist()
                        errores_val = []

                        # Fecha
                        _fecha_ok = False
                        for _fc in ['FECHA_VENTA', 'FECHA-VENTA', 'FECHA VENTA']:
                            if _fc in cols_raw:
                                df_con_raw[_fc] = pd.to_datetime(df_con_raw[_fc], errors='coerce')
                                n_nf = df_con_raw[_fc].isna().sum()
                                if n_nf > 0:
                                    errores_val.append(f"⚠️ {n_nf} fechas no parseables en `{_fc}`.")
                                if _fc != 'FECHA_VENTA':
                                    df_con_raw = df_con_raw.rename(columns={_fc: 'FECHA_VENTA'})
                                _fecha_ok = True
                                break
                        if not _fecha_ok:
                            errores_val.append("❌ Columna de fecha no encontrada (FECHA_VENTA / FECHA-VENTA / FECHA VENTA).")

                        # Modelo
                        _modelo_ok = False
                        for _mc in ['MODELO2', 'MODELO3', 'MODELO']:
                            if _mc in cols_raw:
                                if _mc != 'MODELO_NORM':
                                    df_con_raw = df_con_raw.rename(columns={_mc: 'MODELO_NORM'})
                                _modelo_ok = True
                                break
                        if not _modelo_ok:
                            errores_val.append("❌ Columna de modelo no encontrada (MODELO2 / MODELO3 / MODELO).")

                        # MARCA
                        if 'MARCA' not in df_con_raw.columns:
                            errores_val.append("⚠️ Columna MARCA no encontrada — se usarán todos los registros.")

                        # Concesionario
                        if not any(c in df_con_raw.columns for c in ['DET_CC', 'AGE', 'SUCURSAL', 'CONCESIONARIO']):
                            errores_val.append("⚠️ Columna de concesionario no encontrada (CONCESIONARIO / DET_CC / AGE / SUCURSAL).")

                        for msg in errores_val:
                            (st.error if msg.startswith("❌") else st.warning)(msg)

                        if not any(m.startswith("❌") for m in errores_val):
                            st.session_state['df_concesionarios'] = df_con_raw
                            n_ch = (len(df_con_raw[df_con_raw['MARCA'] == 'CHERY'])
                                    if 'MARCA' in df_con_raw.columns else len(df_con_raw))
                            st.success(f"✅ {len(df_con_raw):,} registros cargados · {n_ch:,} CHERY")
                            st.rerun()
                    except Exception as _e:
                        st.error(f"❌ Error al leer el archivo: {_e}")

        if 'df_concesionarios' not in st.session_state:
            st.info("Carga el Excel de concesionarios usando el expander de arriba.")
        else:
            df_c = st.session_state['df_concesionarios'].copy()

            # Filtrar CHERY
            if 'MARCA' in df_c.columns:
                df_c = df_c[df_c['MARCA'] == 'CHERY']

            # Detectar columnas
            conc_col   = next((c for c in ['DET_CC', 'AGE', 'SUCURSAL', 'CONCESIONARIO']
                               if c in df_c.columns), None)
            ciudad_col = next((c for c in ['CLI-DPTO', 'CLI-PROV', 'AGE', 'CIUDAD', 'REGION']
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
                            color_discrete_sequence=[COLORS['primary']]
                        )
                    fig_bar.update_traces(textposition='outside',
                                          textfont=dict(color='#94A3B8', size=11))
                    apply_chart_theme(fig_bar, height=max(350, 60 + len(df_bar) * 35),
                                      title='Ventas por Concesionario')
                    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'},
                                          margin=dict(r=100), showlegend=True)
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
                                markers=True,
                                color_discrete_sequence=COLORS['series'],
                            )
                            apply_chart_theme(fig_ts, height=420,
                                              title='Evolución Mensual por Concesionario')
                            fig_ts.update_layout(hovermode='x unified',
                                                 xaxis_title='Mes', yaxis_title='Unidades',
                                                 legend_title='Concesionario')
                            st.plotly_chart(fig_ts, use_container_width=True,
                                            config={'displayModeBar': False})

                    # ── Gráfico 3: modelos por concesionario ──────────────────
                    if modelo_col:
                        st.subheader("🚗 Distribución de modelos por concesionario")
                        df_mod = (df_c.groupby([conc_col, modelo_col])
                                  .size().reset_index(name='Ventas'))
                        fig_mod = px.bar(
                            df_mod, x=conc_col, y='Ventas', color=modelo_col,
                            barmode='stack',
                            color_discrete_sequence=COLORS['series'],
                        )
                        apply_chart_theme(fig_mod, height=450,
                                          title='Distribución de Modelos por Concesionario')
                        fig_mod.update_layout(
                            xaxis_tickangle=-30, xaxis_title='', yaxis_title='Unidades',
                            legend_title='Modelo',
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

st.markdown(
    '<div class="app-footer">Sistema TIGGO 2 &nbsp;·&nbsp; ISDI &nbsp;·&nbsp; Predicción de Demanda</div>',
    unsafe_allow_html=True,
)
