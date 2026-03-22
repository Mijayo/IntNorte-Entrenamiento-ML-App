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

import supabase_io as sio
from auth_system import (init_session_state, show_login_page, show_user_info,
                         check_session_timeout, has_permission)

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

# ── Cargar datos ──────────────────────────────────────────────────────────────

with st.spinner('Cargando datos...'):
    metricas, pred_total, grid_search, walk_forward, hist_total = sio.load_precargados(selected_run)

# ── Header ───────────────────────────────────────────────────────────────────

st.title("🚗 Dashboard Predicción TIGGO 2")
st.markdown(
    f"**Sistema de Predicción de Demanda** &nbsp;|&nbsp; "
    f"Modelo: `{sio.format_run_label(selected_run)}` "
    f"{'🟢' if is_latest else '🔵'}",
    unsafe_allow_html=True
)

role_badges = {
    'admin':   '<span class="role-badge admin-badge">👑 ADMIN</span>',
    'manager': '<span class="role-badge manager-badge">💼 GERENTE</span>',
    'analyst': '<span class="role-badge analyst-badge">📊 ANALISTA</span>',
    'viewer':  '<span class="role-badge viewer-badge">👁️ VIEWER</span>'
}
st.markdown(role_badges.get(st.session_state.role, ''), unsafe_allow_html=True)
show_user_info()

# ── Tabs según rol ────────────────────────────────────────────────────────────

if st.session_state.role in ['admin', 'analyst']:
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones", "🔬 ACF/PACF",
                    "🔍 Grid Search", "🔄 Walk-Forward", "📋 Métricas Técnicas"])
elif st.session_state.role == 'manager':
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones", "💼 Recomendaciones"])
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
        col2.metric("Mejor AIC", f"{grid_search['aic'].min():.2f}")
        col3.metric("MAPE (mejor)", f"{grid_search.loc[grid_search['aic'].idxmin(), 'mape']:.2f}%")

        st.subheader("Top 10 Modelos por AIC")
        top10 = grid_search.nsmallest(10, 'aic')[
            ['p', 'd', 'q', 'P', 'D', 'Q', 'aic', 'bic', 'mape', 'mae', 'rmse']
        ]
        st.dataframe(
            top10.style
                 .background_gradient(subset=['aic'], cmap='Greens_r')
                 .background_gradient(subset=['mape'], cmap='RdYlGn_r')
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

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'><strong>Dashboard TIGGO 2</strong></div>",
            unsafe_allow_html=True)
