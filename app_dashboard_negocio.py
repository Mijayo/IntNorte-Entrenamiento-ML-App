"""
============================================================================
APP STREAMLIT: DASHBOARD DE NEGOCIO
Con login, datos precargados y vistas por rol
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Directorio base: siempre relativo a este fichero, en cualquier máquina
DATA_DIR = Path(__file__).parent / "data dashboard"


def get_available_runs():
    """Devuelve lista de runs disponibles ordenados del más reciente al más antiguo"""
    if not DATA_DIR.exists():
        return []
    return sorted(
        [d.name for d in DATA_DIR.iterdir()
         if d.is_dir() and d.name[:8].isdigit()],
        reverse=True
    )


def get_default_run(runs):
    """Devuelve el run activo según latest.txt, o el más reciente si no existe"""
    latest_path = DATA_DIR / 'latest.txt'
    if latest_path.exists():
        candidate = latest_path.read_text().strip()
        if candidate in runs:
            return candidate
    return runs[0] if runs else None


def format_run_label(run_name):
    """Formatea 20260322_143000 → 22/03/2026 14:30"""
    try:
        dt = datetime.strptime(run_name, '%Y%m%d_%H%M%S')
        return dt.strftime('%d/%m/%Y  %H:%M')
    except ValueError:
        return run_name


# Importar sistema de autenticación
from auth_system import (init_session_state, show_login_page, show_user_info,
                         check_session_timeout, has_permission)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="Dashboard TIGGO 2",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    h1 {color: #065A82;}
    h2 {color: #1C7293;}
    .role-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 15px;
        font-weight: bold;
        margin: 5px 0;
    }
    .admin-badge {background-color: #ffd700; color: #000;}
    .manager-badge {background-color: #4CAF50; color: white;}
    .analyst-badge {background-color: #2196F3; color: white;}
    .viewer-badge {background-color: #9E9E9E; color: white;}
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

init_session_state()

# Verificar timeout
if check_session_timeout():
    st.warning("⏱️ Tu sesión ha expirado. Por favor inicia sesión nuevamente.")
    st.stop()

# ============================================================================
# AUTENTICACIÓN
# ============================================================================

if not st.session_state.authenticated:
    show_login_page("🚗 Dashboard TIGGO 2")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.title("🚗 Dashboard Predicción TIGGO 2")
st.markdown(
    f"**Sistema de Predicción de Demanda** &nbsp;|&nbsp; "
    f"Modelo: `{format_run_label(selected_run)}` "
    f"{'🟢' if is_latest else '🔵'}",
    unsafe_allow_html=True
)

# Badge de rol
role_badges = {
    'admin': '<span class="role-badge admin-badge">👑 ADMIN</span>',
    'manager': '<span class="role-badge manager-badge">💼 GERENTE</span>',
    'analyst': '<span class="role-badge analyst-badge">📊 ANALISTA</span>',
    'viewer': '<span class="role-badge viewer-badge">👁️ VIEWER</span>'
}

st.markdown(role_badges.get(st.session_state.role, ''), unsafe_allow_html=True)

show_user_info()

# ============================================================================
# CARGAR DATOS PRECARGADOS
# ============================================================================

@st.cache_data
def load_precargados(run_folder: str):
    """Cargar datos del run especificado"""
    run_dir = DATA_DIR / run_folder
    try:
        with open(run_dir / 'metricas_mejoradas.json', 'r') as f:
            metricas = json.load(f)

        pred_total = pd.read_excel(run_dir / 'prediccion_total_mejorada.xlsx', engine='openpyxl')
        pred_total['Fecha'] = pd.to_datetime(pred_total['Fecha'])

        grid_search = pd.read_excel(run_dir / 'grid_search_results.xlsx', engine='openpyxl')

        walk_forward = pd.read_excel(run_dir / 'walk_forward_validation.xlsx', engine='openpyxl')
        walk_forward['fecha'] = pd.to_datetime(walk_forward['fecha'])

        hist_total = pd.read_excel(run_dir / 'historico_total_mejorado.xlsx',
                                   engine='openpyxl', index_col=0)
        hist_total.index = pd.to_datetime(hist_total.index)
        hist_total = hist_total.squeeze()

        return metricas, pred_total, grid_search, walk_forward, hist_total

    except Exception as e:
        st.error(f"❌ Error cargando datos de `{run_folder}`: {str(e)}")
        st.stop()


# ── Selector de versión del modelo ─────────────────────────────────────────
available_runs = get_available_runs()

if not available_runs:
    st.error("❌ No hay modelos entrenados. Ejecuta primero la app de entrenamiento.")
    st.stop()

default_run = get_default_run(available_runs)

selected_run = st.sidebar.selectbox(
    "📦 Versión del modelo",
    options=available_runs,
    index=available_runs.index(default_run),
    format_func=format_run_label,
    help="Selecciona qué ejecución de entrenamiento quieres visualizar."
)

is_latest = (DATA_DIR / 'latest.txt').exists() and \
            (DATA_DIR / 'latest.txt').read_text().strip() == selected_run
st.sidebar.caption(
    f"{'🟢 Activo en producción' if is_latest else '🔵 Versión histórica'}"
)

# Cargar datos del run seleccionado
with st.spinner('Cargando datos...'):
    metricas, pred_total, grid_search, walk_forward, hist_total = load_precargados(selected_run)

# ============================================================================
# DEFINIR TABS SEGÚN ROL
# ============================================================================

# Tabs disponibles por rol
if st.session_state.role == 'admin':
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones", "🔬 ACF/PACF", "🔍 Grid Search", "🔄 Walk-Forward", "📋 Métricas Técnicas"])
elif st.session_state.role == 'analyst':
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones", "🔬 ACF/PACF", "🔍 Grid Search", "🔄 Walk-Forward", "📋 Métricas Técnicas"])
elif st.session_state.role == 'manager':
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones", "💼 Recomendaciones"])
else:  # viewer
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones"])

# ============================================================================
# TAB 1: DASHBOARD (TODOS)
# ============================================================================

with tabs[0]:
    st.header("📊 Dashboard General", divider='blue')
    
    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Ventas", f"{metricas['datos_limpios']['total_ventas']:,}")
    with col2:
        st.metric("Meses de Datos", metricas['datos_limpios']['meses_datos'])
    with col3:
        mape = metricas['walk_forward_validation']['mape']
        st.metric("MAPE", f"{mape:.2f}%", delta=f"{mape - 25.46:.2f}%", delta_color="inverse")
    with col4:
        st.metric("Próximo Mes", f"{int(metricas['predicciones_futuras']['proximo_mes'])}")
    
    # Gráfico histórico
    st.subheader("Serie Temporal Histórica")
    
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Scatter(
        x=hist_total.index,
        y=hist_total.values,
        mode='lines+markers',
        name='Ventas Mensuales',
        line=dict(color='#065A82', width=3),
        marker=dict(size=6)
    ))
    
    # Media
    fig_hist.add_hline(
        y=hist_total.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Media: {hist_total.mean():.1f}",
        annotation_position="right"
    )
    
    fig_hist.update_layout(
        title='Ventas Mensuales TIGGO 2 (2021-2026)',
        xaxis_title='Fecha',
        yaxis_title='Ventas',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
    
    # Estadísticas descriptivas
    if st.session_state.role in ['admin', 'analyst']:
        with st.expander("📊 Estadísticas Descriptivas"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Ventas Mensuales:**")
                st.code(f"""
Promedio:  {hist_total.mean():.1f}
Mediana:   {hist_total.median():.1f}
Mínimo:    {hist_total.min():.0f}
Máximo:    {hist_total.max():.0f}
Desv.Est:  {hist_total.std():.1f}
                """)
            
            with col2:
                st.markdown("**Información del Modelo:**")
                orden = metricas['mejor_modelo']['order']
                orden_est = metricas['mejor_modelo']['seasonal_order']
                st.code(f"""
Modelo: SARIMA{orden}{orden_est}
AIC: {metricas['mejor_modelo']['aic']:.2f}
BIC: {metricas['mejor_modelo']['bic']:.2f}
                """)

# ============================================================================
# TAB 2: PREDICCIONES (TODOS)
# ============================================================================

with tabs[1]:
    st.header("🔮 Predicciones Futuras", divider='green')
    
    # KPIs predicción
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Próximo Mes", f"{pred_total['Predicción'].iloc[0]:.0f} ventas")
    with col2:
        st.metric("Total 6 Meses", f"{pred_total['Predicción'].sum():.0f} ventas")
    with col3:
        st.metric("Promedio Mensual", f"{pred_total['Predicción'].mean():.1f} ventas")
    
    # Gráfico predicción
    fig_pred = go.Figure()
    
    # Histórico
    fig_pred.add_trace(go.Scatter(
        x=hist_total.index,
        y=hist_total.values,
        mode='lines',
        name='Histórico',
        line=dict(color='#065A82', width=2)
    ))
    
    # Predicción
    fig_pred.add_trace(go.Scatter(
        x=pred_total['Fecha'],
        y=pred_total['Predicción'],
        mode='lines+markers',
        name='Predicción',
        line=dict(color='red', width=3),
        marker=dict(size=10, symbol='star')
    ))
    
    # IC 95%
    fig_pred.add_trace(go.Scatter(
        x=pred_total['Fecha'].tolist() + pred_total['Fecha'].tolist()[::-1],
        y=pred_total['IC_Superior'].tolist() + pred_total['IC_Inferior'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.15)',
        line=dict(color='rgba(255,0,0,0)'),
        name='IC 95%'
    ))
    
    # Línea divisoria
    fig_pred.add_shape(
        type="line",
        x0=hist_total.index[-1],
        x1=hist_total.index[-1],
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash")
    )
    
    fig_pred.update_layout(
        title='Histórico + Predicción Futura',
        xaxis_title='Fecha',
        yaxis_title='Ventas',
        template='plotly_white',
        height=600,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_pred, use_container_width=True, config={'displayModeBar': False})
    
    # Tabla predicciones
    st.subheader("📋 Tabla de Predicciones")
    st.dataframe(
        pred_total[['Mes', 'Predicción', 'IC_Inferior', 'IC_Superior']],
        use_container_width=True,
        hide_index=True
    )
    
    # Exportar (solo admin, manager, analyst)
    if has_permission('exportar'):
        csv = pred_total.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exportar CSV",
            data=csv,
            file_name=f"predicciones_tiggo2_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ============================================================================
# TAB 3: RECOMENDACIONES (SOLO MANAGER)
# ============================================================================

if st.session_state.role == 'manager':
    with tabs[2]:
        st.header("💼 Recomendaciones de Compra", divider='orange')
        
        proximo_mes_pred = pred_total['Predicción'].iloc[0]
        ic_inferior = pred_total['IC_Inferior'].iloc[0]
        ic_superior = pred_total['IC_Superior'].iloc[0]
        
        st.markdown(f"""
        ### 📊 Análisis para el próximo mes
        
        **Predicción:** {proximo_mes_pred:.0f} unidades  
        **Rango seguro (IC 95%):** {ic_inferior:.0f} - {ic_superior:.0f} unidades
        """)
        
        # Recomendación de compra
        st.subheader("✅ Recomendación de Compra")
        
        # Estrategia conservadora: IC superior + 10%
        compra_conservadora = ic_superior * 1.1
        
        # Estrategia agresiva: IC superior + 20%
        compra_agresiva = ic_superior * 1.2
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **📉 Estrategia Conservadora:**
            
            Comprar: **{compra_conservadora:.0f} unidades**
            
            - Minimiza riesgo de sobrestock
            - Basado en IC superior + 10%
            - Recomendado si mercado es volátil
            """)
        
        with col2:
            st.success(f"""
            **📈 Estrategia Agresiva:**
            
            Comprar: **{compra_agresiva:.0f} unidades**
            
            - Maximiza cobertura de demanda
            - Basado en IC superior + 20%
            - Recomendado si tendencia es creciente
            """)
        
        # Análisis de tendencia
        st.subheader("📈 Análisis de Tendencia")
        
        # Últimos 3 meses
        ultimos_3 = hist_total.iloc[-3:].mean()
        promedio_historico = hist_total.mean()
        
        tendencia_pct = ((ultimos_3 - promedio_historico) / promedio_historico) * 100
        
        if tendencia_pct > 10:
            st.success(f"✅ **Tendencia CRECIENTE:** Los últimos 3 meses promediaron {ultimos_3:.1f} ventas (+{tendencia_pct:.1f}% vs. histórico). Mercado en crecimiento.")
            st.info("💡 Recomendación: Considerar **estrategia agresiva**")
        elif tendencia_pct < -10:
            st.warning(f"⚠️ **Tendencia DECRECIENTE:** Los últimos 3 meses promediaron {ultimos_3:.1f} ventas ({tendencia_pct:.1f}% vs. histórico). Mercado en contracción.")
            st.info("💡 Recomendación: Considerar **estrategia conservadora**")
        else:
            st.info(f"📊 **Tendencia ESTABLE:** Los últimos 3 meses promediaron {ultimos_3:.1f} ventas ({tendencia_pct:+.1f}% vs. histórico). Mercado estable.")
            st.info("💡 Recomendación: Usar predicción directa ({:.0f} unidades)".format(proximo_mes_pred))
        
        # Alertas
        st.subheader("🚨 Alertas y Avisos")
        
        # Alert si predicción es muy diferente del promedio
        if abs(proximo_mes_pred - promedio_historico) / promedio_historico > 0.3:
            st.warning(f"⚠️ La predicción ({proximo_mes_pred:.0f}) difiere más del 30% del promedio histórico ({promedio_historico:.1f}). Revisar factores externos.")

# ============================================================================
# TABS TÉCNICOS (SOLO ADMIN Y ANALYST)
# ============================================================================

if has_permission('ver_acf_pacf'):
    # TAB: ACF/PACF
    tab_idx = 2 if st.session_state.role in ['admin', 'analyst'] else None
    
    if tab_idx is not None:
        with tabs[tab_idx]:
            st.header("🔬 Análisis ACF/PACF", divider='blue')
            
            st.markdown("""
            **Análisis de Autocorrelación** para selección de parámetros SARIMA.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ACF - Autocorrelación")
                try:
                    image_acf = Image.open(DATA_DIR / 'acf_plot.png')
                    st.image(image_acf, use_container_width=True)
                except:
                    st.warning("Archivo acf_plot.png no disponible")
            
            with col2:
                st.subheader("PACF - Autocorrelación Parcial")
                try:
                    image_pacf = Image.open(DATA_DIR / 'pacf_plot.png')
                    st.image(image_pacf, use_container_width=True)
                except:
                    st.warning("Archivo pacf_plot.png no disponible")

if has_permission('ver_grid_search'):
    # TAB: Grid Search
    tab_idx = 3 if st.session_state.role in ['admin', 'analyst'] else None
    
    if tab_idx is not None:
        with tabs[tab_idx]:
            st.header("🔍 Grid Search de Parámetros", divider='blue')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Combinaciones Evaluadas", len(grid_search))
            with col2:
                best_aic = grid_search['aic'].min()
                st.metric("Mejor AIC", f"{best_aic:.2f}")
            with col3:
                best_mape = grid_search.loc[grid_search['aic'].idxmin(), 'mape']
                st.metric("MAPE (mejor modelo)", f"{best_mape:.2f}%")
            
            st.subheader("Top 10 Modelos por AIC")
            
            top10 = grid_search.nsmallest(10, 'aic')[['p', 'd', 'q', 'P', 'D', 'Q', 'aic', 'bic', 'mape', 'mae', 'rmse']]
            
            st.dataframe(
                top10.style.background_gradient(subset=['aic'], cmap='Greens_r')
                      .background_gradient(subset=['mape'], cmap='RdYlGn_r')
                      .format({'aic': '{:.2f}', 'bic': '{:.2f}', 'mape': '{:.2f}%', 'mae': '{:.2f}', 'rmse': '{:.2f}'}),
                use_container_width=True,
                hide_index=True
            )
            
            # Gráfico
            fig_grid = px.scatter(
                grid_search,
                x='aic',
                y='mape',
                color='p',
                size='mae',
                hover_data=['p', 'd', 'q', 'P', 'D', 'Q'],
                title='Grid Search: AIC vs MAPE'
            )
            
            fig_grid.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig_grid, use_container_width=True, config={'displayModeBar': False})

    # TAB: Walk-Forward
    tab_idx = 4 if st.session_state.role in ['admin', 'analyst'] else None
    
    if tab_idx is not None:
        with tabs[tab_idx]:
            st.header("🔄 Walk-Forward Validation", divider='blue')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MAPE Promedio", f"{walk_forward['error_pct'].mean():.2f}%")
            with col2:
                st.metric("Mejor Mes", f"{walk_forward['error_pct'].min():.2f}%")
            with col3:
                st.metric("Peor Mes", f"{walk_forward['error_pct'].max():.2f}%")
            with col4:
                st.metric("Meses Evaluados", len(walk_forward))
            
            # Gráfico
            fig_wf = go.Figure()
            
            fig_wf.add_trace(go.Scatter(
                x=walk_forward['fecha'],
                y=walk_forward['real'],
                mode='lines+markers',
                name='Real',
                line=dict(color='green', width=3)
            ))
            
            fig_wf.add_trace(go.Scatter(
                x=walk_forward['fecha'],
                y=walk_forward['prediccion'],
                mode='lines+markers',
                name='Predicción',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            fig_wf.update_layout(
                title='Walk-Forward: Real vs Predicción',
                xaxis_title='Fecha',
                yaxis_title='Ventas',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_wf, use_container_width=True, config={'displayModeBar': False})
            
            # Tabla
            wf_display = walk_forward.copy()
            wf_display['fecha'] = wf_display['fecha'].dt.strftime('%B %Y')
            wf_display.columns = ['Mes', 'Real', 'Predicción', 'Error Abs', 'Error %']
            
            st.dataframe(
                wf_display.style.background_gradient(subset=['Error %'], cmap='RdYlGn_r')
                         .format({'Real': '{:.0f}', 'Predicción': '{:.1f}', 'Error Abs': '{:.2f}', 'Error %': '{:.2f}%'}),
                use_container_width=True,
                hide_index=True
            )

    # TAB: Métricas Técnicas
    tab_idx = 5 if st.session_state.role in ['admin', 'analyst'] else None
    
    if tab_idx is not None:
        with tabs[tab_idx]:
            st.header("📋 Métricas Técnicas Completas", divider='gray')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Parámetros SARIMA:**")
                orden = metricas['mejor_modelo']['order']
                orden_est = metricas['mejor_modelo']['seasonal_order']
                st.code(f"""
order = ({orden[0]}, {orden[1]}, {orden[2]})
seasonal_order = ({orden_est[0]}, {orden_est[1]}, {orden_est[2]}, 12)

p = {orden[0]} (autoregresivo)
d = {orden[1]} (diferenciación)
q = {orden[2]} (media móvil)
P = {orden_est[0]} (AR estacional)
D = {orden_est[1]} (dif estacional)
Q = {orden_est[2]} (MA estacional)
m = 12 (estacionalidad mensual)
                """)
            
            with col2:
                st.markdown("**Métricas de Ajuste:**")
                st.code(f"""
AIC: {metricas['mejor_modelo']['aic']:.2f}
BIC: {metricas['mejor_modelo']['bic']:.2f}

MAPE (walk-forward): {metricas['walk_forward_validation']['mape']:.2f}%

Dataset:
- Ventas: {metricas['datos_limpios']['total_ventas']:,}
- Meses: {metricas['datos_limpios']['meses_datos']}
- Período: {metricas['datos_limpios']['periodo']}
                """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Dashboard TIGGO 2 - Modelo Mejorado</strong></p>
    <p>Interamericana Norte Perú | Equipo Mira Murati - ISDI Master 2026</p>
</div>
""", unsafe_allow_html=True)
