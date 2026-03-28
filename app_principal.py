"""
============================================================================
APP PRINCIPAL — Entry point para Streamlit Cloud
Gestiona la autenticación compartida y muestra la página de inicio.
============================================================================
"""

import streamlit as st
from auth_system import init_session_state, show_login_page, check_session_timeout, show_user_info, show_header

st.set_page_config(
    page_title="Sistema TIGGO 2",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()

if check_session_timeout():
    st.warning("⏱️ Tu sesión ha expirado. Por favor inicia sesión nuevamente.")
    st.stop()

if not st.session_state.authenticated:
    show_login_page("🚗 Sistema TIGGO 2")
    st.stop()

# ── Página de inicio ─────────────────────────────────────────────────────────

show_header("Sistema de Predicción TIGGO 2", "Selecciona una aplicación en el menú lateral izquierdo.")

show_user_info()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
<div class="feature-card blue">
  <span class="feature-card-icon">🤖</span>
  <h3>Entrenamiento</h3>
  <p>Carga datos de ventas, entrena un nuevo modelo SARIMA con búsqueda bayesiana
     (Optuna) y publícalo en el Dashboard con un clic.</p>
  <span class="feature-card-badge badge-tech">Admin · Analista</span>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div class="feature-card green">
  <span class="feature-card-icon">📊</span>
  <h3>Dashboard</h3>
  <p>Visualiza predicciones, KPIs y métricas del modelo activo.
     Cambia entre versiones históricas desde el panel lateral.</p>
  <span class="feature-card-badge badge-all">Todos los roles</span>
</div>
""", unsafe_allow_html=True)

with col3:
    st.markdown("""
<div class="feature-card amber">
  <span class="feature-card-icon">🏆</span>
  <h3>Comparativa ML</h3>
  <p>Enfrenta SARIMA, Prophet, Regresión Lineal, Random Forest y XGBoost
     para encontrar el mejor predictor mensual del Tiggo 2.</p>
  <span class="feature-card-badge badge-tech">Admin · Analista</span>
</div>
""", unsafe_allow_html=True)
