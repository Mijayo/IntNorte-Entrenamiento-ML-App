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

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    ### 🤖 Entrenamiento
    Carga datos de ventas, entrena un nuevo modelo SARIMA
    y publícalo en el Dashboard con un clic.

    > Solo disponible para **Administradores** y **Analistas**.
    """)

with col2:
    st.success("""
    ### 🚗 Dashboard
    Visualiza predicciones, KPIs y métricas del modelo activo.
    Cambia entre versiones históricas desde el panel lateral.

    > Disponible para **todos los roles**.
    """)

with col3:
    st.warning("""
    ### ⚔️ Prophet vs SARIMA
    Compara el rendimiento de Prophet (Meta) contra SARIMA
    sobre el mismo histórico de ventas.

    > Solo disponible para **Administradores** y **Analistas**.
    """)
