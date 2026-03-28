"""
============================================================================
MÓDULO: SISTEMA DE AUTENTICACIÓN
Para App Entrenamiento y Dashboard Negocio
============================================================================
"""

import streamlit as st
import hashlib
import time
from datetime import datetime, timedelta

from styles import get_global_css, get_login_css

LOGO_URL = "https://cdn.brandfetch.io/idbC6t7DJN/w/904/h/196/theme/light/logo.png?c=1bxid64Mup7aczewSAYMX&t=1766585238441"

# ============================================================================
# CONFIGURACIÓN DE USUARIOS Y ROLES
# Cargado desde .streamlit/secrets.toml (nunca en el código fuente)
# Ver secrets.toml.example para la estructura esperada
# ============================================================================

USERS_CONFIG = st.secrets["users"]

SESSION_TIMEOUT = 30  # minutos
MAX_LOGIN_ATTEMPTS = 3

# ============================================================================
# FUNCIONES DE AUTENTICACIÓN
# ============================================================================

def hash_password(password):
    """Hash SHA256 de contraseña"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_credentials(username, password):
    """Verificar credenciales (texto plano o hash)"""
    if username not in USERS_CONFIG:
        return False
    
    stored_pass = USERS_CONFIG[username]['password']
    
    # Verificar texto plano
    if password == stored_pass:
        return True
    
    # Verificar hash SHA256
    if hash_password(password) == stored_pass:
        return True
    
    return False

def init_session_state():
    """Inicializar session state"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0
    if 'permissions' not in st.session_state:
        st.session_state.permissions = {}

def check_session_timeout():
    """Verificar si la sesión ha expirado"""
    if st.session_state.authenticated and st.session_state.login_time:
        elapsed = datetime.now() - st.session_state.login_time
        if elapsed > timedelta(minutes=SESSION_TIMEOUT):
            logout()
            return True
    return False

def login(username, password):
    """Intentar login"""
    if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
        return False, "Demasiados intentos fallidos. Espera 5 minutos."
    
    if verify_credentials(username, password):
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.role = USERS_CONFIG[username]['role']
        st.session_state.login_time = datetime.now()
        st.session_state.login_attempts = 0
        st.session_state.permissions = USERS_CONFIG[username]['permissions']
        st.session_state.user_name = USERS_CONFIG[username]['name']
        st.session_state.user_icon = USERS_CONFIG[username]['icon']
        return True, "Login exitoso"
    else:
        st.session_state.login_attempts += 1
        remaining = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
        return False, f"Credenciales incorrectas. Intentos restantes: {remaining}"

def logout():
    """Cerrar sesión"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.login_time = None
    st.session_state.permissions = {}

def show_login_page(app_title="Sistema TIGGO 2"):
    """Mostrar página de login corporativa — dark premium."""
    st.markdown(get_login_css(), unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])

    with col:
        st.markdown(f"""
        <div class="login-card">
          <img src="{LOGO_URL}" class="login-logo">
          <div class="login-title">{app_title}</div>
          <div class="login-subtitle">Sistema de Predicción de Demanda</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Usuario", placeholder="Ingresa tu usuario")
            password = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña")
            submit   = st.form_submit_button("Ingresar →", use_container_width=True)

            if submit:
                if username and password:
                    success, message = login(username, password)
                    if success:
                        st.success("Acceso concedido")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Por favor completa todos los campos")

        st.markdown(
            '<p class="login-footer-txt">Acceso restringido · Sistema TIGGO 2 · ISDI</p>',
            unsafe_allow_html=True,
        )

def show_user_info():
    """Mostrar info del usuario logueado en el sidebar — diseño premium."""
    remaining = SESSION_TIMEOUT
    if st.session_state.login_time:
        elapsed   = datetime.now() - st.session_state.login_time
        remaining = max(0, SESSION_TIMEOUT - int(elapsed.total_seconds() / 60))

    role_badges = {
        'admin':   '<span class="role-badge admin-badge">👑 Admin</span>',
        'manager': '<span class="role-badge manager-badge">💼 Gerente</span>',
        'analyst': '<span class="role-badge analyst-badge">📊 Analista</span>',
        'viewer':  '<span class="role-badge viewer-badge">👁 Viewer</span>',
    }
    badge = role_badges.get(st.session_state.role, '')

    st.sidebar.markdown(f"""
<div class="user-info-card">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
    <span style="font-size:1.55rem;line-height:1">{st.session_state.user_icon}</span>
    <div>
      <div class="user-name">{st.session_state.user_name}</div>
      <div class="user-handle">@{st.session_state.username}</div>
    </div>
  </div>
  {badge}
  <div class="session-timer">⏱ Sesión expira en {remaining} min</div>
</div>
""", unsafe_allow_html=True)

    if st.sidebar.button("Cerrar Sesión", use_container_width=True):
        logout()
        st.rerun()

def require_permission(permission_name):
    """Decorator para requerir permisos específicos"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not st.session_state.get('authenticated', False):
                st.error("❌ No estás autenticado")
                return None
            
            if not st.session_state.permissions.get(permission_name, False):
                st.error(f"❌ No tienes permiso para: {permission_name}")
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def has_permission(permission_name):
    """Verificar si el usuario tiene un permiso"""
    if not st.session_state.get('authenticated', False):
        return False
    return st.session_state.permissions.get(permission_name, False)

def show_header(title, subtitle=""):
    """Header corporativo premium — inyecta el CSS global y muestra logo + título."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    sub_html = f'<div class="header-sub">{subtitle}</div>' if subtitle else ''
    st.markdown(f"""
<div class="page-header">
  <img src="{LOGO_URL}">
  <div class="header-divider"></div>
  <div class="header-text">
    <h1>{title}</h1>
    {sub_html}
  </div>
</div>
""", unsafe_allow_html=True)
