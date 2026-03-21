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
    """Mostrar página de login corporativa"""
    
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 40px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .login-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .login-logo {
            width: 200px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Logo
        st.markdown(f"""
            <div class="login-header">
                <img src="https://cdn.brandfetch.io/idbC6t7DJN/w/904/h/196/theme/dark/logo.png" 
                     class="login-logo">
                <h2 style="color: #003B5C;">{app_title}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Formulario
        with st.form("login_form"):
            username = st.text_input("👤 Usuario", placeholder="Ingrese su usuario")
            password = st.text_input("🔒 Contraseña", type="password", placeholder="Ingrese su contraseña")
            submit = st.form_submit_button("🚀 Ingresar", use_container_width=True)
            
            if submit:
                if username and password:
                    success, message = login(username, password)
                    
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Por favor complete todos los campos")
        
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_user_info():
    """Mostrar info del usuario logueado en sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
        ### {st.session_state.user_icon} Usuario
        **{st.session_state.user_name}**  
        `{st.session_state.username}`
        
        **Rol:** {st.session_state.role}
    """)
    
    # Tiempo de sesión
    if st.session_state.login_time:
        elapsed = datetime.now() - st.session_state.login_time
        remaining = SESSION_TIMEOUT - int(elapsed.total_seconds() / 60)
        st.sidebar.info(f"⏱️ Sesión expira en: {remaining} min")
    
    if st.sidebar.button("🚪 Cerrar Sesión", use_container_width=True):
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
