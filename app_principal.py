"""
============================================================================
APP PRINCIPAL — Entry point para Streamlit Cloud
Gestiona la autenticación compartida y muestra la página de inicio.
============================================================================
"""

import streamlit as st
from core.auth_system import init_session_state, show_login_page, check_session_timeout, show_user_info, show_header
import core.supabase_io as sio
from core.styles import get_home_css

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
st.markdown(get_home_css(), unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def _get_mape_activo() -> str:
    try:
        m = sio.load_current_model()
        if m:
            val = m["walk_forward_validation"]["mape"]
            return f"{val:.2f} %"
    except Exception:
        pass
    return "— %"


def _render_hero() -> None:
    st.markdown("""
<div class="home-hero">
  <div class="hero-meta-row">
    <span class="hero-meta-tag active">INTERAMERICANA / NORTE</span>
    <span class="hero-meta-sep">·</span>
    <span class="hero-meta-tag">FILE&nbsp;&nbsp;2025 / SAA</span>
    <span class="hero-meta-sep">·</span>
    <span class="hero-meta-tag">CHERY TIGGO 2</span>
  </div>
  <h2 class="home-hero-title">
    Vender lo que de verdad <span class="hero-accent">se vende.</span>
  </h2>
  <p class="home-hero-sub">
    Solución Analítica Avanzada de Predicción de Demanda&nbsp;&nbsp;·&nbsp;&nbsp;Norte Perú&nbsp;&nbsp;·&nbsp;&nbsp;SARIMA (1,1,0)(1,0,2)[12]
  </p>
</div>
""", unsafe_allow_html=True)


def _render_context() -> None:
    st.markdown("""
<div style="margin-bottom:28px;padding:18px 24px;
            background:linear-gradient(135deg,rgba(0,115,255,0.06),rgba(194,255,0,0.04));
            border:1px solid rgba(0,115,255,0.14);border-radius:8px;position:relative;">
  <div style="font-family:'Rajdhani',sans-serif;font-size:.6rem;font-weight:700;
              letter-spacing:.2em;text-transform:uppercase;color:#3F5060;margin-bottom:10px;">
    CONTEXTO DE SELECCIÓN
  </div>
  <div style="display:flex;flex-direction:column;gap:20px;">
    <div>
      <span style="font-family:'Rajdhani',sans-serif;font-size:1.02rem;font-weight:700;
                   color:#C9D8E6;">¿Por qué Chery Tiggo&nbsp;2?</span>
      <p style="color:#7A95A8;font-size:.84rem;line-height:1.75;margin-top:6px;">
        Al analizar el portafolio histórico de Interamericana Norte, el Tiggo&nbsp;2 es el modelo
        con <strong style="color:#C2FF00;">mayor demanda sostenida</strong> en el segmento SUV
        compacto y el <strong style="color:#C2FF00;">historial más largo disponible</strong>
        (51+ meses, 3+ ciclos estacionales completos) — condición necesaria para entrenar SARIMA.
      </p>
    </div>
    <div class="contexto-kpis">
      <div style="text-align:center;">
        <div style="font-family:'Rajdhani',sans-serif;font-size:2.8rem;font-weight:700;color:#0073FF;line-height:1;">2,047</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#3F5060;text-transform:uppercase;letter-spacing:.12em;margin-top:3px;">unidades históricas</div>
      </div>
      <div style="text-align:center;">
        <div style="font-family:'Rajdhani',sans-serif;font-size:2.8rem;font-weight:700;color:#C2FF00;line-height:1;">51+</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#3F5060;text-transform:uppercase;letter-spacing:.12em;margin-top:3px;">meses de datos</div>
      </div>
      <div style="text-align:center;">
        <div style="font-family:'Rajdhani',sans-serif;font-size:2.8rem;font-weight:700;color:#00F5A0;line-height:1;">#1</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#3F5060;text-transform:uppercase;letter-spacing:.12em;margin-top:3px;">modelo Chery en volumen</div>
      </div>
      <div style="text-align:center;">
        <div style="font-family:'Rajdhani',sans-serif;font-size:2.8rem;font-weight:700;color:#A78BFA;line-height:1;">Piloto</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#3F5060;text-transform:uppercase;letter-spacing:.12em;margin-top:3px;">→ escalable a todo el portafolio</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


def _feature_card(num, color, title, desc, kpi_value, kpi_value_class, kpi_label, badge_class, badge_text) -> str:
    kpi_cls = f"card-kpi-value {kpi_value_class}" if kpi_value_class else "card-kpi-value"
    return f"""
<div class="feature-card {color}">
  <span class="fc-number">{num}</span>
  <h3>{title}</h3>
  <p>{desc}</p>
  <div class="card-kpi">
    <div class="{kpi_cls}">{kpi_value}</div>
    <div class="card-kpi-label">{kpi_label}</div>
  </div>
  <span class="feature-card-badge {badge_class}">{badge_text}</span>
</div>"""


# ── Datos de cards ────────────────────────────────────────────────────────────

_mape_display = _get_mape_activo()

_CARDS = [
    {
        "num": "01 / ENTRENAMIENTO",  "color": "blue",  "title": "Entrenamiento",
        "desc": "Carga datos de ventas, entrena un nuevo modelo SARIMA con búsqueda bayesiana (Optuna) y publícalo en el Dashboard.",
        "kpi_value": "SARIMAx",       "kpi_value_class": "",       "kpi_label": "Modelo activo · Optuna TPE",
        "badge_class": "badge-tech",  "badge_text": "Admin · Analista",
    },
    {
        "num": "02 / DASHBOARD",      "color": "green", "title": "Dashboard",
        "desc": "Visualiza predicciones, KPIs y métricas del modelo activo. Cambia entre versiones históricas desde el panel lateral.",
        "kpi_value": _mape_display,   "kpi_value_class": "lime",   "kpi_label": "MAPE · walk-forward",
        "badge_class": "badge-all",   "badge_text": "Todos los roles",
    },
    {
        "num": "03 / COMPARATIVA",    "color": "amber", "title": "Comparativa ML",
        "desc": "Enfrenta SARIMA, Prophet, Regresión Lineal, Random Forest y XGBoost para encontrar el mejor predictor mensual del Tiggo 2.",
        "kpi_value": "5 modelos",     "kpi_value_class": "violet", "kpi_label": "SARIMA · Prophet · RF · XGB · LR",
        "badge_class": "badge-tech",  "badge_text": "Admin · Analista",
    },
    {
        "num": "04 / CONCESIONARIOS", "color": "blue",  "title": "Concesionarios",
        "desc": "Analiza ventas históricas y predicciones distribuidas por tienda. Simula escenarios de apertura, cierre y campañas locales.",
        "kpi_value": "Shares",        "kpi_value_class": "",       "kpi_label": "Distribución por tienda · IC 95%",
        "badge_class": "badge-tech",  "badge_text": "Admin · Analista · Manager",
    },
    {
        "num": "05 / PROYECCIÓN",     "color": "green", "title": "Proyección Ingresos",
        "desc": "Traduce la predicción SARIMA en cifras financieras en USD. Configura precio, margen neto y tipo de cambio.",
        "kpi_value": "USD",           "kpi_value_class": "lime",   "kpi_label": "Ingresos proyectados · IC 95%",
        "badge_class": "badge-tech",  "badge_text": "Admin · Analista · Financiero",
    },
    {
        "num": "06 / ESCALABILIDAD",  "color": "amber", "title": "Escalabilidad",
        "desc": "Hoja de ruta para exportar el pipeline a otras marcas, líneas de negocio y mercados geográficos de LatAm.",
        "kpi_value": "Multi-Marca",   "kpi_value_class": "violet", "kpi_label": "Portafolio · Onboarding · LatAm",
        "badge_class": "badge-all",   "badge_text": "Todos los roles",
    },
]

# ── Render ────────────────────────────────────────────────────────────────────

_render_hero()
_render_context()

for i in range(0, len(_CARDS), 3):
    cols = st.columns(3)
    for col, card in zip(cols, _CARDS[i : i + 3]):
        with col:
            st.markdown(_feature_card(**card), unsafe_allow_html=True)
    if i + 3 < len(_CARDS):
        st.markdown('<div style="margin-top:28px;"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="
  margin-top: 48px; padding-top: 16px;
  border-top: 1px solid rgba(0,115,255,0.06);
  display: flex; justify-content: space-between; align-items: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: .55rem; letter-spacing: .16em;
  color: var(--c-muted); text-transform: uppercase; opacity: .6;
">
  <span>EQUIPO&nbsp;&nbsp;Maria · Cristina · Ingrid · Juan · Diego</span>
  <span>PROG&nbsp;&nbsp;ISDI 2025 2026</span>
  <span>CLNT&nbsp;&nbsp;Interamericana Norte S.A.C.</span>
</div>
""", unsafe_allow_html=True)
