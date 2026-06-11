"""
============================================================================
APP PRINCIPAL — Entry point para Streamlit Cloud
Gestiona la autenticación compartida y muestra la página de inicio.
============================================================================
"""

from dataclasses import dataclass

import streamlit as st

import core.supabase_io as sio
from core.auth_system import guard_page, show_user_info, show_header
from core.logger import get_logger
from core.styles import get_home_css

log = get_logger("home")

st.set_page_config(
    page_title="Sistema TIGGO 2",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

guard_page("🚗 Sistema TIGGO 2")

show_header("Sistema de Predicción TIGGO 2", "Selecciona una aplicación en el menú lateral izquierdo.")
show_user_info()
st.markdown(get_home_css(), unsafe_allow_html=True)


# ── Datos del modelo activo ───────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _get_model_info() -> tuple[str, str]:
    """Returns (mape_display, sarima_order_display) from the active model."""
    try:
        m = sio.load_current_model()
        if m:
            mape = f"{m['walk_forward_validation']['mape']:.2f} %"
            best = m.get("mejor_modelo", {})
            order = best.get("order")
            seasonal = best.get("seasonal_order")
            if order and seasonal and len(order) == 3 and len(seasonal) == 4:
                p, d, q = order
                P, D, Q, s = seasonal
                order_str = f"SARIMA ({p},{d},{q})({P},{D},{Q})[{s}]"
            else:
                order_str = "SARIMA"
            return mape, order_str
    except Exception:
        log.warning("No se pudo cargar info del modelo activo")
    return "— %", "SARIMA"


# ── Feature cards ─────────────────────────────────────────────────────────────

@dataclass
class FeatureCard:
    num: str
    color: str
    title: str
    desc: str
    kpi_value: str
    kpi_value_class: str
    kpi_label: str
    badge_class: str
    badge_text: str


def _render_card(card: FeatureCard) -> str:
    kpi_cls = f"card-kpi-value {card.kpi_value_class}" if card.kpi_value_class else "card-kpi-value"
    return f"""
<div class="feature-card {card.color}">
  <span class="fc-number">{card.num}</span>
  <h3>{card.title}</h3>
  <p>{card.desc}</p>
  <div class="card-kpi">
    <div class="{kpi_cls}">{card.kpi_value}</div>
    <div class="card-kpi-label">{card.kpi_label}</div>
  </div>
  <span class="feature-card-badge {card.badge_class}">{card.badge_text}</span>
</div>"""


# ── Secciones de página ───────────────────────────────────────────────────────

def _render_hero(order_str: str) -> None:
    st.markdown(f"""
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
    Solución Analítica Avanzada de Predicción de Demanda&nbsp;&nbsp;·&nbsp;&nbsp;Norte Perú&nbsp;&nbsp;·&nbsp;&nbsp;{order_str}
  </p>
</div>
""", unsafe_allow_html=True)


def _render_context() -> None:
    st.markdown("""
<div class="context-panel">
  <div class="context-panel-label">CONTEXTO DE SELECCIÓN</div>
  <div class="context-panel-body">
    <div>
      <span class="context-title">¿Por qué Chery Tiggo&nbsp;2?</span>
      <p class="context-text">
        Al analizar el portafolio histórico de Interamericana Norte, el Tiggo&nbsp;2 es el modelo
        con <strong>mayor demanda sostenida</strong> en el segmento SUV
        compacto y el <strong>historial más largo disponible</strong>
        (51+ meses, 3+ ciclos estacionales completos) — condición necesaria para entrenar SARIMA.
      </p>
    </div>
    <div class="contexto-kpis">
      <div>
        <div class="ctx-kpi-value blue">2,047</div>
        <div class="ctx-kpi-sub">unidades históricas</div>
      </div>
      <div>
        <div class="ctx-kpi-value lime">51+</div>
        <div class="ctx-kpi-sub">meses de datos</div>
      </div>
      <div>
        <div class="ctx-kpi-value green">#1</div>
        <div class="ctx-kpi-sub">modelo Chery en volumen</div>
      </div>
      <div>
        <div class="ctx-kpi-value violet">Piloto</div>
        <div class="ctx-kpi-sub">→ escalable a todo el portafolio</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Render ────────────────────────────────────────────────────────────────────

_mape_display, _order_str = _get_model_info()

_CARDS = [
    FeatureCard(
        num="01 / ENTRENAMIENTO", color="blue",  title="Entrenamiento",
        desc="Carga datos de ventas, entrena un nuevo modelo SARIMA con búsqueda bayesiana (Optuna) y publícalo en el Dashboard.",
        kpi_value="SARIMAx",      kpi_value_class="",       kpi_label="Modelo activo · Optuna TPE",
        badge_class="badge-tech", badge_text="Admin · Analista",
    ),
    FeatureCard(
        num="02 / DASHBOARD",     color="green", title="Dashboard",
        desc="Visualiza predicciones, KPIs y métricas del modelo activo. Cambia entre versiones históricas desde el panel lateral.",
        kpi_value=_mape_display,  kpi_value_class="lime",   kpi_label="MAPE · walk-forward",
        badge_class="badge-all",  badge_text="Todos los roles",
    ),
    FeatureCard(
        num="03 / COMPARATIVA",   color="amber", title="Comparativa ML",
        desc="Enfrenta SARIMA, Prophet, Regresión Lineal, Random Forest y XGBoost para encontrar el mejor predictor mensual del Tiggo 2.",
        kpi_value="5 modelos",    kpi_value_class="violet", kpi_label="SARIMA · Prophet · RF · XGB · LR",
        badge_class="badge-tech", badge_text="Admin · Analista",
    ),
    FeatureCard(
        num="04 / CONCESIONARIOS", color="blue", title="Concesionarios",
        desc="Analiza ventas históricas y predicciones distribuidas por tienda. Simula escenarios de apertura, cierre y campañas locales.",
        kpi_value="Shares",       kpi_value_class="",       kpi_label="Distribución por tienda · IC 95%",
        badge_class="badge-tech", badge_text="Admin · Analista · Manager",
    ),
    FeatureCard(
        num="05 / PROYECCIÓN",    color="green", title="Proyección Ingresos",
        desc="Traduce la predicción SARIMA en cifras financieras en USD. Configura precio, margen neto y tipo de cambio.",
        kpi_value="USD",          kpi_value_class="lime",   kpi_label="Ingresos proyectados · IC 95%",
        badge_class="badge-tech", badge_text="Admin · Analista · Financiero",
    ),
    FeatureCard(
        num="06 / ESCALABILIDAD", color="amber", title="Escalabilidad",
        desc="Hoja de ruta para exportar el pipeline a otras marcas, líneas de negocio y mercados geográficos de LatAm.",
        kpi_value="Multi-Marca",  kpi_value_class="violet", kpi_label="Portafolio · Onboarding · LatAm",
        badge_class="badge-all",  badge_text="Todos los roles",
    ),
]

_render_hero(_order_str)
_render_context()

_cards_html = "".join(_render_card(c) for c in _CARDS)
st.markdown(f'<div class="cards-grid">{_cards_html}</div>', unsafe_allow_html=True)

st.markdown("""
<div class="home-footer">
  <span>EQUIPO&nbsp;&nbsp;Maria · Cristina · Ingrid · Juan · Diego</span>
  <span>PROG&nbsp;&nbsp;ISDI 2025 2026</span>
  <span>CLNT&nbsp;&nbsp;Interamericana Norte S.A.C.</span>
</div>
""", unsafe_allow_html=True)
