"""
============================================================================
APP PRINCIPAL — Entry point para Streamlit Cloud
Gestiona la autenticación compartida y muestra la página de inicio.
============================================================================
"""

import streamlit as st
from core.auth_system import init_session_state, show_login_page, check_session_timeout, show_user_info, show_header
import core.supabase_io as sio

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

_mape_display = _get_mape_activo()

# ── Hero section — línea gráfica slides ──────────────────────────────────────

st.markdown("""
<style>
.home-hero {
  padding: 44px 0 40px;
  margin-bottom: 8px;
  border-bottom: 1px solid rgba(0,115,255,0.10);
  position: relative;
}
.home-hero::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, rgba(0,115,255,0.5), rgba(194,255,0,0.3), transparent);
}
.hero-meta-row {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 22px;
  font-family: 'JetBrains Mono', monospace;
  font-size: .58rem; letter-spacing: .18em;
  color: var(--c-muted); text-transform: uppercase;
}
.hero-meta-sep { opacity: .3; }
.hero-meta-tag { opacity: .7; }
.hero-meta-tag.active { color: var(--c-cyan); opacity: 1; }
.home-hero-title {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: clamp(2.4rem, 4vw, 3.8rem) !important;
  font-weight: 700 !important;
  color: var(--c-text) !important;
  line-height: 1.08 !important;
  letter-spacing: .02em !important;
  text-transform: uppercase !important;
  margin: 0 0 20px !important;
  padding: 0 !important;
}
.hero-accent { color: var(--c-cyan); }
.home-hero-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: .72rem; color: var(--c-muted);
  letter-spacing: .06em; margin-bottom: 0;
}
/* ── Feature cards rediseño ── */
.feature-card {
  background: var(--c-surface) !important;
  border: 1px solid rgba(0,115,255,0.08) !important;
  position: relative !important;
  padding: 32px 28px 28px !important;
  transition: border-color .25s, transform .2s !important;
}
.feature-card:hover {
  border-color: rgba(0,115,255,.22) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(0,0,0,.55), 0 0 24px rgba(0,115,255,.08) !important;
}
.feature-card .fc-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: .55rem; letter-spacing: .2em;
  text-transform: uppercase; color: var(--c-muted);
  margin-bottom: 18px; display: block;
}
.feature-card h3 {
  font-size: 1.35rem !important;
}
.feature-card p {
  font-size: .84rem; line-height: 1.75;
}
/* Card accent bars */
.feature-card.blue::before  { background: linear-gradient(90deg, #0073FF, rgba(0,115,255,.15)); box-shadow: 0 1px 10px rgba(0,115,255,.2); }
.feature-card.green::before { background: linear-gradient(90deg, #C2FF00, rgba(194,255,0,.15)); box-shadow: 0 1px 10px rgba(194,255,0,.15); }
.feature-card.amber::before { background: linear-gradient(90deg, #A78BFA, rgba(167,139,250,.15)); box-shadow: 0 1px 10px rgba(167,139,250,.15); }
/* KPI badge inside card */
.card-kpi {
  margin-top: 18px; padding-top: 16px;
  border-top: 1px solid rgba(0,115,255,0.07);
  font-family: 'JetBrains Mono', monospace;
}
.card-kpi-value {
  font-size: 1.45rem; font-weight: 400;
  color: var(--c-cyan); line-height: 1;
  text-shadow: 0 0 18px rgba(0,115,255,.3);
}
.card-kpi-value.lime  { color: var(--c-gold); text-shadow: 0 0 18px rgba(194,255,0,.25); }
.card-kpi-value.violet { color: var(--c-purple); text-shadow: 0 0 18px rgba(167,139,250,.2); }
.card-kpi-label {
  font-size: .55rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--c-muted);
  margin-top: 4px;
}
</style>

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

# ── Contexto de mercado — por qué Chery Tiggo 2 ──────────────────────────────

st.markdown("""
<div style="margin-bottom:28px;padding:18px 24px;
            background:linear-gradient(135deg,rgba(0,115,255,0.06),rgba(194,255,0,0.04));
            border:1px solid rgba(0,115,255,0.14);border-radius:8px;position:relative;">
<div style="font-family:'Rajdhani',sans-serif;font-size:.6rem;font-weight:700;
            letter-spacing:.2em;text-transform:uppercase;color:#3F5060;margin-bottom:10px;">
CONTEXTO DE SELECCIÓN
</div>
<div style="display:flex;gap:40px;align-items:flex-start;flex-wrap:wrap;">
  <div style="flex:1;min-width:220px;">
    <span style="font-family:'Rajdhani',sans-serif;font-size:1.02rem;font-weight:700;
                 color:#C9D8E6;">¿Por qué Chery Tiggo&nbsp;2?</span>
    <p style="color:#7A95A8;font-size:.84rem;line-height:1.75;margin-top:6px;">
      Al analizar el portafolio histórico de Interamericana Norte, el Tiggo&nbsp;2 es el modelo
      con <strong style="color:#C2FF00;">mayor demanda sostenida</strong> en el segmento SUV
      compacto y el <strong style="color:#C2FF00;">historial más largo disponible</strong>
      (51+ meses, 3+ ciclos estacionales completos) — condición necesaria para entrenar SARIMA.
    </p>
  </div>
  <div style="display:flex;gap:32px;flex-wrap:wrap;align-items:center;">
    <div style="text-align:center;">
      <div style="font-family:'Rajdhani',sans-serif;font-size:1.9rem;font-weight:700;
                  color:#0073FF;line-height:1;">2,047</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;
                  color:#3F5060;text-transform:uppercase;letter-spacing:.12em;margin-top:3px;">
        unidades históricas
      </div>
    </div>
    <div style="text-align:center;">
      <div style="font-family:'Rajdhani',sans-serif;font-size:1.9rem;font-weight:700;
                  color:#C2FF00;line-height:1;">51+</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;
                  color:#3F5060;text-transform:uppercase;letter-spacing:.12em;margin-top:3px;">
        meses de datos
      </div>
    </div>
    <div style="text-align:center;">
      <div style="font-family:'Rajdhani',sans-serif;font-size:1.9rem;font-weight:700;
                  color:#00F5A0;line-height:1;">#1</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;
                  color:#3F5060;text-transform:uppercase;letter-spacing:.12em;margin-top:3px;">
        modelo Chery en volumen
      </div>
    </div>
    <div style="text-align:center;">
      <div style="font-family:'Rajdhani',sans-serif;font-size:1.9rem;font-weight:700;
                  color:#A78BFA;line-height:1;">Piloto</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;
                  color:#3F5060;text-transform:uppercase;letter-spacing:.12em;margin-top:3px;">
        → escalable a todo el portafolio
      </div>
    </div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
<div class="feature-card blue">
  <span class="fc-number">01 / ENTRENAMIENTO</span>
  <h3>Entrenamiento</h3>
  <p>Carga datos de ventas, entrena un nuevo modelo SARIMA con búsqueda bayesiana
     (Optuna) y publícalo en el Dashboard.</p>
  <div class="card-kpi">
    <div class="card-kpi-value">SARIMAx</div>
    <div class="card-kpi-label">Modelo activo · Optuna TPE</div>
  </div>
  <span class="feature-card-badge badge-tech">Admin · Analista</span>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
<div class="feature-card green">
  <span class="fc-number">02 / DASHBOARD</span>
  <h3>Dashboard</h3>
  <p>Visualiza predicciones, KPIs y métricas del modelo activo.
     Cambia entre versiones históricas desde el panel lateral.</p>
  <div class="card-kpi">
    <div class="card-kpi-value lime">{_mape_display}</div>
    <div class="card-kpi-label">MAPE · walk-forward</div>
  </div>
  <span class="feature-card-badge badge-all">Todos los roles</span>
</div>
""", unsafe_allow_html=True)

with col3:
    st.markdown("""
<div class="feature-card amber">
  <span class="fc-number">03 / COMPARATIVA</span>
  <h3>Comparativa ML</h3>
  <p>Enfrenta SARIMA, Prophet, Regresión Lineal, Random Forest y XGBoost
     para encontrar el mejor predictor mensual del Tiggo 2.</p>
  <div class="card-kpi">
    <div class="card-kpi-value violet">5 modelos</div>
    <div class="card-kpi-label">SARIMA · Prophet · RF · XGB · LR</div>
  </div>
  <span class="feature-card-badge badge-tech">Admin · Analista</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
<div class="feature-card blue">
  <span class="fc-number">04 / CONCESIONARIOS</span>
  <h3>Concesionarios</h3>
  <p>Analiza ventas históricas y predicciones distribuidas por tienda.
     Simula escenarios de apertura, cierre y campañas locales.</p>
  <div class="card-kpi">
    <div class="card-kpi-value">Shares</div>
    <div class="card-kpi-label">Distribución por tienda · IC 95%</div>
  </div>
  <span class="feature-card-badge badge-tech">Admin · Analista · Manager</span>
</div>
""", unsafe_allow_html=True)

with col5:
    st.markdown("""
<div class="feature-card green">
  <span class="fc-number">05 / PROYECCIÓN</span>
  <h3>Proyección Ingresos</h3>
  <p>Traduce la predicción SARIMA en cifras financieras en USD.
     Configura precio, margen neto y tipo de cambio.</p>
  <div class="card-kpi">
    <div class="card-kpi-value lime">USD</div>
    <div class="card-kpi-label">Ingresos proyectados · IC 95%</div>
  </div>
  <span class="feature-card-badge badge-tech">Admin · Analista · Financiero</span>
</div>
""", unsafe_allow_html=True)

with col6:
    st.markdown("""
<div class="feature-card amber">
  <span class="fc-number">06 / ESCALABILIDAD</span>
  <h3>Escalabilidad</h3>
  <p>Hoja de ruta para exportar el pipeline a otras marcas,
     líneas de negocio y mercados geográficos de LatAm.</p>
  <div class="card-kpi">
    <div class="card-kpi-value violet">Multi-Marca</div>
    <div class="card-kpi-label">Portafolio · Onboarding · LatAm</div>
  </div>
  <span class="feature-card-badge badge-all">Todos los roles</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
  margin-top: 48px;
  padding-top: 16px;
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
