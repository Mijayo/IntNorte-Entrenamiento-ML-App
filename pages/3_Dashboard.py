"""
============================================================================
PÁGINA: DASHBOARD DE NEGOCIO
============================================================================
Tabs disponibles según rol:
  admin / analyst : Dashboard · Predicciones · Recomendaciones · Walk-Forward ·
                    Métricas Técnicas (sub-tabs: Resumen · ACF/PACF · Grid Search) ·
                    Asistente IA
  manager         : Dashboard · Predicciones · Recomendaciones · Asistente IA
  viewer          : Dashboard · Predicciones

La proyección financiera (💰 Proyección Ingresos) se trasladó a la página
independiente pages/5_Proyeccion_Ingresos.py (2026-05-27).
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
from core.auth_system import (guard_page, show_user_info, has_permission, show_header)
from core.styles import kpi_card, section_header, apply_chart_theme, COLORS

# ── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dashboard TIGGO 2", page_icon="🚗",
    layout="wide", initial_sidebar_state="expanded"
)

# CSS inyectado por show_header() vía styles.get_global_css()

# ── Auth ─────────────────────────────────────────────────────────────────────

guard_page("🚗 Dashboard TIGGO 2")

if 'cache_llm_tiggo' not in st.session_state:
    st.session_state.cache_llm_tiggo = {}
if 'cache_llm_run' not in st.session_state:
    st.session_state.cache_llm_run = None

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

# ── Alerta predictiva proactiva ───────────────────────────────────────────────
# Se activa cuando el próximo mes proyectado desvía ≥15% de la media histórica
# reciente, SIN necesitar ventas reales registradas — es forward-looking.

_hist_12_avg  = hist_total.iloc[-12:].mean() if len(hist_total) >= 12 else hist_total.mean()
_fc_next      = pred_total['Predicción'].iloc[0]
_fc_mes       = pred_total['Mes'].iloc[0]
_fc_ic_inf    = pred_total['IC_Inferior'].iloc[0]
_fc_ic_sup    = pred_total['IC_Superior'].iloc[0]
_fc_dev_pct   = (_fc_next - _hist_12_avg) / (_hist_12_avg + 0.01) * 100

if abs(_fc_dev_pct) >= 15:
    if _fc_dev_pct > 0:
        st.markdown(f"""
<div style="background:rgba(255,193,7,0.07);border:1px solid rgba(255,193,7,0.32);
            border-left:4px solid #FFC107;border-radius:8px;
            padding:14px 20px;margin-bottom:16px;">
  <span style="font-family:'Rajdhani',sans-serif;font-size:.72rem;font-weight:700;
               color:#FFC107;text-transform:uppercase;letter-spacing:.14em;">
    ⚡ Alerta predictiva — Demanda inusualmente alta
  </span><br>
  <span style="color:#C9D8E6;font-size:.9rem;">
    El modelo anticipa <strong style="color:#FCD34D;">{_fc_next:.0f} uds</strong>
    para <strong>{_fc_mes}</strong>
    (<strong style="color:#FCD34D;">+{_fc_dev_pct:.1f}%</strong> vs media
    últimos 12 m: {_hist_12_avg:.0f} uds · IC 95%: {_fc_ic_inf:.0f}–{_fc_ic_sup:.0f} uds).
  </span><br>
  <span style="color:#94A3B8;font-size:.82rem;">
    💡 Considera anticipar el pedido o incrementar el stock de seguridad
    antes de que se confirme la demanda.
  </span>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div style="background:rgba(56,189,248,0.07);border:1px solid rgba(56,189,248,0.28);
            border-left:4px solid #38BDF8;border-radius:8px;
            padding:14px 20px;margin-bottom:16px;">
  <span style="font-family:'Rajdhani',sans-serif;font-size:.72rem;font-weight:700;
               color:#38BDF8;text-transform:uppercase;letter-spacing:.14em;">
    ⚡ Alerta predictiva — Demanda inusualmente baja
  </span><br>
  <span style="color:#C9D8E6;font-size:.9rem;">
    El modelo anticipa <strong style="color:#7DD3FC;">{_fc_next:.0f} uds</strong>
    para <strong>{_fc_mes}</strong>
    (<strong style="color:#7DD3FC;">{_fc_dev_pct:.1f}%</strong> vs media
    últimos 12 m: {_hist_12_avg:.0f} uds · IC 95%: {_fc_ic_inf:.0f}–{_fc_ic_sup:.0f} uds).
  </span><br>
  <span style="color:#94A3B8;font-size:.82rem;">
    💡 Considera reducir el pedido del próximo período para evitar sobrestock.
  </span>
</div>
""", unsafe_allow_html=True)

# ── Indicador de frescura del modelo (sidebar) ────────────────────────────────

_age = sio.get_model_age_days(selected_run)
if _age is not None:
    if _age < 30:
        _fc, _fl, _fi = "#00F5A0", "Reciente", "🟢"
    elif _age < 90:
        _fc, _fl, _fi = "#FCD34D", "Envejeciendo", "🟡"
    else:
        _fc, _fl, _fi = "#FF3A5C", "Desactualizado", "🔴"
    st.sidebar.markdown(f"""
<div style="background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.07);
            border-radius:8px;padding:10px 13px;margin-top:10px;">
  <div style="font-family:'Rajdhani',sans-serif;font-size:0.72rem;color:#3F5060;
              text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px;">
    Frescura del modelo
  </div>
  <div style="font-family:'Rajdhani',sans-serif;font-size:1.05rem;font-weight:700;
              color:{_fc};">{_fi} {_fl}</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.76rem;
              color:#3F5060;margin-top:2px;">Entrenado hace {_age} días</div>
</div>
""", unsafe_allow_html=True)
    if _age >= 90:
        st.sidebar.warning("⚠️ Modelo con más de 90 días. Considera reentrenar.")

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
                    "💼 Recomendaciones", "🔄 Walk-Forward",
                    "📋 Métricas Técnicas", "🤖 Asistente IA"])
elif st.session_state.role == 'manager':
    tabs = st.tabs(["📊 Dashboard", "🔮 Predicciones",
                    "💼 Recomendaciones", "🤖 Asistente IA"])
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

    # ── Seguimiento en producción (ventas reales vs predicción) ──────────────
    _ventas_reales = sio.get_ventas_reales()
    if _ventas_reales:
        st.markdown(section_header("Seguimiento en Producción — Real vs Predicción", "📡"),
                    unsafe_allow_html=True)
        _vr_df = pd.DataFrame(_ventas_reales)
        _vr_df["fecha"] = pd.to_datetime(_vr_df["fecha"])
        _vr_df = _vr_df.sort_values("fecha")

        _pred_lookup = {
            pd.Timestamp(row["Fecha"]): row["Predicción"]
            for _, row in pred_total.iterrows()
        }
        _vr_df["prediccion"] = _vr_df["fecha"].map(_pred_lookup)
        _vr_matched = _vr_df.dropna(subset=["prediccion"])

        if not _vr_matched.empty:
            _vr_matched = _vr_matched.copy()
            _vr_matched["error_pct"] = (
                abs(_vr_matched["ventas"] - _vr_matched["prediccion"])
                / _vr_matched["ventas"] * 100
            )
            _max_err = _vr_matched["error_pct"].max()
            if _max_err > 15:
                st.error(
                    f"⚠️ **Drift detectado:** el error máximo entre ventas reales y predicción "
                    f"es **{_max_err:.1f}%** (umbral: 15%). Considera reentrenar el modelo."
                )
            else:
                st.success(
                    f"✅ Modelo en producción dentro del rango — error máximo: {_max_err:.1f}%"
                )

            _fig_drift = go.Figure()
            _fig_drift.add_trace(go.Scatter(
                x=_vr_matched["fecha"], y=_vr_matched["ventas"],
                mode="lines+markers", name="Real registrado",
                line=dict(color=COLORS["success"], width=2.5),
                marker=dict(size=9, color=COLORS["success"]),
            ))
            _fig_drift.add_trace(go.Scatter(
                x=_vr_matched["fecha"], y=_vr_matched["prediccion"],
                mode="lines+markers", name="Predicción modelo",
                line=dict(color=COLORS["accent"], width=2.5, dash="dot"),
                marker=dict(size=8, color=COLORS["accent"], symbol="diamond"),
            ))
            apply_chart_theme(_fig_drift, height=320,
                              title="Ventas reales vs predicción del modelo")
            _fig_drift.update_layout(hovermode="x unified",
                                     xaxis_title="Fecha", yaxis_title="Unidades")
            st.plotly_chart(_fig_drift, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info(
                "💡 Hay ventas reales registradas, pero no coinciden con las fechas de "
                "predicción del modelo activo. Registra ventas de los próximos meses "
                "en **Registrar Ventas**."
            )
    else:
        if st.session_state.role in ['admin', 'analyst']:
            st.info(
                "📡 Registra ventas reales en **Registrar Ventas** para ver el seguimiento "
                "de producción y detectar drift del modelo automáticamente."
            )

    # ── Contexto de Mercado — Chery vs Competencia ────────────────────────────
    if st.session_state.role in ['admin', 'analyst', 'manager']:
        st.markdown(section_header("Contexto de Mercado — Chery vs Competencia (Perú)", "🏁"),
                    unsafe_allow_html=True)

        _mkt = pd.DataFrame({
            'Marca':     ['Toyota', 'Hyundai', 'Kia', 'Chery', 'Chevrolet',
                          'Nissan', 'Suzuki', 'Otras'],
            'Share (%)': [24.5, 14.8, 11.9, 8.3, 7.6, 6.9, 5.2, 20.8],
        })
        _mkt_colors = [
            COLORS['success'] if m == 'Chery' else COLORS['muted']
            for m in _mkt['Marca']
        ]

        fig_mkt = go.Figure()
        fig_mkt.add_trace(go.Bar(
            x=_mkt['Marca'], y=_mkt['Share (%)'],
            marker=dict(color=_mkt_colors, opacity=0.88),
            text=[f"{v:.1f}%" for v in _mkt['Share (%)']],
            textposition='outside',
            textfont=dict(family="JetBrains Mono, monospace", size=11, color="#7A95A8"),
            hovertemplate='%{x}: %{y:.1f}% market share<extra></extra>',
        ))
        apply_chart_theme(fig_mkt, height=300,
                          title='Market Share Automotriz — Perú 2024 (estimado)')
        fig_mkt.update_layout(
            xaxis_title='', yaxis_title='Share (%)',
            showlegend=False, yaxis_ticksuffix='%',
        )
        st.plotly_chart(fig_mkt, use_container_width=True,
                        config={'displayModeBar': False})

        # Cuadro de posición competitiva
        _chery_rank = int(_mkt.sort_values('Share (%)', ascending=False)
                          .reset_index(drop=True)
                          .loc[lambda d: d['Marca'] == 'Chery'].index[0]) + 1
        _chery_share = _mkt.loc[_mkt['Marca'] == 'Chery', 'Share (%)'].values[0]
        _kia_share   = _mkt.loc[_mkt['Marca'] == 'Kia',   'Share (%)'].values[0]
        st.markdown(f"""
<div style="background:rgba(0,245,160,0.05);border:1px solid rgba(0,245,160,0.16);
            border-radius:8px;padding:12px 18px;margin-top:4px;">
  <span style="font-family:'Rajdhani',sans-serif;font-size:.72rem;font-weight:700;
               color:#00F5A0;text-transform:uppercase;letter-spacing:.12em;">
    Posición Chery
  </span>
  <span style="color:#94A3B8;font-size:.85rem;margin-left:10px;">
    Chery ocupa el <strong style="color:#C9D8E6;">{_chery_rank}.º lugar</strong>
    con un <strong style="color:#00F5A0;">{_chery_share:.1f}% de market share</strong>,
    a {abs(_chery_share - _kia_share):.1f}pp de Kia en el segmento SUV compacto.
    El sistema de predicción optimiza la disponibilidad del Tiggo 2 para defender
    y expandir esa posición frente a la competencia.
  </span>
</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;
            color:#3F5060;margin-top:6px;text-align:right;">
  Fuente: estimado ARAPER / ASOAVEC 2024 — referencia ilustrativa
</div>
""", unsafe_allow_html=True)

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
        _xls = sio.build_export_excel(pred_total, walk_forward, metricas, hist_total)
        st.download_button(
            "📥 Exportar Excel (predicciones + walk-forward + histórico)",
            _xls,
            f"reporte_tiggo2_{datetime.now().strftime('%Y%m%d')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ── Simulador de Escenarios — ¿Qué pasa si…? ─────────────────────────────
    with st.expander("🎯 Simulador de Escenarios — ¿Qué pasa si…?", expanded=False):
        st.markdown("""
Aplica un factor de ajuste sobre la predicción SARIMA para simular el impacto de
factores externos: campañas promocionales, cambio de precio, entrada de un competidor,
restricción de importaciones, o estacionalidad atípica.
""")
        sc1, sc2 = st.columns([2, 3])
        with sc1:
            _sc_ajuste = st.slider(
                "Ajuste de demanda (%)",
                min_value=-50, max_value=100, value=0, step=5,
                format="%+d%%",
                help=(
                    "+20% → campaña de ventas o lanzamiento de versión nueva.  \n"
                    "-15% → entrada de competidor directo o aumento de precio."
                ),
            )
            _sc_razon = st.text_input(
                "Descripción del escenario",
                placeholder="Ej: Campaña agosto +15%, precio reducido",
                max_chars=80,
            )

        with sc2:
            if _sc_ajuste != 0:
                _sc_factor = 1 + _sc_ajuste / 100
                _sc_pred   = pred_total.copy()
                _sc_pred['Pred_Ajust']    = (_sc_pred['Predicción'] * _sc_factor).round(1)
                _sc_pred['IC_Inf_Ajust']  = (_sc_pred['IC_Inferior'] * _sc_factor).clip(0).round(1)
                _sc_pred['IC_Sup_Ajust']  = (_sc_pred['IC_Superior'] * _sc_factor).round(1)

                _sc_color = COLORS['success'] if _sc_ajuste > 0 else COLORS['accent']
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=_sc_pred['Fecha'], y=_sc_pred['Predicción'],
                    mode='lines+markers', name='Base SARIMA',
                    line=dict(color=COLORS['accent'], width=2, dash='dot'),
                    marker=dict(size=7, color=COLORS['accent']),
                ))
                # Banda IC ajustada
                fig_sc.add_trace(go.Scatter(
                    x=_sc_pred['Fecha'].tolist() + _sc_pred['Fecha'].tolist()[::-1],
                    y=_sc_pred['IC_Sup_Ajust'].tolist() + _sc_pred['IC_Inf_Ajust'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0,245,160,0.06)' if _sc_ajuste > 0 else 'rgba(255,58,92,0.06)',
                    line=dict(color='rgba(0,0,0,0)'), showlegend=False,
                ))
                fig_sc.add_trace(go.Scatter(
                    x=_sc_pred['Fecha'], y=_sc_pred['Pred_Ajust'],
                    mode='lines+markers',
                    name=f'Escenario ({_sc_ajuste:+d}%)',
                    line=dict(color=_sc_color, width=2.5),
                    marker=dict(size=9, symbol='diamond', color=_sc_color,
                                line=dict(color='#080D18', width=1.5)),
                ))
                apply_chart_theme(
                    fig_sc, height=300,
                    title=_sc_razon or f"Escenario: demanda {_sc_ajuste:+d}%",
                )
                fig_sc.update_layout(hovermode='x unified',
                                     xaxis_title='Fecha', yaxis_title='Unidades')
                st.plotly_chart(fig_sc, use_container_width=True,
                                config={'displayModeBar': False})

                _sc_total_base = pred_total['Predicción'].sum()
                _sc_total_adj  = _sc_pred['Pred_Ajust'].sum()
                _sc_delta      = _sc_total_adj - _sc_total_base
                m1, m2, m3 = st.columns(3)
                m1.metric("Total base (6 m)", f"{_sc_total_base:.0f} uds")
                m2.metric("Total escenario (6 m)", f"{_sc_total_adj:.0f} uds",
                          delta=f"{_sc_delta:+.0f} uds")
                m3.metric("Factor aplicado", f"{_sc_factor:.2f}×")
            else:
                st.info("Mueve el slider para simular un escenario alternativo.")

# ── Tab 3: Recomendaciones (admin + analyst + manager) ───────────────────────

if st.session_state.role in ['admin', 'analyst', 'manager']:
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

        # ── Análisis del Ciclo de Valor ───────────────────────────────────────
        st.markdown("---")
        st.subheader("📆 Análisis del Ciclo de Valor — Estacionalidad y Negocio")

        # Calcular medias por mes del año
        monthly_avg = hist_total.groupby(hist_total.index.month).mean()
        month_names = ['Ene','Feb','Mar','Abr','May','Jun',
                       'Jul','Ago','Sep','Oct','Nov','Dic']
        _peak_month_idx = monthly_avg.idxmax()
        _low_month_idx  = monthly_avg.idxmin()
        _peak_name  = month_names[_peak_month_idx - 1]
        _low_name   = month_names[_low_month_idx  - 1]
        _peak_val   = monthly_avg.max()
        _low_val    = monthly_avg.min()
        _estac_ratio = _peak_val / _low_val

        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.markdown(kpi_card("Mes pico histórico",  f"{_peak_name}",  "📈"), unsafe_allow_html=True)
        col_s2.markdown(kpi_card("Mes valle histórico", f"{_low_name}",   "📉", "amber"), unsafe_allow_html=True)
        col_s3.markdown(kpi_card("Ratio pico/valle",    f"{_estac_ratio:.1f}x", "📊", "blue"), unsafe_allow_html=True)

        fig_seas = go.Figure()
        bar_colors_seas = [
            COLORS['accent'] if v == monthly_avg.max() else
            (COLORS['primary'] if v >= monthly_avg.mean() else COLORS['muted'])
            for v in monthly_avg.values
        ]
        fig_seas.add_trace(go.Bar(
            x=month_names,
            y=monthly_avg.values,
            marker=dict(color=bar_colors_seas, opacity=0.88),
            text=[f"{v:.1f}" for v in monthly_avg.values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=11, color="#7A95A8"),
            hovertemplate="%{x}: %{y:.1f} uds/mes (media histórica)<extra></extra>",
        ))
        fig_seas.add_hline(
            y=monthly_avg.mean(), line_dash="dot", line_color=COLORS['secondary'],
            annotation_text=f"Media: {monthly_avg.mean():.1f} uds",
            annotation_font_color=COLORS['secondary'],
        )
        apply_chart_theme(fig_seas, height=340,
                          title="Media histórica de ventas por mes del año — TIGGO 2")
        fig_seas.update_layout(xaxis_title="Mes", yaxis_title="Unidades (media)")
        st.plotly_chart(fig_seas, use_container_width=True, config={"displayModeBar": False})

        # Insights de negocio sobre la estacionalidad
        st.markdown(f"""
<div style="background:rgba(0,115,255,0.06);border:1px solid rgba(0,115,255,0.18);
            border-radius:10px;padding:16px 20px;margin-top:4px;">
<span style="font-family:'Rajdhani',sans-serif;font-size:1.0rem;font-weight:700;
             color:#0073FF;text-transform:uppercase;letter-spacing:.08em;">
💡 Lectura de Negocio — Ciclo de Valor del Tiggo 2
</span>
<ul style="color:#94A3B8;font-size:0.9rem;margin-top:10px;padding-left:20px;line-height:1.9;">
  <li><strong style="color:#C9D8E6;">Pico en {_peak_name}:</strong> concentración de ventas
      que puede estar ligada al <em>rappel del proveedor</em> — las distribuidoras aceleran
      cierres de mes/trimestre para alcanzar metas de volumen y acceder a descuentos retroactivos
      del fabricante Chery.</li>
  <li><strong style="color:#C9D8E6;">Ratio {_estac_ratio:.1f}x pico/valle:</strong> variabilidad
      significativa que justifica la predicción mes a mes en lugar de usar una media anual simple.</li>
  <li><strong style="color:#C9D8E6;">Oportunidad de des-estacionalización:</strong> si la concentración
      de ventas en un mes responde a incentivos comerciales internos (rappel, cuota), la solución
      podría orientarse a <em>suavizar</em> esa concentración distribuyendo el esfuerzo de venta
      a lo largo del trimestre — reduciendo el riesgo de quiebre de stock en el pico.</li>
</ul>
</div>
""", unsafe_allow_html=True)

        # ── Explicación académica ──────────────────────────────────────────────
        with st.expander("📚 Marco teórico — ¿Por qué estas estrategias?", expanded=False):
            st.markdown("""
### Fundamentos teóricos de las estrategias de compra

---

#### 1. Por qué se usa el intervalo de confianza superior como base

El sistema construye la recomendación a partir del **límite superior del IC 95%** del modelo SARIMA,
no de la predicción puntual. La razón es asimétrica por naturaleza:

> *El coste de quedarse sin stock (venta perdida, daño a la relación con el cliente) supera
> habitualmente al coste de mantener unidades adicionales en inventario.*

Este principio está formalizado en el **Problema del Vendedor de Periódicos** (*Newsvendor Problem*,
Scarf, 1958): la cantidad óptima de pedido se desplaza hacia el percentil más alto cuando el
*ratio crítico* `c_u / (c_u + c_o)` es elevado, donde `c_u` = coste de sub-stock y `c_o` = coste
de sobrestock. En el sector automotriz, un `c_u` alto es habitual dado el tiempo de reposición
de unidades importadas.

El IC 95% de SARIMA corresponde aproximadamente al **percentil 97.5 de la distribución
predictiva**, asumiendo innovaciones gaussianas (Box & Jenkins, 1976). Usarlo como base
garantiza que el pedido cubre la demanda real en ~97 de cada 100 meses bajo los supuestos
del modelo.

---

#### 2. Estrategia Conservadora — IC superior × 1.10

El buffer del **+10%** sobre el IC superior actúa como **stock de seguridad** para absorber:

- **Error residual del modelo**: el MAPE walk-forward indica la desviación media esperada entre
  predicción y demanda real. Un buffer del 10% es coherente con errores de modelo del orden
  del 5–15%.
- **Variabilidad no capturada**: factores exógenos puntuales (campañas, tipo de cambio) no
  modelados por SARIMA.

**Marco de referencia:** Fórmula clásica de stock de seguridad de Silver, Pyke & Thomas (1998):

```
SS = z · σ_L
```

donde `z` es el factor de servicio (z = 1.65 para nivel de servicio del 95%) y `σ_L` la
desviación estándar de la demanda durante el lead time. El +10% es una aproximación práctica
cuando no se dispone de `σ_L` directa pero sí de un IC estadístico del modelo.

**Cuándo aplicarla:** tendencia estable o decreciente, capacidad de almacenamiento limitada,
alto coste financiero del inventario.

---

#### 3. Estrategia Agresiva — IC superior × 1.20

El buffer del **+20%** corresponde a una política de **nivel de servicio tipo I** más exigente,
apuntando a cubrir prácticamente todos los escenarios de demanda del mes siguiente.

**Marco de referencia:** En gestión de inventarios, el *fill rate* objetivo determina el
percentil de la distribución de demanda que se debe cubrir. Un buffer del +20% sobre el percentil
97.5 del modelo equivale a apuntar a un nivel de servicio del orden del **99%**, apropiado cuando:

- La tendencia reciente es **creciente** (la demanda puede superar la distribución histórica).
- El tiempo de reposición es largo (importaciones: 60–90 días), haciendo costoso un stockout.
- La elasticidad precio-demanda del Tiggo 2 es baja (el cliente espera la unidad en vez de
  sustituir el modelo).

**Coste de oportunidad vs. coste de sobrestock:** si el margen bruto por unidad vendida supera
al coste mensual de financiar el inventario sobrante, la estrategia agresiva maximiza el
beneficio esperado (principio del *expected profit maximization*, Zipkin, 2000).

---

#### 4. Cómo leer la señal de tendencia

El cálculo compara la **media de los últimos 3 meses** con el **promedio histórico completo**:

```
tendencia (%) = (media_3m − media_histórica) / media_histórica × 100
```

- **> +10 %** → la demanda reciente supera la media histórica → el modelo puede estar
  subestimando; la estrategia agresiva cubre ese riesgo.
- **< −10 %** → la demanda reciente está por debajo de la media → riesgo de sobrestock;
  la estrategia conservadora protege el capital circulante.
- **Entre ±10 %** → el modelo está bien calibrado con el ciclo histórico; la predicción
  puntual es referencia suficiente.

---

#### 5. Limitaciones y condiciones de validez

| Condición | Impacto si no se cumple |
|-----------|------------------------|
| Innovaciones SARIMA gaussianas | Los IC 95% pueden ser más amplios de lo necesario o insuficientes ante colas pesadas |
| Estacionariedad tras diferenciación | Si la serie tiene cambios estructurales recientes, la predicción se sesga |
| Ausencia de outliers no tratados | Un mes atípico no depurado distorsiona el IC y las recomendaciones |
| MAPE walk-forward < 15 % | Con errores > 15 %, añadir un buffer adicional del 5–10 % sobre las estrategias es prudente |

> **Referencia principal:** Box, G.E.P., Jenkins, G.M. & Reinsel, G.C. (1976).
> *Time Series Analysis: Forecasting and Control*. Prentice Hall.
""")


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

    # Walk-Forward
    with tabs[3]:
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

    # Métricas Técnicas (con sub-tabs: Resumen · ACF/PACF · Grid Search · vs Descartados)
    with tabs[4]:
        st.header("📋 Métricas Técnicas Completas", divider='gray')
        sub_tabs = st.tabs(["📊 Resumen", "🔬 ACF/PACF", "🔍 Grid Search", "🏆 vs Descartados"])

        # ── Sub-tab 1: Resumen ────────────────────────────────────────────────
        with sub_tabs[0]:
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

        # ── Sub-tab 2: ACF/PACF ───────────────────────────────────────────────
        with sub_tabs[1]:
            st.subheader("🔬 Análisis ACF/PACF")
            acf_bytes, pacf_bytes = sio.load_acf_pacf_images(selected_run)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ACF — Autocorrelación")
                if acf_bytes:
                    st.image(acf_bytes, use_container_width=True)
                else:
                    st.warning("Imagen ACF no disponible")
            with col2:
                st.subheader("PACF — Autocorrelación Parcial")
                if pacf_bytes:
                    st.image(pacf_bytes, use_container_width=True)
                else:
                    st.warning("Imagen PACF no disponible")

        # ── Sub-tab 3: Grid Search ────────────────────────────────────────────
        with sub_tabs[2]:
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

        # ── Sub-tab 4: vs Descartados ─────────────────────────────────────────
        with sub_tabs[3]:
            st.markdown(section_header("SARIMA vs Modelos Alternativos", "🏆"), unsafe_allow_html=True)

            cml = st.session_state.get("cml_resultados")
            # Si no hay resultados en sesión, intentar cargarlos desde Supabase
            if cml is None:
                cml = sio.load_cml_resultados(selected_run)

            if cml is not None:
                # Resultados en tiempo real desde Comparativa ML
                df_met_cmp = cml["metricas"]
                ganador_cmp = cml["ganador"]

                st.markdown(f"""
<div style="background:rgba(0,245,160,0.08);border:1px solid rgba(0,245,160,0.25);
            border-radius:10px;padding:12px 20px;margin-bottom:18px;">
<span style="font-size:1.0rem;font-weight:700;color:#00F5A0;">🏆 Ganador: {ganador_cmp}</span>
<span style="color:#94A3B8;font-size:0.88rem;margin-left:12px;">
Resultados en tiempo real &nbsp;·&nbsp; Comparativa ejecutada en esta sesión
</span>
</div>
""", unsafe_allow_html=True)

                st.dataframe(
                    df_met_cmp.style
                        .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], axis=0,
                                       color="rgba(0,245,160,0.12)")
                        .highlight_max(subset=["R²"], axis=0,
                                       color="rgba(0,245,160,0.12)")
                        .format({"MAE": "{:.2f}", "RMSE": "{:.2f}",
                                 "MAPE (%)": "{:.1f}%", "R²": "{:.3f}"}),
                    use_container_width=True,
                )

                fig_cmp = go.Figure()
                bar_colors = [COLORS['success'] if m == ganador_cmp else COLORS['primary']
                              for m in df_met_cmp.index]
                fig_cmp.add_trace(go.Bar(
                    x=df_met_cmp.index,
                    y=df_met_cmp["MAPE (%)"],
                    marker=dict(color=bar_colors, opacity=0.88),
                    text=[f"{v:.1f}%" for v in df_met_cmp["MAPE (%)"]],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono, monospace", size=12, color="#7A95A8"),
                ))
                fig_cmp.add_hline(y=15, line_dash="dot", line_color=COLORS['accent'],
                                  annotation_text="Objetivo < 15%",
                                  annotation_font_color=COLORS['accent'])
                apply_chart_theme(fig_cmp, height=380,
                                  title="MAPE por modelo — período de test (menor = mejor)")
                fig_cmp.update_layout(yaxis_title="MAPE (%)", showlegend=False,
                                      yaxis=dict(ticksuffix="%"))
                st.plotly_chart(fig_cmp, use_container_width=True,
                                config={"displayModeBar": False})

            else:
                st.info(
                    "💡 Ejecuta la página **Comparativa ML** con este run para guardar los "
                    "resultados numéricos aquí. Mientras tanto, aquí está la justificación "
                    "metodológica de la selección."
                )

                # KPIs SARIMA desde datos live
                c1, c2, c3 = st.columns(3)
                c1.markdown(kpi_card("MAPE SARIMA (walk-forward)", f"{mape_wf:.1f}%",
                                     "🎯", "amber"), unsafe_allow_html=True)
                c2.markdown(kpi_card("Combinaciones evaluadas", len(grid_search),
                                     "🔢", "blue"), unsafe_allow_html=True)
                c3.markdown(kpi_card("Mejor AIC",
                                     f"{grid_search.loc[grid_search['mape'].idxmin(),'aic']:.0f}",
                                     "📐"), unsafe_allow_html=True)

            # Tabla de justificación metodológica (siempre visible)
            st.markdown(section_header("¿Por qué se descartaron las otras familias?", "📋"),
                        unsafe_allow_html=True)
            df_eval = pd.DataFrame([
                {
                    "Modelo":         "SARIMA (p,d,q)(P,D,Q)[12]  ✅",
                    "Familia":        "Serie Temporal",
                    "Estacionalidad": "Explícita — periodo m=12",
                    "Variable exóg.": "✅ ventas_otros (Pearson)",
                    "Req. mínimo":    "24 meses",
                    "Por qué ganó":   "Modela directamente la estructura AR+MA estacional del Tiggo 2 con exógena; menor MAPE en test.",
                },
                {
                    "Modelo":         "Prophet",
                    "Familia":        "Serie Temporal",
                    "Estacionalidad": "Automática (changepoints)",
                    "Variable exóg.": "❌ no incorporada",
                    "Req. mínimo":    "24 meses",
                    "Por qué no":     "Sin variable exógena pierde el efecto Pearson. Sobreajusta en series cortas con picos de diciembre.",
                },
                {
                    "Modelo":         "Regresión Lineal",
                    "Familia":        "ML — lag features",
                    "Estacionalidad": "Implícita (lag_12)",
                    "Variable exóg.": "Como feature",
                    "Req. mínimo":    "17 meses + test",
                    "Por qué no":     "Asume relación lineal entre lags y ventas futuras; no captura tendencia ni no-linealidad estacional.",
                },
                {
                    "Modelo":         "Random Forest",
                    "Familia":        "ML — lag features",
                    "Estacionalidad": "Implícita (lag_12)",
                    "Variable exóg.": "Como feature",
                    "Req. mínimo":    "17 meses + test",
                    "Por qué no":     "Overfitting con n pequeño (51 meses). No extrapola tendencia — predice dentro del rango de entrenamiento.",
                },
                {
                    "Modelo":         "XGBoost",
                    "Familia":        "ML — lag features",
                    "Estacionalidad": "Implícita (lag_12)",
                    "Variable exóg.": "Como feature",
                    "Req. mínimo":    "17 meses + test",
                    "Por qué no":     "Igual que RF. Los árboles no extrapolan — MAPE se dispara fuera del rango histórico.",
                },
            ])
            st.dataframe(df_eval.set_index("Modelo"), use_container_width=True)

# ── Tab LLM (admin / analyst) ────────────────────────────────────────────────

if st.session_state.role in ['admin', 'analyst']:
    with tabs[5]:
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

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="app-footer">Sistema TIGGO 2 &nbsp;·&nbsp; ISDI &nbsp;·&nbsp; Predicción de Demanda</div>',
    unsafe_allow_html=True,
)
