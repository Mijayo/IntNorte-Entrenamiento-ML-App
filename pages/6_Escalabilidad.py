"""
============================================================================
PÁGINA: ESCALABILIDAD — Plataforma Multi-Marca
============================================================================
Demuestra cómo el pipeline SARIMA se generaliza a otras marcas, modelos
de vehículo, líneas de negocio y mercados geográficos.
El tribunal aconseja explícitamente exportar el modelo a otras marcas
y líneas de negocio — esta página plasma esa hoja de ruta técnica.
Acceso: todos los roles autenticados.
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.auth_system import (guard_page, show_user_info, show_header)
from core.styles import kpi_card, section_header, apply_chart_theme, COLORS

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Escalabilidad — Plataforma Multi-Marca", page_icon="🌐",
    layout="wide", initial_sidebar_state="expanded"
)

# ── Auth ──────────────────────────────────────────────────────────────────────

guard_page("🌐 Escalabilidad — Plataforma Multi-Marca")

show_header("Escalabilidad — Plataforma Multi-Marca",
            "Generalización del pipeline SARIMA a otras marcas, modelos y líneas de negocio")
show_user_info()

# ── KPI Banner ────────────────────────────────────────────────────────────────

st.markdown('<div style="margin-bottom:18px"></div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card("Marcas Compatibles", "∞", icon="🏷️",
                         sub="Cualquier marca con historial ≥ 36 meses"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("Semanas para Onboarding", "2–4", icon="⚡", color_class="green",
                         sub="Desde datos raw hasta dashboard en vivo"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("Datos Mínimos", "36 meses", icon="📅", color_class="amber",
                         sub="3 ciclos estacionales completos para SARIMA"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("Líneas de Negocio", "6+", icon="💼", color_class="purple",
                         sub="Vehículos, repuestos, seguros, servicio…"), unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🏗️ Arquitectura",
    "🚗 Portafolio",
    "💼 Líneas de Negocio",
    "📋 Playbook de Onboarding",
    "🌎 Expansión Geográfica",
    "🚀 Visión del Producto",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: ARQUITECTURA MULTI-MARCA
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown(section_header("Pipeline Genérico — Brand-Agnostic SARIMA", "🏗️"),
                unsafe_allow_html=True)

    # ── Diagrama de pipeline ──────────────────────────────────────────────────

    st.markdown("""
<style>
.arch-pipeline {
  display: flex; align-items: stretch; gap: 0;
  margin: 28px 0 10px; overflow-x: auto;
  padding: 18px 4px 10px;
}
.arch-node {
  background: linear-gradient(150deg, #131210 0%, #1A1815 100%);
  border: 1px solid rgba(0,115,255,0.18);
  border-radius: 6px;
  padding: 18px 12px 14px;
  min-width: 140px; flex: 1; flex-shrink: 0;
  text-align: center; position: relative;
  display: flex; flex-direction: column; align-items: center;
}
.arch-node.n-generic { border-color: rgba(0,115,255,0.35); }
.arch-node.n-active  { border-color: rgba(0,245,160,0.45);
  box-shadow: 0 0 18px rgba(0,245,160,0.07); }
.arch-node-badge {
  position: absolute; top: -9px; left: 50%; transform: translateX(-50%);
  font-family: 'Rajdhani', sans-serif;
  font-size: .56rem; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; padding: 2px 8px; border-radius: 2px;
  white-space: nowrap;
}
.badge-var   { background: rgba(0,245,160,.10); color: #00F5A0; border: 1px solid rgba(0,245,160,.25); }
.badge-fixed { background: rgba(0,115,255,.09); color: #0073FF; border: 1px solid rgba(0,115,255,.22); }
.arch-icon   { font-size: 1.7rem; margin-bottom: 10px; }
.arch-title  {
  font-family: 'Rajdhani', sans-serif;
  font-size: .8rem; font-weight: 700;
  color: #C9D8E6; letter-spacing: .1em;
  text-transform: uppercase; margin-bottom: 6px;
}
.arch-sub    {
  font-family: 'JetBrains Mono', monospace;
  font-size: .6rem; color: #3F5060;
  letter-spacing: .02em; line-height: 1.6;
}
.arch-arrow  {
  display: flex; align-items: center; justify-content: center;
  padding: 0 2px; flex-shrink: 0;
  font-size: 1.3rem; color: #1A2838;
}
</style>

<div class="arch-pipeline">
  <div class="arch-node">
    <span class="arch-node-badge badge-var">Variable por marca</span>
    <span class="arch-icon">📄</span>
    <span class="arch-title">Datos de Entrada</span>
    <span class="arch-sub">Excel / API<br>FECHA-VENTA<br>MARCA · MODELO3<br>≥ 36 meses</span>
  </div>
  <div class="arch-arrow">›</div>
  <div class="arch-node n-generic">
    <span class="arch-node-badge badge-fixed">Genérico</span>
    <span class="arch-icon">✅</span>
    <span class="arch-title">Validación</span>
    <span class="arch-sub">utils_validacion<br>nulos, outliers<br>ADF estacion.<br>temporal cover.</span>
  </div>
  <div class="arch-arrow">›</div>
  <div class="arch-node n-generic">
    <span class="arch-node-badge badge-fixed">Genérico</span>
    <span class="arch-icon">🔍</span>
    <span class="arch-title">Optuna TPE</span>
    <span class="arch-sub">80 trials bayesianos<br>p∈{0-3} d∈{0-1}<br>q∈{0-3} P∈{0-1}<br>criterio: MAPE mín.</span>
  </div>
  <div class="arch-arrow">›</div>
  <div class="arch-node n-generic">
    <span class="arch-node-badge badge-fixed">Genérico</span>
    <span class="arch-icon">🤖</span>
    <span class="arch-title">SARIMAX</span>
    <span class="arch-sub">(p,d,q)(P,D,Q)[12]<br>exog: ventas_otros<br>walk-forward val.<br>IC 95% forecast</span>
  </div>
  <div class="arch-arrow">›</div>
  <div class="arch-node n-generic">
    <span class="arch-node-badge badge-fixed">Genérico</span>
    <span class="arch-icon">☁️</span>
    <span class="arch-title">Supabase</span>
    <span class="arch-sub">Storage .pkl.gz<br>PostgreSQL<br>training_runs<br>audit_log</span>
  </div>
  <div class="arch-arrow">›</div>
  <div class="arch-node n-generic">
    <span class="arch-node-badge badge-fixed">Genérico</span>
    <span class="arch-icon">📊</span>
    <span class="arch-title">Dashboard</span>
    <span class="arch-sub">KPIs · Predicciones<br>Walk-Forward<br>Asistente IA<br>Multi-rol RBAC</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="success-box">
<strong>Insight clave:</strong> El único componente que cambia entre marcas es el archivo de entrada.
Todo el pipeline — validación automática, búsqueda bayesiana de hiperparámetros, entrenamiento SARIMA,
almacenamiento en Supabase y visualización en el Dashboard — es completamente genérico y reutilizable
sin modificar una sola línea de código.
</div>
""", unsafe_allow_html=True)

    # ── Qué cambia vs qué se reutiliza ───────────────────────────────────────

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(section_header("Lo que cambia por marca / modelo", "🔄"),
                    unsafe_allow_html=True)
        st.markdown("""
**Parámetros configurables en la página de Entrenamiento:**

| Parámetro | Tiggo 2 (actual) | Nueva marca |
|-----------|:----------------:|:-----------:|
| Archivo Excel | `veh_ml_features.xlsx` | cualquier `.xlsx` |
| Filtro MARCA | `CHERY` | p.ej. `JAC` |
| Filtro MODELO3 | `TIGGO 2` | p.ej. `HUNTER` |
| Fecha inicio | `2024-01-01` | según disponibilidad |
| Máx. ventas/mes | `100` uds | según segmento |
| Horizonte | `6` meses | 3–12 configurable |
""")

    with col2:
        st.markdown(section_header("Lo que se reutiliza sin cambios", "♻️"),
                    unsafe_allow_html=True)
        st.markdown("""
**Infraestructura y código 100% reutilizables:**

- **Pipeline completo** — validación, Optuna, SARIMA, walk-forward
- **Variable exógena automática** — `ventas_otros` (Pearson r ≥ 0.3)
- **Supabase Storage + PostgreSQL** — misma instancia, distintos runs
- **Dashboard multi-rol RBAC** — mismo código, run activo configurable
- **Comparativa de 5 modelos ML** — SARIMA · Prophet · LR · RF · XGB
- **Proyección de Ingresos** — precio, margen y tipo de cambio ajustables
- **Asistente IA Gemini** — contexto de la marca/modelo cargado automát.
- **Sistema de autenticación** — Supabase Auth + roles, sin cambios
- **17 tests unitarios** — cobertura total, sin modificar
""")

    # ── Cómo el sistema ya soporta multi-marca ────────────────────────────────

    st.markdown(section_header("El sistema ya es multi-marca hoy", "⚙️"), unsafe_allow_html=True)

    st.markdown("""
<div class="warning-box">
<strong>La tabla <code>training_runs</code> en Supabase ya tiene columnas <code>marca</code> y <code>modelo</code></strong>
que se guardan con cada entrenamiento. Lo único necesario para operar con múltiples marcas en paralelo es:
<ol style="margin-top:8px; padding-left:20px; font-family:'Rajdhani',sans-serif; font-size:.9rem;">
  <li>Añadir una columna <code>activo_marca_modelo</code> con índice único por <em>(marca, modelo)</em>
      en lugar del índice global actual — permite un modelo activo por par marca+modelo.</li>
  <li>Agregar un selector <strong>Marca / Modelo</strong> al sidebar del Dashboard para filtrar el run activo.</li>
  <li>Namespacing en Storage: <code>{marca}/{modelo}/runs/{run_name}/</code> en lugar de la ruta plana actual.</li>
</ol>
Estimación: <strong>1–2 días de desarrollo</strong> para completar la migración multi-tenant.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: PORTAFOLIO DE EXPANSIÓN
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown(section_header("Portafolio de Modelos — Estado de Expansión", "🚗"),
                unsafe_allow_html=True)

    # ── Datos del portafolio ──────────────────────────────────────────────────

    portfolio = pd.DataFrame([
        # Chery activos y en evaluación
        {"Marca": "CHERY", "Modelo": "TIGGO 2",    "Segmento": "SUV Compacto",   "Estado": "✅ Activo",
         "Dem. Est. (uds/mes)": 65, "Historial (meses)": 51, "MAPE Est. (%)": 10.3, "Prioridad": 1},
        {"Marca": "CHERY", "Modelo": "TIGGO 4 PRO","Segmento": "SUV Compacto",   "Estado": "🔄 En evaluación",
         "Dem. Est. (uds/mes)": 45, "Historial (meses)": 38, "MAPE Est. (%)": 12.0, "Prioridad": 2},
        {"Marca": "CHERY", "Modelo": "ARRIZO 5",   "Segmento": "Sedán",           "Estado": "🔄 En evaluación",
         "Dem. Est. (uds/mes)": 22, "Historial (meses)": 40, "MAPE Est. (%)": 13.5, "Prioridad": 2},
        {"Marca": "CHERY", "Modelo": "TIGGO 5X",   "Segmento": "SUV Mediano",     "Estado": "📋 Pendiente",
         "Dem. Est. (uds/mes)": 30, "Historial (meses)": 36, "MAPE Est. (%)": 14.0, "Prioridad": 3},
        {"Marca": "CHERY", "Modelo": "TIGGO 7 PRO","Segmento": "SUV Mediano",     "Estado": "📋 Pendiente",
         "Dem. Est. (uds/mes)": 18, "Historial (meses)": 30, "MAPE Est. (%)": 17.0, "Prioridad": 3},
        {"Marca": "CHERY", "Modelo": "TIGGO 8 PRO","Segmento": "SUV Grande",      "Estado": "📋 Pendiente",
         "Dem. Est. (uds/mes)": 10, "Historial (meses)": 28, "MAPE Est. (%)": 20.0, "Prioridad": 3},
        # Otras marcas del grupo / competidores clave
        {"Marca": "JAC",   "Modelo": "HUNTER PLUS","Segmento": "Pick-up 4x4",     "Estado": "🎯 Potencial",
         "Dem. Est. (uds/mes)": 55, "Historial (meses)": 0,  "MAPE Est. (%)": None, "Prioridad": 4},
        {"Marca": "JAC",   "Modelo": "SEI 7",      "Segmento": "SUV Grande",      "Estado": "🎯 Potencial",
         "Dem. Est. (uds/mes)": 20, "Historial (meses)": 0,  "MAPE Est. (%)": None, "Prioridad": 4},
        {"Marca": "BYD",   "Modelo": "ATTO 3",     "Segmento": "SUV Eléctrico",   "Estado": "🎯 Potencial",
         "Dem. Est. (uds/mes)": 30, "Historial (meses)": 0,  "MAPE Est. (%)": None, "Prioridad": 4},
        {"Marca": "MG",    "Modelo": "ZS",          "Segmento": "SUV Compacto",   "Estado": "🎯 Potencial",
         "Dem. Est. (uds/mes)": 35, "Historial (meses)": 0,  "MAPE Est. (%)": None, "Prioridad": 4},
    ])

    # Filtro de marcas
    marcas_disponibles = ["Todas"] + sorted(portfolio["Marca"].unique().tolist())
    marca_sel = st.selectbox("Filtrar por marca", marcas_disponibles, key="port_marca")
    df_port = portfolio if marca_sel == "Todas" else portfolio[portfolio["Marca"] == marca_sel]

    # ── Gráfico: demanda estimada por modelo ──────────────────────────────────

    color_map = {
        "✅ Activo":          COLORS["success"],
        "🔄 En evaluación":  COLORS["primary"],
        "📋 Pendiente":      COLORS["secondary"],
        "🎯 Potencial":      COLORS["purple"],
    }
    df_chart = df_port.sort_values("Dem. Est. (uds/mes)", ascending=True)
    bar_colors = [color_map[e] for e in df_chart["Estado"]]

    fig_port = go.Figure()
    fig_port.add_trace(go.Bar(
        y=df_chart["Marca"] + " · " + df_chart["Modelo"],
        x=df_chart["Dem. Est. (uds/mes)"],
        orientation="h",
        marker=dict(color=bar_colors, opacity=0.88),
        text=df_chart["Dem. Est. (uds/mes)"].apply(lambda x: f"{x} uds/mes"),
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color="#7A95A8"),
        hovertemplate="<b>%{y}</b><br>Demanda estimada: %{x} uds/mes<extra></extra>",
    ))
    apply_chart_theme(fig_port, height=380,
                      title="Demanda estimada mensual por modelo (unidades / mes)")
    fig_port.update_layout(
        xaxis_title="Unidades / mes",
        margin=dict(l=20, r=80, t=50, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig_port, use_container_width=True, config={"displayModeBar": False})

    # ── Leyenda de estados ────────────────────────────────────────────────────

    leg1, leg2, leg3, leg4 = st.columns(4)
    for col, estado, color, desc in [
        (leg1, "✅ Activo",         "#00F5A0", "En producción — modelo entrenado y publicado"),
        (leg2, "🔄 En evaluación",  "#0073FF", "Datos disponibles ≥ 36 meses — listo para entrenar"),
        (leg3, "📋 Pendiente",      "#C2FF00", "Historia < 36 meses — requerirá espera de datos"),
        (leg4, "🎯 Potencial",      "#A78BFA", "Marca externa — requiere acceso a datos de ventas"),
    ]:
        col.markdown(
            f'<div style="border-left:3px solid {color};padding:8px 12px;'
            f'background:rgba(0,0,0,.25);border-radius:0 4px 4px 0;margin-bottom:6px">'
            f'<span style="font-family:\'Rajdhani\',sans-serif;font-size:.78rem;font-weight:700;'
            f'color:{color};text-transform:uppercase;letter-spacing:.08em">{estado}</span>'
            f'<br><span style="font-family:\'JetBrains Mono\',monospace;font-size:.62rem;'
            f'color:#3F5060">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Tabla detallada ───────────────────────────────────────────────────────

    st.markdown(section_header("Tabla Detallada — Portafolio", "📋"), unsafe_allow_html=True)
    display_cols = ["Marca", "Modelo", "Segmento", "Estado",
                    "Dem. Est. (uds/mes)", "Historial (meses)", "MAPE Est. (%)"]
    st.dataframe(
        df_port[display_cols].reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )

    # ── Nota sobre portencial total ───────────────────────────────────────────

    total_activo = portfolio.loc[portfolio["Estado"] == "✅ Activo", "Dem. Est. (uds/mes)"].sum()
    total_eval   = portfolio.loc[portfolio["Estado"].isin(["✅ Activo", "🔄 En evaluación"]), "Dem. Est. (uds/mes)"].sum()
    total_chery  = portfolio.loc[portfolio["Marca"] == "CHERY", "Dem. Est. (uds/mes)"].sum()
    total_todo   = portfolio["Dem. Est. (uds/mes)"].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Cobertura actual", f"{total_activo} uds/mes",
              help="Demanda mensual cubierta con el modelo activo (Tiggo 2)")
    m2.metric("Con portafolio Chery completo", f"{total_chery} uds/mes",
              delta=f"+{total_chery - total_activo} uds/mes",
              help="Suma de todos los modelos Chery con historial suficiente")
    m3.metric("Con portafolio completo (multi-marca)", f"{total_todo} uds/mes",
              delta=f"+{total_todo - total_activo} uds/mes",
              help="Incluyendo marcas potenciales (JAC, BYD, MG)")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: LÍNEAS DE NEGOCIO
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown(section_header("Expansión a Líneas de Negocio Adyacentes", "💼"),
                unsafe_allow_html=True)

    st.markdown("""
El mismo framework de predicción de series temporales que predice ventas de vehículos
puede aplicarse directamente a otras líneas de negocio del distribuidor automotriz,
con adaptaciones menores en los datos de entrada.
""")

    # ── Datos de líneas de negocio ────────────────────────────────────────────

    lineas = [
        {
            "Línea de Negocio": "🚗 Vehículos Nuevos",
            "Estado":           "✅ Activo",
            "Frecuencia":       "Mensual",
            "Algoritmo":        "SARIMAX + Optuna",
            "MAPE Esperado":    "10–15 %",
            "Sem. Implementar": 0,
            "ROI Potencial":    "Alto",
            "Disponib. Datos":  "Alta",
            "Complejidad":      "Media",
            "Descripción":      "Predicción de unidades vendidas por modelo. Caso base implementado.",
        },
        {
            "Línea de Negocio": "🔧 Repuestos y Accesorios",
            "Estado":           "🔄 Siguiente paso",
            "Frecuencia":       "Mensual / Semanal",
            "Algoritmo":        "SARIMAX / LSTM",
            "MAPE Esperado":    "12–18 %",
            "Sem. Implementar": 3,
            "ROI Potencial":    "Alto",
            "Disponib. Datos":  "Alta",
            "Complejidad":      "Media",
            "Descripción":      "Demanda de repuestos correlacionada con la flota activa y el historial de servicio.",
        },
        {
            "Línea de Negocio": "🛡️ Seguros de Vehículo",
            "Estado":           "📋 Planificado",
            "Frecuencia":       "Mensual",
            "Algoritmo":        "SARIMA + exog ventas",
            "MAPE Esperado":    "10–14 %",
            "Sem. Implementar": 2,
            "ROI Potencial":    "Muy Alto",
            "Disponib. Datos":  "Media",
            "Complejidad":      "Baja",
            "Descripción":      "Predicción de pólizas emitidas; fuertemente correlacionada con ventas de vehículos nuevos.",
        },
        {
            "Línea de Negocio": "🔩 Servicio Post-Venta",
            "Estado":           "📋 Planificado",
            "Frecuencia":       "Mensual",
            "Algoritmo":        "SARIMAX + flota exog",
            "MAPE Esperado":    "14–20 %",
            "Sem. Implementar": 4,
            "ROI Potencial":    "Alto",
            "Disponib. Datos":  "Media",
            "Complejidad":      "Alta",
            "Descripción":      "Órdenes de servicio y citas de taller. Exógena: tamaño de la flota activa en zona.",
        },
        {
            "Línea de Negocio": "🚙 Vehículos Usados (CPO)",
            "Estado":           "🎯 Potencial",
            "Frecuencia":       "Mensual",
            "Algoritmo":        "SARIMA + exog economía",
            "MAPE Esperado":    "15–22 %",
            "Sem. Implementar": 5,
            "ROI Potencial":    "Medio",
            "Disponib. Datos":  "Baja",
            "Complejidad":      "Alta",
            "Descripción":      "Mercado secondary con mayor volatilidad. Requiere datos de permuta y precios de mercado.",
        },
        {
            "Línea de Negocio": "💳 Financiamiento Vehicular",
            "Estado":           "🎯 Potencial",
            "Frecuencia":       "Mensual",
            "Algoritmo":        "Prophet / SARIMA",
            "MAPE Esperado":    "12–18 %",
            "Sem. Implementar": 4,
            "ROI Potencial":    "Muy Alto",
            "Disponib. Datos":  "Baja",
            "Complejidad":      "Media",
            "Descripción":      "Solicitudes de crédito y aprobaciones. Requiere integración con entidad financiera.",
        },
    ]
    df_lineas = pd.DataFrame(lineas)

    # ── Radar chart por dimensión ─────────────────────────────────────────────

    _disp_num = {"Alta": 3, "Media": 2, "Baja": 1}
    _comp_num = {"Baja": 3, "Media": 2, "Alta": 1}
    _roi_num  = {"Muy Alto": 4, "Alto": 3, "Medio": 2, "Bajo": 1}
    _sem_num  = lambda s: max(1, 5 - s // 2)

    categories = ["Disponib. datos", "ROI potencial", "Facilidad impl.",
                  "Madurez datos", "Inmediatez"]

    radar_data = []
    for _, row in df_lineas.iterrows():
        disp  = _disp_num.get(row["Disponib. Datos"], 2)
        roi   = _roi_num.get(row["ROI Potencial"], 2)
        comp  = _comp_num.get(row["Complejidad"], 2)
        sem   = max(1, 5 - row["Sem. Implementar"] // 2)
        mad   = disp
        radar_data.append([disp, roi, comp, sem, mad])

    def _to_rgba(c, a=0.08):
        if c.startswith("rgba"):
            return c
        if c.startswith("rgb"):
            return c.replace("rgb(", "rgba(").replace(")", f", {a})")
        h = c.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    fig_radar = go.Figure()
    colors_radar = [COLORS["success"], COLORS["primary"], "#38BDF8",
                    COLORS["secondary"], COLORS["purple"], COLORS["accent"]]
    for i, (_, row) in enumerate(df_lineas.iterrows()):
        vals = radar_data[i] + [radar_data[i][0]]
        cats = categories + [categories[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats,
            fill="toself", name=row["Línea de Negocio"].split(" ", 1)[1],
            line=dict(color=colors_radar[i], width=2),
            fillcolor=_to_rgba(colors_radar[i]),
            opacity=0.9,
        ))

    apply_chart_theme(fig_radar, height=420,
                      title="Dimensiones de viabilidad por línea de negocio (mayor = mejor)")
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(4,8,15,0.9)",
            radialaxis=dict(
                visible=True, range=[0, 4],
                gridcolor="rgba(0,115,255,0.12)",
                tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#3F5060"),
                tickvals=[1, 2, 3, 4],
            ),
            angularaxis=dict(
                gridcolor="rgba(0,115,255,0.12)",
                tickfont=dict(family="Rajdhani, sans-serif", size=12, color="#7A95A8"),
            ),
        ),
        showlegend=True,
        legend=dict(orientation="v", x=1.08, y=0.5),
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    # ── Tabla de líneas de negocio ────────────────────────────────────────────

    st.markdown(section_header("Resumen Ejecutivo — Líneas de Negocio", "📋"),
                unsafe_allow_html=True)
    display_lineas = ["Línea de Negocio", "Estado", "Algoritmo",
                      "MAPE Esperado", "Sem. Implementar", "ROI Potencial", "Disponib. Datos"]
    st.dataframe(df_lineas[display_lineas].reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    # ── Descripción expandida ─────────────────────────────────────────────────

    st.markdown(section_header("Detalle por Línea", "🔍"), unsafe_allow_html=True)
    for _, row in df_lineas.iterrows():
        with st.expander(f"{row['Línea de Negocio']} — {row['Estado']}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Frecuencia de datos", row["Frecuencia"])
            c2.metric("Algoritmo recomendado", row["Algoritmo"])
            c3.metric("MAPE esperado", row["MAPE Esperado"])
            c4.metric("Semanas de implementación",
                      f"{row['Sem. Implementar']} sem." if row["Sem. Implementar"] > 0 else "En producción")
            st.markdown(f"**Descripción:** {row['Descripción']}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: PLAYBOOK DE ONBOARDING
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown(section_header("Playbook de Onboarding — Nueva Marca / Modelo", "📋"),
                unsafe_allow_html=True)

    st.markdown("""
Guía paso a paso para incorporar cualquier nueva combinación marca/modelo al sistema.
El proceso está diseñado para completarse en **2 a 4 semanas** dependiendo de la disponibilidad de datos.
""")

    # ── Timeline Gantt ────────────────────────────────────────────────────────

    gantt_data = [
        # (Fase, Tarea, semana inicio, semana fin, color)
        ("Semana 1", "Extracción de datos históricos (IT + Analista)",         0, 1, COLORS["primary"]),
        ("Semana 1", "Limpieza y homologación de columnas",                    0, 1, COLORS["primary"]),
        ("Semana 2", "Validación automática (≥ 36 meses, nulos, fechas)",      1, 2, COLORS["secondary"]),
        ("Semana 2", "Primer entrenamiento SARIMA + Optuna",                   1, 2, COLORS["secondary"]),
        ("Semana 2", "Análisis ACF/PACF y diagnóstico de residuos",            1, 2, COLORS["secondary"]),
        ("Semana 3", "Walk-forward validation y calibración del horizonte",    2, 3, COLORS["success"]),
        ("Semana 3", "UAT con stakeholders (Dashboard + Proyección Ingresos)", 2, 3, COLORS["success"]),
        ("Semana 3", "Ajuste de parámetros: precio, margen, shares tiendas",   2, 3, COLORS["success"]),
        ("Semana 4", "Aprobación del modelo → activo=TRUE en Supabase",        3, 4, COLORS["accent"]),
        ("Semana 4", "Monitoreo MAPE primeros 30 días",                        3, 4, COLORS["accent"]),
        ("Semana 4", "Documentación del run (CHANGELOG + métricas)",           3, 4, COLORS["accent"]),
    ]

    fig_gantt = go.Figure()
    for i, (fase, tarea, s_ini, s_fin, color) in enumerate(gantt_data):
        fig_gantt.add_trace(go.Bar(
            x=[s_fin - s_ini],
            y=[tarea],
            base=[s_ini],
            orientation="h",
            marker=dict(color=color, opacity=0.80, line=dict(width=0)),
            name=fase,
            showlegend=False,
            hovertemplate=f"<b>{tarea}</b><br>{fase}<extra></extra>",
        ))

    apply_chart_theme(fig_gantt, height=380, title="Timeline de onboarding — nueva marca / modelo")
    fig_gantt.update_layout(
        barmode="overlay",
        xaxis=dict(
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["Inicio", "Semana 1", "Semana 2", "Semana 3", "Semana 4"],
            range=[-0.1, 4.2],
            title="",
        ),
        yaxis=dict(autorange="reversed", title=""),
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    st.plotly_chart(fig_gantt, use_container_width=True, config={"displayModeBar": False})

    # ── Requisitos de datos ───────────────────────────────────────────────────

    col_req, col_sys = st.columns(2)

    with col_req:
        st.markdown(section_header("Requisitos de Datos", "📄"), unsafe_allow_html=True)
        st.markdown("""
**Archivo Excel de ventas — obligatorio:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `FECHA-VENTA` | Fecha | Fecha de cada transacción de venta |
| `MARCA` | Texto | Nombre de la marca (p.ej. `CHERY`) |
| `MODELO3` | Texto | Nombre del modelo (p.ej. `TIGGO 2`) |

**Requisitos adicionales:**

- Mínimo **36 meses** de historial continuo (3 ciclos estacionales completos)
- Máximo **5% de datos faltantes** en las columnas requeridas
- No se requiere preprocesamiento previo — el sistema lo hace automáticamente
- Pueden cargarse **múltiples archivos Excel** si el historial está fragmentado

**Archivo opcional — variable exógena:**

- `veh_ml_features.xlsx` con columna `ventas_otros` (ventas de otros modelos de la misma marca)
- Incluida automáticamente si el coeficiente de Pearson r ≥ 0.3 con la serie objetivo
""")

    with col_sys:
        st.markdown(section_header("Configuración del Sistema", "⚙️"), unsafe_allow_html=True)
        st.markdown("""
**En Supabase PostgreSQL — tabla `training_runs`:**

No se requieren cambios de esquema. Cada entrenamiento nuevo genera un `run_name`
con timestamp único y almacena los campos `marca` y `modelo` correspondientes.

**Para soporte multi-tenant completo (1–2 días adicionales):**

```sql
-- Índice único por (marca, modelo) en lugar de global
CREATE UNIQUE INDEX idx_active_per_model
  ON training_runs (marca, modelo)
  WHERE activo = TRUE;

-- Añadir columna de namespace para Storage
ALTER TABLE training_runs
  ADD COLUMN IF NOT EXISTS storage_prefix TEXT
  GENERATED ALWAYS AS (marca || '/' || modelo) STORED;
```

**En `.streamlit/secrets.toml`:**

```toml
# No se requieren nuevas claves para marcas adicionales
# El bucket `modelos-ml` almacena todos los runs con su timestamp
```

**Pasos en la página de Entrenamiento:**

1. Subir el Excel de la nueva marca/modelo
2. Cambiar los filtros `MARCA` y `MODELO3` en la pestaña **🤖 Entrenamiento**
3. Ajustar `Fecha inicio`, `Máx. ventas` y `Horizonte`
4. Ejecutar → el sistema entrena, valida y publica automáticamente
""")

    # ── Checklist de go-live ──────────────────────────────────────────────────

    st.markdown(section_header("Checklist de Go-Live", "✅"), unsafe_allow_html=True)

    checklist_items = [
        ("Datos", "Historial ≥ 36 meses en formato Excel con columnas FECHA-VENTA, MARCA, MODELO3"),
        ("Datos", "Porcentaje de nulos < 5% en columnas requeridas"),
        ("Datos", "Test ADF pasa estacionariedad (o la serie puede diferenciarse)"),
        ("Modelo", "Optuna completó ≥ 50 trials válidos (de 80)"),
        ("Modelo", "MAPE walk-forward < 20% en período de validación"),
        ("Modelo", "Residuos sin autocorrelación significativa (ACF/PACF revisados)"),
        ("Sistema", "Artefactos subidos a Supabase Storage sin error"),
        ("Sistema", "Entrada registrada en tabla training_runs con marca y modelo correctos"),
        ("Dashboard", "Run visible en selector de versiones del Dashboard"),
        ("Dashboard", "KPIs, predicciones y walk-forward se muestran correctamente"),
        ("Financiero", "Precio unitario y margen configurados en Proyección de Ingresos"),
        ("Concesionarios", "Shares por tienda calibrados para el nuevo modelo"),
        ("Aprobación", "Stakeholder aprueba el modelo → activo=TRUE en Supabase"),
    ]

    checks_by_cat: dict = {}
    for cat, item in checklist_items:
        checks_by_cat.setdefault(cat, []).append(item)

    cat_color = {
        "Datos":         COLORS["primary"],
        "Modelo":        COLORS["success"],
        "Sistema":       COLORS["secondary"],
        "Dashboard":     "#38BDF8",
        "Financiero":    COLORS["purple"],
        "Concesionarios": COLORS["accent"],
        "Aprobación":    COLORS["secondary"],
    }
    check_cols = st.columns(len(checks_by_cat))
    for col, (cat, items) in zip(check_cols, checks_by_cat.items()):
        color = cat_color.get(cat, COLORS["primary"])
        col.markdown(
            f'<div style="border-top:2px solid {color};padding-top:10px;margin-bottom:10px">'
            f'<span style="font-family:\'Rajdhani\',sans-serif;font-size:.75rem;font-weight:700;'
            f'color:{color};text-transform:uppercase;letter-spacing:.1em">{cat}</span></div>',
            unsafe_allow_html=True,
        )
        for item in items:
            col.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.65rem;'
                f'color:#5A4F44;padding:4px 0;border-bottom:1px solid rgba(0,115,255,0.04)">'
                f'☐ {item}</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: EXPANSIÓN GEOGRÁFICA
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown(section_header("Hoja de Ruta — Expansión Geográfica LatAm", "🌎"),
                unsafe_allow_html=True)

    st.markdown("""
La arquitectura cloud-native (Streamlit Cloud + Supabase) permite desplegar instancias
independientes por mercado sin infraestructura adicional. Cada mercado opera con sus propios
datos de ventas locales y puede calibrar su modelo SARIMA de forma autónoma.
""")

    # ── Datos de mercados ─────────────────────────────────────────────────────

    mercados = pd.DataFrame([
        {"Fase": "Fase 0 — Actual",    "País": "🇵🇪 Perú",       "Distribuidor": "Interamericana Norte S.A.C.",
         "Chery Mkt. Share": "~8 %", "Datos Disponibles": "51+ meses",
         "Est. Onboarding": "Activo",    "Prioridad": 1, "Notas": "Caso base. SARIMA MAPE 10.3%"},
        {"Fase": "Fase 1 — 6–12 meses", "País": "🇨🇴 Colombia",  "Distribuidor": "Grupo Automotriz Chery CO",
         "Chery Mkt. Share": "~5 %", "Datos Disponibles": "36–48 meses",
         "Est. Onboarding": "2–3 sem.", "Prioridad": 2, "Notas": "Mercado grande, datos estructurados. Alta prioridad."},
        {"Fase": "Fase 1 — 6–12 meses", "País": "🇪🇨 Ecuador",   "Distribuidor": "Autec S.A.",
         "Chery Mkt. Share": "~9 %", "Datos Disponibles": "36+ meses",
         "Est. Onboarding": "2–3 sem.", "Prioridad": 2, "Notas": "Alta penetración Chery. Estructura de datos similar a Perú."},
        {"Fase": "Fase 2 — 12–24 meses","País": "🇧🇴 Bolivia",    "Distribuidor": "TBD",
         "Chery Mkt. Share": "~6 %", "Datos Disponibles": "24–36 meses",
         "Est. Onboarding": "3–4 sem.", "Prioridad": 3, "Notas": "Requiere evaluación de disponibilidad de datos."},
        {"Fase": "Fase 2 — 12–24 meses","País": "🇵🇾 Paraguay",   "Distribuidor": "TBD",
         "Chery Mkt. Share": "~4 %", "Datos Disponibles": "TBD",
         "Est. Onboarding": "4 sem.",   "Prioridad": 3, "Notas": "Mercado emergente. Estacionalidad propia."},
        {"Fase": "Fase 2 — 12–24 meses","País": "🇺🇾 Uruguay",    "Distribuidor": "TBD",
         "Chery Mkt. Share": "~3 %", "Datos Disponibles": "TBD",
         "Est. Onboarding": "4 sem.",   "Prioridad": 3, "Notas": "Volúmenes menores — evaluar viabilidad SARIMA."},
        {"Fase": "Fase 3 — 24+ meses",  "País": "🇨🇱 Chile",      "Distribuidor": "Automotores Gildemeister",
         "Chery Mkt. Share": "~4 %", "Datos Disponibles": "TBD",
         "Est. Onboarding": "3–4 sem.", "Prioridad": 4, "Notas": "Mercado competitivo y regulado. Alto potencial."},
        {"Fase": "Fase 3 — 24+ meses",  "País": "🇲🇽 México",     "Distribuidor": "TBD",
         "Chery Mkt. Share": "~2 %", "Datos Disponibles": "TBD",
         "Est. Onboarding": "4–6 sem.", "Prioridad": 4, "Notas": "Mercado más grande de LatAm. Complejidad alta."},
    ])

    # ── Timeline de expansión ─────────────────────────────────────────────────

    fase_colors = {
        "Fase 0 — Actual":    COLORS["success"],
        "Fase 1 — 6–12 meses": COLORS["primary"],
        "Fase 2 — 12–24 meses": COLORS["secondary"],
        "Fase 3 — 24+ meses": COLORS["purple"],
    }
    fase_ranges = {
        "Fase 0 — Actual":    (0, 1),
        "Fase 1 — 6–12 meses": (6, 12),
        "Fase 2 — 12–24 meses": (12, 24),
        "Fase 3 — 24+ meses": (24, 36),
    }

    fig_geo = go.Figure()
    for _, row in mercados.sort_values("Prioridad").iterrows():
        s_ini, s_fin = fase_ranges[row["Fase"]]
        color = fase_colors[row["Fase"]]
        fig_geo.add_trace(go.Bar(
            x=[s_fin - s_ini],
            y=[row["País"]],
            base=[s_ini],
            orientation="h",
            marker=dict(color=color, opacity=0.80),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['País']}</b><br>{row['Fase']}<br>"
                f"Distribuidor: {row['Distribuidor']}<br>"
                f"Onboarding estimado: {row['Est. Onboarding']}<br>"
                f"{row['Notas']}<extra></extra>"
            ),
        ))

    apply_chart_theme(fig_geo, height=320,
                      title="Roadmap de expansión geográfica — meses desde inicio del proyecto")
    fig_geo.update_layout(
        barmode="overlay",
        xaxis=dict(
            tickvals=[0, 6, 12, 24, 36],
            ticktext=["Hoy", "+6 meses", "+12 meses", "+24 meses", "+36 meses"],
            range=[-1, 37],
            title="",
        ),
        yaxis=dict(autorange="reversed", title=""),
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    st.plotly_chart(fig_geo, use_container_width=True, config={"displayModeBar": False})

    # ── Tabla de mercados ─────────────────────────────────────────────────────

    st.markdown(section_header("Detalle por Mercado", "🗺️"), unsafe_allow_html=True)
    display_geo = ["Fase", "País", "Distribuidor", "Chery Mkt. Share",
                   "Datos Disponibles", "Est. Onboarding", "Notas"]
    st.dataframe(mercados[display_geo].reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    # ── Factores clave de éxito ───────────────────────────────────────────────

    st.markdown(section_header("Factores Clave de Éxito para Cada Mercado", "🔑"),
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    for col, icon, titulo, puntos in [
        (f1, "📄", "Datos",
         ["Historial ≥ 36 meses de ventas por modelo",
          "Columnas homologadas (FECHA-VENTA, MARCA, MODELO3)",
          "Acceso al sistema ERP / DMS del distribuidor",
          "Actualización mensual de ventas cerradas"]),
        (f2, "🏗️", "Infraestructura",
         ["Cuenta Supabase (free tier suficiente para ≤ 3 marcas)",
          "Credenciales en Streamlit Cloud Secrets",
          "Usuario con rol 'analista' o 'admin' en el sistema",
          "Conexión a internet estable para el dashboard"]),
        (f3, "👥", "Organización",
         ["Analista de datos local que entienda el mercado",
          "Sponsor ejecutivo en el distribuidor (buy-in)",
          "Acceso a precio unitario y márgenes por modelo",
          "Proceso de actualización mensual de datos definido"]),
    ]:
        col.markdown(
            f'<div style="background:#131210;border:1px solid rgba(0,115,255,0.15);'
            f'border-radius:5px;padding:18px 16px">'
            f'<div style="font-family:\'Rajdhani\',sans-serif;font-size:.9rem;font-weight:700;'
            f'color:#C9D8E6;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px">'
            f'{icon} {titulo}</div>' +
            "".join(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.65rem;'
                f'color:#3F5060;padding:5px 0;border-bottom:1px solid rgba(0,115,255,0.04)">'
                f'→ {p}</div>'
                for p in puntos
            ) +
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Footer callout ────────────────────────────────────────────────────────

    st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="success-box">
<strong>Visión a 3 años:</strong> Una única plataforma SaaS multi-tenant con un modelo SARIMA
independiente por combinación (marca × modelo × mercado), todo gestionado desde la misma infraestructura
Supabase + Streamlit Cloud. El coste marginal de incorporar un nuevo mercado es prácticamente cero
en infraestructura — solo requiere los datos y el tiempo de onboarding del analista local.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: VISIÓN DEL PRODUCTO — REACTIVO → PROACTIVO → AUTÓNOMO
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown(section_header("Evolución del Producto — De Reactivo a Autónomo", "🚀"),
                unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(167,139,250,0.07);border:1px solid rgba(167,139,250,0.22);
            border-radius:10px;padding:14px 20px;margin-bottom:24px;">
<span style="font-size:.95rem;font-weight:600;color:#A78BFA;">
De un asistente que responde preguntas hoy, a un sistema que toma decisiones mañana.
</span><br>
<span style="color:#94A3B8;font-size:.88rem;">
El mismo pipeline SARIMA es el motor en las tres etapas — lo que evoluciona es el nivel de autonomía
del sistema y el valor que entrega al negocio.
</span>
</div>
""", unsafe_allow_html=True)

    # ── Timeline de evolución ─────────────────────────────────────────────────

    ev1, ev2, ev3 = st.columns(3)

    stage_style = (
        "border-radius:8px;padding:24px 20px;height:100%;min-height:320px;"
        "display:flex;flex-direction:column;gap:10px;"
    )
    label_style = (
        "font-family:'JetBrains Mono',monospace;font-size:.55rem;"
        "letter-spacing:.2em;text-transform:uppercase;margin-bottom:6px;"
    )
    title_style = (
        "font-family:'Rajdhani',sans-serif;font-size:1.25rem;font-weight:700;"
        "text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;"
    )
    item_style = (
        "font-family:'JetBrains Mono',monospace;font-size:.68rem;"
        "color:#7A95A8;padding:4px 0;border-bottom:1px solid rgba(0,115,255,0.06);"
    )

    with ev1:
        st.markdown(f"""
<div style="{stage_style}background:rgba(0,115,255,0.07);border:1px solid rgba(0,115,255,0.25);">
  <div>
    <div style="{label_style}color:#3F5060;">ETAPA 1 · HOY · 2025–2026</div>
    <div style="{title_style}color:#0073FF;">Sistema<br>Reactivo</div>
  </div>
  <div style="flex:1;">
    <div style="{item_style}">→ El analista carga datos mensualmente</div>
    <div style="{item_style}">→ El sistema entrena SARIMA con Optuna</div>
    <div style="{item_style}">→ Dashboard muestra predicción + IC 95%</div>
    <div style="{item_style}">→ Recomendación de compra conservadora/agresiva</div>
    <div style="{item_style}">→ Asistente IA responde preguntas del equipo</div>
    <div style="{item_style}">→ Proyección financiera en USD</div>
  </div>
  <div style="margin-top:12px;padding:10px 12px;background:rgba(0,115,255,0.1);
              border-radius:5px;font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#0073FF;">
    🎯 Estado: <strong>EN PRODUCCIÓN</strong><br>
    MAPE objetivo &lt; 15% · 1 marca · 1 modelo
  </div>
</div>
""", unsafe_allow_html=True)

    with ev2:
        st.markdown(f"""
<div style="{stage_style}background:rgba(194,255,0,0.05);border:1px solid rgba(194,255,0,0.22);">
  <div>
    <div style="{label_style}color:#3F5060;">ETAPA 2 · AÑO 1 · 2026–2027</div>
    <div style="{title_style}color:#C2FF00;">Sistema<br>Proactivo</div>
  </div>
  <div style="flex:1;">
    <div style="{item_style}">→ Monitoreo automático de MAPE (data drift)</div>
    <div style="{item_style}">→ Re-entrenamiento automático cuando MAPE &gt; umbral</div>
    <div style="{item_style}">→ Alertas push al equipo (email / Slack)</div>
    <div style="{item_style}">→ Pipeline multi-marca: Chery + JAC + BYD</div>
    <div style="{item_style}">→ Integración con ERP / DMS del distribuidor</div>
    <div style="{item_style}">→ Recomendación de pedido enviada al proveedor</div>
  </div>
  <div style="margin-top:12px;padding:10px 12px;background:rgba(194,255,0,0.08);
              border-radius:5px;font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#C2FF00;">
    🎯 Estado: <strong>EN ROADMAP</strong><br>
    4–6 marcas · Alertas automáticas · LatAm Fase 1
  </div>
</div>
""", unsafe_allow_html=True)

    with ev3:
        st.markdown(f"""
<div style="{stage_style}background:rgba(0,245,160,0.05);border:1px solid rgba(0,245,160,0.2);">
  <div>
    <div style="{label_style}color:#3F5060;">ETAPA 3 · AÑO 2+ · 2027–2028</div>
    <div style="{title_style}color:#00F5A0;">Sistema<br>Autónomo</div>
  </div>
  <div style="flex:1;">
    <div style="{item_style}">→ Plataforma SaaS multi-tenant (marca × mercado)</div>
    <div style="{item_style}">→ Modelos independientes por SKU y concesionario</div>
    <div style="{item_style}">→ Optimización de precios dinámica (price elasticity)</div>
    <div style="{item_style}">→ Gestión autónoma de stock de seguridad</div>
    <div style="{item_style}">→ Integración financiera (forecast de cash flow)</div>
    <div style="{item_style}">→ API pública para distribuidores LatAm</div>
  </div>
  <div style="margin-top:12px;padding:10px 12px;background:rgba(0,245,160,0.08);
              border-radius:5px;font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#00F5A0;">
    🎯 Estado: <strong>VISIÓN ESTRATÉGICA</strong><br>
    Portafolio completo LatAm · SaaS B2B
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Gráfico de valor por etapa ────────────────────────────────────────────

    st.markdown('<div style="margin-top:28px"></div>', unsafe_allow_html=True)
    st.markdown(section_header("Valor de Negocio por Etapa de Evolución", "📈"),
                unsafe_allow_html=True)

    etapas        = ["Reactivo<br>(Hoy)", "Proactivo<br>(Año 1)", "Autónomo<br>(Año 2+)"]
    valor_negocio = [30, 65, 100]
    autonomia     = [15, 55, 95]
    marcas_cub    = [1, 6, 20]

    fig_ev = go.Figure()
    fig_ev.add_trace(go.Bar(
        name="Valor de negocio (índice)",
        x=etapas, y=valor_negocio,
        marker=dict(color=[COLORS['primary'], COLORS['secondary'], COLORS['success']],
                    opacity=0.85),
        text=[f"{v}" for v in valor_negocio],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=12, color="#7A95A8"),
    ))
    fig_ev.add_trace(go.Scatter(
        name="Autonomía del sistema (%)",
        x=etapas, y=autonomia,
        mode="lines+markers",
        line=dict(color=COLORS['purple'], width=2.5, dash="dot"),
        marker=dict(size=10, color=COLORS['purple'], symbol="diamond",
                    line=dict(color="#080D18", width=1.5)),
        yaxis="y2",
    ))
    apply_chart_theme(fig_ev, height=360,
                      title="Evolución del valor de negocio y autonomía del sistema")
    fig_ev.update_layout(
        barmode="group",
        yaxis=dict(title="Valor de negocio (índice 0–100)", ticksuffix=""),
        yaxis2=dict(title="Autonomía (%)", overlaying="y", side="right",
                    range=[0, 110], ticksuffix="%",
                    gridcolor="rgba(167,139,250,0.06)",
                    tickfont=dict(family="JetBrains Mono, monospace",
                                  color="#A78BFA", size=11)),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_ev, use_container_width=True, config={"displayModeBar": False})

    # ── Callout final ─────────────────────────────────────────────────────────

    st.markdown("""
<div class="success-box" style="margin-top:24px;">
<strong>El mismo modelo SARIMA que hoy predice el Tiggo 2 en Norte Perú</strong>
es la base del sistema que mañana gestionará el portafolio completo de
Interamericana — y pasado mañana, el de cualquier distribuidor automotriz de LatAm.
El código no cambia; cambia el nivel de integración y autonomía que el negocio
decide activar.
</div>
""", unsafe_allow_html=True)
