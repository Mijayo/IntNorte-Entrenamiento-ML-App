"""
============================================================================
PÁGINA: PROYECCIÓN DE INGRESOS
============================================================================
Traduce la predicción SARIMA del modelo activo en cifras financieras en USD.
Inputs configurables: precio unitario, margen neto y tipo de cambio.
Accesible para roles: admin, analista y financiero (permiso ver_ingresos).

Extraída del Dashboard (tab 3) en 2026-05-27 como página independiente.
============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import core.supabase_io as sio
from core.auth_system import (init_session_state, show_login_page, show_user_info,
                              check_session_timeout, has_permission, show_header)
from core.styles import kpi_card, section_header, apply_chart_theme, COLORS

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Proyección Ingresos — TIGGO 2", page_icon="💰",
    layout="wide", initial_sidebar_state="expanded"
)

# ── Auth ──────────────────────────────────────────────────────────────────────

init_session_state()

if check_session_timeout():
    st.warning("⏱️ Tu sesión ha expirado.")
    st.stop()
if not st.session_state.authenticated:
    show_login_page("💰 Proyección de Ingresos — TIGGO 2")
    st.stop()

if not has_permission('ver_ingresos'):
    st.error("🔒 Acceso restringido — Esta página está disponible sólo para **Admin**, **Analista** y **Financiero**.")
    st.stop()

# ── Selector de versión (sidebar) ─────────────────────────────────────────────

available_runs = sio.get_available_runs()

if not available_runs:
    st.error("❌ No hay modelos entrenados. Ejecuta primero la app de **Entrenamiento**.")
    st.stop()

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

# ── Cargar datos ──────────────────────────────────────────────────────────────

with st.spinner('Cargando datos del modelo...'):
    metricas, pred_total, _gs, _wf, _hist = sio.load_precargados(selected_run)

# ── Header ────────────────────────────────────────────────────────────────────

show_header(
    "Proyección de Ingresos — TIGGO 2",
    f"Escenario financiero basado en predicción SARIMA  |  Modelo: {sio.format_run_label(selected_run)} {'🟢' if is_latest else '🔵'}"
)
show_user_info()

# ── Contenido ────────────────────────────────────────────────────────────────

st.markdown(section_header("Proyección de Ingresos · Horizonte 6 Meses", "💰"), unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.2);
            border-radius:10px;padding:14px 20px;margin-bottom:18px;">
<span style="font-size:1.0rem;font-weight:600;color:#4ADE80;">Escenario financiero basado en predicción SARIMA</span><br>
<span style="color:#94A3B8;font-size:0.9rem;">
Los ingresos se calculan mes a mes a partir de la predicción central del modelo.
El intervalo de confianza al 95% define el rango pesimista–optimista.
Ajusta el precio unitario y el margen neto para tu escenario real.
</span>
</div>
""", unsafe_allow_html=True)

# ── Inputs de escenario ───────────────────────────────────────────────────────

inp_col1, inp_col2, inp_col3 = st.columns(3)
with inp_col1:
    precio_usd = st.number_input(
        "Precio medio por unidad (USD $)",
        min_value=1_000, max_value=500_000, value=27_000, step=500,
        format="%d",
        help="Precio de venta neto por unidad en dólares.",
    )
with inp_col2:
    margen_usd_pct = st.number_input(
        "Margen neto estimado (%)",
        min_value=0.0, max_value=100.0, value=8.0, step=0.5,
        format="%.1f",
        help="Porcentaje de beneficio neto sobre ingresos. 0 = omitir.",
    )
with inp_col3:
    tc = st.number_input(
        "Tipo de cambio (USD / moneda local)",
        min_value=0.01, max_value=10000.0, value=1.0, step=0.01,
        format="%.2f",
        help="Factor para convertir el precio a moneda local si aplica. Deja en 1 si ya está en USD.",
    )

# ── Cálculo de ingresos ───────────────────────────────────────────────────────

df_usd = pred_total[['Mes', 'Predicción', 'IC_Inferior', 'IC_Superior']].copy()
precio_efectivo      = precio_usd * tc
df_usd['Ingresos ($)'] = (df_usd['Predicción'] * precio_efectivo).round(0).astype(int)
df_usd['IC Inf ($)']   = (df_usd['IC_Inferior'] * precio_efectivo).round(0).astype(int)
df_usd['IC Sup ($)']   = (df_usd['IC_Superior'] * precio_efectivo).round(0).astype(int)
if margen_usd_pct > 0:
    df_usd['Beneficio ($)'] = (df_usd['Ingresos ($)'] * margen_usd_pct / 100).round(0).astype(int)

total_uds_usd  = int(df_usd['Predicción'].sum())
total_ing_usd  = int(df_usd['Ingresos ($)'].sum())
ic_inf_usd     = int(df_usd['IC Inf ($)'].sum())
ic_sup_usd     = int(df_usd['IC Sup ($)'].sum())

# ── KPIs ──────────────────────────────────────────────────────────────────────

ku1, ku2, ku3 = st.columns(3)
ku1.markdown(kpi_card("Unidades (6 meses)", f"{total_uds_usd:,} uds", "📦", "blue"), unsafe_allow_html=True)
ku2.markdown(kpi_card("Ingresos centrales (6 m)", f"${total_ing_usd:,.0f}", "💵"), unsafe_allow_html=True)
ku3.markdown(kpi_card("Rango IC 95% (6 m)", f"${ic_inf_usd:,.0f} – ${ic_sup_usd:,.0f}", "📐", "amber"), unsafe_allow_html=True)

if margen_usd_pct > 0:
    total_ben_usd = int(df_usd['Beneficio ($)'].sum())
    kb1, kb2, _ = st.columns(3)
    kb1.markdown(kpi_card("Beneficio neto (6 m)", f"${total_ben_usd:,.0f}", "💹"), unsafe_allow_html=True)
    kb2.markdown(kpi_card("Margen aplicado", f"{margen_usd_pct:.1f}%", "📊"), unsafe_allow_html=True)

# ── Gráfico: barras de ingresos con rango IC ──────────────────────────────────

fig_rev = go.Figure()

# Banda IC (área semitransparente en overlay)
fig_rev.add_trace(go.Bar(
    x=df_usd['Mes'], y=df_usd['IC Sup ($)'] - df_usd['IC Inf ($)'],
    base=df_usd['IC Inf ($)'],
    name='Rango IC 95%',
    marker=dict(color='rgba(251,191,36,0.18)', line=dict(width=0)),
    hovertemplate='IC 95%: $%{base:,.0f} – $%{customdata:,.0f}<extra></extra>',
    customdata=df_usd['IC Sup ($)'],
))

# Barras de ingresos centrales
fig_rev.add_trace(go.Bar(
    x=df_usd['Mes'], y=df_usd['Ingresos ($)'],
    name='Ingresos proyectados',
    marker=dict(
        color=COLORS['primary'],
        line=dict(color=COLORS['primary'], width=0),
    ),
    text=[f"${v:,.0f}" for v in df_usd['Ingresos ($)']],
    textposition='outside',
    textfont=dict(color='#94A3B8', size=11),
    hovertemplate='%{x}<br>Ingresos: $%{y:,.0f}<extra></extra>',
))

# Línea de beneficio neto (si aplica)
if margen_usd_pct > 0:
    fig_rev.add_trace(go.Scatter(
        x=df_usd['Mes'], y=df_usd['Beneficio ($)'],
        mode='lines+markers', name='Beneficio neto',
        line=dict(color=COLORS['accent'], width=2.5, dash='dot'),
        marker=dict(size=8, color=COLORS['accent'], symbol='diamond',
                    line=dict(color='#080D18', width=1.5)),
        hovertemplate='%{x}<br>Beneficio: $%{y:,.0f}<extra></extra>',
    ))

apply_chart_theme(fig_rev, height=480,
                  title='Proyección de Ingresos en USD — Horizonte 6 Meses')
fig_rev.update_layout(
    barmode='overlay',
    hovermode='x unified',
    xaxis_title='Mes',
    yaxis_title='USD ($)',
    yaxis_tickprefix='$',
    yaxis_tickformat=',.0f',
)
st.plotly_chart(fig_rev, use_container_width=True, config={'displayModeBar': False})

# ── Tabla detallada ───────────────────────────────────────────────────────────

st.subheader("📋 Detalle mensual")
disp_cols_usd = ['Mes', 'Predicción', 'Ingresos ($)', 'IC Inf ($)', 'IC Sup ($)']
fmt_usd = {
    'Predicción':  '{:.0f}',
    'Ingresos ($)': '${:,}',
    'IC Inf ($)':  '${:,}',
    'IC Sup ($)':  '${:,}',
}
if margen_usd_pct > 0:
    disp_cols_usd.append('Beneficio ($)')
    fmt_usd['Beneficio ($)'] = '${:,}'

# Fila de totales
totals_row = {
    'Mes': 'TOTAL',
    'Predicción': df_usd['Predicción'].sum(),
    'Ingresos ($)': total_ing_usd,
    'IC Inf ($)': ic_inf_usd,
    'IC Sup ($)': ic_sup_usd,
}
if margen_usd_pct > 0:
    totals_row['Beneficio ($)'] = total_ben_usd

df_usd_show = pd.concat(
    [df_usd[disp_cols_usd], pd.DataFrame([totals_row])[disp_cols_usd]],
    ignore_index=True
)

st.dataframe(
    df_usd_show.style
               .background_gradient(subset=['Ingresos ($)'], cmap='Greens')
               .format(fmt_usd),
    use_container_width=True, hide_index=True,
)

if has_permission('exportar'):
    csv_usd = df_usd[disp_cols_usd].to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Exportar CSV proyección USD",
        csv_usd,
        f"proyeccion_ingresos_usd_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="app-footer">Sistema TIGGO 2 &nbsp;·&nbsp; ISDI &nbsp;·&nbsp; Predicción de Demanda</div>',
    unsafe_allow_html=True,
)
