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
from core.auth_system import (guard_page, show_user_info, show_header, has_permission)
from core.styles import kpi_card, section_header, apply_chart_theme, COLORS

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Proyección Ingresos — TIGGO 2", page_icon="💰",
    layout="wide", initial_sidebar_state="expanded"
)

# ── Auth ──────────────────────────────────────────────────────────────────────

guard_page("💰 Proyección de Ingresos — TIGGO 2", permission="ver_ingresos")

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
    metricas, pred_total, _gs, _wf, _hist, _exog = sio.load_precargados(selected_run)

# ── Header ────────────────────────────────────────────────────────────────────

show_header(
    "Proyección de Ingresos — TIGGO 2",
    f"Escenario financiero basado en predicción SARIMA  |  Modelo: {sio.format_run_label(selected_run)} {'🟢' if is_latest else '🔵'}"
)
show_user_info()

# ── Pestañas ──────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["💰 Proyección de Ingresos", "💎 Valor Estratégico del Sistema"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PROYECCIÓN DE INGRESOS
# ══════════════════════════════════════════════════════════════════════════════

with tab1:

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

    # ── Inputs de escenario ───────────────────────────────────────────────────

    inp_col1, inp_col2, inp_col3 = st.columns(3)
    with inp_col1:
        precio_usd = st.number_input(
            "Precio medio por unidad (USD $)",
            min_value=1_000, max_value=500_000, value=15_000, step=500,
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

    # ── Cálculo de ingresos ───────────────────────────────────────────────────

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

    # ── KPIs ──────────────────────────────────────────────────────────────────

    ku1, ku2, ku3 = st.columns(3)
    ku1.markdown(kpi_card("Unidades (6 meses)", f"{total_uds_usd:,} uds", "📦", "blue"), unsafe_allow_html=True)
    ku2.markdown(kpi_card("Ingresos centrales (6 m)", f"${total_ing_usd:,.0f}", "💵"), unsafe_allow_html=True)
    ku3.markdown(kpi_card("Rango IC 95% (6 m)", f"${ic_inf_usd:,.0f} – ${ic_sup_usd:,.0f}", "📐", "amber"), unsafe_allow_html=True)

    if margen_usd_pct > 0:
        total_ben_usd = int(df_usd['Beneficio ($)'].sum())
        kb1, kb2, _ = st.columns(3)
        kb1.markdown(kpi_card("Beneficio neto (6 m)", f"${total_ben_usd:,.0f}", "💹"), unsafe_allow_html=True)
        kb2.markdown(kpi_card("Margen aplicado", f"{margen_usd_pct:.1f}%", "📊"), unsafe_allow_html=True)

    # ── Gráfico: barras de ingresos con rango IC ──────────────────────────────

    fig_rev = go.Figure()

    fig_rev.add_trace(go.Bar(
        x=df_usd['Mes'], y=df_usd['IC Sup ($)'] - df_usd['IC Inf ($)'],
        base=df_usd['IC Inf ($)'],
        name='Rango IC 95%',
        marker=dict(color='rgba(251,191,36,0.18)', line=dict(width=0)),
        hovertemplate='IC 95%: $%{base:,.0f} – $%{customdata:,.0f}<extra></extra>',
        customdata=df_usd['IC Sup ($)'],
    ))

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

    # ── Tabla detallada ───────────────────────────────────────────────────────

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
        _xls_proy = sio.build_proyeccion_excel(df_usd, disp_cols_usd)
        st.download_button(
            "📥 Exportar Excel proyección de ingresos",
            _xls_proy,
            f"proyeccion_ingresos_{datetime.now().strftime('%Y%m%d')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VALOR ESTRATÉGICO DEL SISTEMA
# ══════════════════════════════════════════════════════════════════════════════

with tab2:

    st.markdown(section_header("Valor Estratégico del Sistema — ¿Cuánto vale predecir bien?", "💎"),
                unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(167,139,250,0.07);border:1px solid rgba(167,139,250,0.22);
            border-radius:10px;padding:14px 20px;margin-bottom:20px;">
<span style="font-size:.95rem;font-weight:600;color:#A78BFA;">Cuantificación del impacto económico del sistema de predicción</span><br>
<span style="color:#94A3B8;font-size:0.88rem;">
Ajusta los parámetros de negocio para calcular el ahorro anual estimado vs. gestión sin predicción.
</span>
</div>
""", unsafe_allow_html=True)

    roi_c1, roi_c2, roi_c3 = st.columns(3)
    with roi_c1:
        st.markdown("**Parámetros de inventario**")
        sobrestock_actual = st.number_input(
            "Sobrestock promedio sin predicción (uds/mes)",
            min_value=0, max_value=50, value=5, step=1,
            help="Unidades extra compradas por encima de la demanda real, por mes, sin sistema de predicción.",
        )
        costo_fin_pct = st.number_input(
            "Costo mensual de capital inmovilizado (% del precio/ud)",
            min_value=0.1, max_value=5.0, value=0.7, step=0.1, format="%.1f",
            help=(
                "Costo financiero mensual por unidad en inventario. "
                "Chery: tasa referencial 8% anual = 0.67%/mes. "
                "Primeros 60 días en stock son libres de interés — el costo aplica desde el día 61."
            ),
        )
    with roi_c2:
        st.markdown("**Parámetros de venta perdida**")
        stockout_mes = st.number_input(
            "Ventas perdidas por mes sin predicción (uds/mes)",
            min_value=0, max_value=20, value=2, step=1,
            help="Unidades que no se vendieron por quiebre de stock estimado sin el sistema.",
        )
        reduccion_stockout_pct = st.number_input(
            "Reducción de stockout con el sistema (%)",
            min_value=0, max_value=100, value=70, step=5,
            help="Porcentaje de stockouts evitados gracias a la predicción (estimación conservadora: 60–80%).",
        )
    with roi_c3:
        st.markdown("**Parámetros del sistema**")
        reduccion_sobrestock_pct = st.number_input(
            "Reducción de sobrestock con el sistema (%)",
            min_value=0, max_value=100, value=60, step=5,
            help="Porcentaje de unidades sobrestock evitadas con mejor previsión (estimación: 50–70%).",
        )
        costo_sistema_anual = st.number_input(
            "Costo anual del sistema (USD $)",
            min_value=0, max_value=50_000, value=1_200, step=100,
            help="Coste total anual: Streamlit Cloud + Supabase + mantenimiento (~$100/mes).",
        )

    # ── Cálculos ──────────────────────────────────────────────────────────────

    costo_por_ud_mes = precio_usd * (costo_fin_pct / 100)

    ahorro_sobrestock_anual = (
        sobrestock_actual
        * (reduccion_sobrestock_pct / 100)
        * costo_por_ud_mes
        * 12
    )

    margen_por_ud = precio_usd * (margen_usd_pct / 100)
    ahorro_stockout_anual = (
        stockout_mes
        * (reduccion_stockout_pct / 100)
        * margen_por_ud
        * 12
    )

    valor_bruto_anual = ahorro_sobrestock_anual + ahorro_stockout_anual
    roi_neto = valor_bruto_anual - costo_sistema_anual
    roi_ratio = (valor_bruto_anual / costo_sistema_anual) if costo_sistema_anual > 0 else float('inf')
    payback_meses = (costo_sistema_anual / (valor_bruto_anual / 12)) if valor_bruto_anual > 0 else float('inf')

    # ── KPIs del ROI ──────────────────────────────────────────────────────────

    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(kpi_card("Ahorro sobrestock/año",
                         f"${ahorro_sobrestock_anual:,.0f}", "📦"), unsafe_allow_html=True)
    r2.markdown(kpi_card("Ingresos recuperados/año",
                         f"${ahorro_stockout_anual:,.0f}", "💹", "green"), unsafe_allow_html=True)
    r3.markdown(kpi_card("Valor neto anual del sistema",
                         f"${roi_neto:,.0f}", "💎", "blue"), unsafe_allow_html=True)
    r4.markdown(kpi_card("ROI del sistema",
                         f"{roi_ratio:.0f}x", "🚀", "amber"), unsafe_allow_html=True)

    # ── Gráfico waterfall ─────────────────────────────────────────────────────

    fig_roi = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "total", "relative", "total"],
        x=["Ahorro<br>sobrestock", "Ingresos<br>recuperados",
           "Valor bruto", "Costo<br>del sistema", "Valor neto"],
        y=[ahorro_sobrestock_anual, ahorro_stockout_anual, 0,
           -costo_sistema_anual, 0],
        text=[f"${ahorro_sobrestock_anual:,.0f}", f"${ahorro_stockout_anual:,.0f}",
              f"${valor_bruto_anual:,.0f}", f"-${costo_sistema_anual:,.0f}",
              f"${roi_neto:,.0f}"],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=12, color="#7A95A8"),
        connector=dict(line=dict(color="rgba(0,115,255,0.25)", width=1.5)),
        increasing=dict(marker=dict(color=COLORS['success'])),
        decreasing=dict(marker=dict(color=COLORS['accent'])),
        totals=dict(marker=dict(color=COLORS['primary'])),
        opacity=0.85,
    ))
    apply_chart_theme(fig_roi, height=400, title="Valor anual del sistema — Waterfall ($USD)")
    fig_roi.update_layout(yaxis_title="USD ($)", yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    st.plotly_chart(fig_roi, use_container_width=True, config={"displayModeBar": False})

    # ── Tabla resumen ─────────────────────────────────────────────────────────

    col_ta, col_tb = st.columns(2)
    with col_ta:
        st.markdown(f"""
<div style="background:#0D1117;border:1px solid rgba(0,245,160,0.2);border-radius:8px;padding:18px 20px;">
<div style="font-family:'Rajdhani',sans-serif;font-size:.75rem;font-weight:700;
            color:#00F5A0;text-transform:uppercase;letter-spacing:.12em;margin-bottom:12px;">
✅ Con sistema de predicción
</div>
<table style="width:100%;font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#94A3B8;border-collapse:collapse;">
<tr><td style="padding:5px 0;border-bottom:1px solid rgba(0,115,255,0.06)">Sobrestock reducido</td>
    <td style="text-align:right;color:#00F5A0;">{sobrestock_actual * reduccion_sobrestock_pct/100:.1f} uds/mes</td></tr>
<tr><td style="padding:5px 0;border-bottom:1px solid rgba(0,115,255,0.06)">Stockouts evitados</td>
    <td style="text-align:right;color:#00F5A0;">{stockout_mes * reduccion_stockout_pct/100:.1f} uds/mes</td></tr>
<tr><td style="padding:5px 0;border-bottom:1px solid rgba(0,115,255,0.06)">Ahorro capital inmovilizado</td>
    <td style="text-align:right;color:#00F5A0;">${ahorro_sobrestock_anual:,.0f}/año</td></tr>
<tr><td style="padding:5px 0;border-bottom:1px solid rgba(0,115,255,0.06)">Ingresos recuperados</td>
    <td style="text-align:right;color:#00F5A0;">${ahorro_stockout_anual:,.0f}/año</td></tr>
<tr style="font-weight:700"><td style="padding:8px 0;color:#C9D8E6">Payback del sistema</td>
    <td style="text-align:right;color:#C2FF00;">{payback_meses:.1f} meses</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    with col_tb:
        st.markdown(f"""
<div style="background:#0D1117;border:1px solid rgba(255,58,92,0.2);border-radius:8px;padding:18px 20px;">
<div style="font-family:'Rajdhani',sans-serif;font-size:.75rem;font-weight:700;
            color:#FF3A5C;text-transform:uppercase;letter-spacing:.12em;margin-bottom:12px;">
❌ Sin sistema de predicción
</div>
<table style="width:100%;font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#94A3B8;border-collapse:collapse;">
<tr><td style="padding:5px 0;border-bottom:1px solid rgba(0,115,255,0.06)">Sobrestock mensual</td>
    <td style="text-align:right;color:#FF3A5C;">{sobrestock_actual} uds/mes</td></tr>
<tr><td style="padding:5px 0;border-bottom:1px solid rgba(0,115,255,0.06)">Stockouts mensuales</td>
    <td style="text-align:right;color:#FF3A5C;">{stockout_mes} uds/mes</td></tr>
<tr><td style="padding:5px 0;border-bottom:1px solid rgba(0,115,255,0.06)">Capital inmovilizado/año</td>
    <td style="text-align:right;color:#FF3A5C;">${sobrestock_actual * costo_por_ud_mes * 12:,.0f}</td></tr>
<tr><td style="padding:5px 0;border-bottom:1px solid rgba(0,115,255,0.06)">Margen perdido/año</td>
    <td style="text-align:right;color:#FF3A5C;">${stockout_mes * margen_por_ud * 12:,.0f}</td></tr>
<tr style="font-weight:700"><td style="padding:8px 0;color:#C9D8E6">Coste total ineficiencia</td>
    <td style="text-align:right;color:#FF3A5C;">${(sobrestock_actual * costo_por_ud_mes + stockout_mes * margen_por_ud) * 12:,.0f}/año</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="app-footer">Sistema TIGGO 2 &nbsp;·&nbsp; ISDI &nbsp;·&nbsp; Predicción de Demanda</div>',
    unsafe_allow_html=True,
)
