"""
Demo Académica — Interamericana Norte · Chery Tiggo 2
Sistema de Predicción de Demanda (ISDI · Troncal)
Datos de ejemplo pre-cargados. Sin autenticación. URL pública.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Demo · TIGGO 2 · Interamericana Norte",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons+Round');

/* ── Design tokens ────────────────────────────────────────────────────── */
:root {
  --c-bg:      #04080F;
  --c-surface: #070C18;
  --c-raised:  #0A1020;
  --c-border:  rgba(0,224,255,0.13);
  --c-cyan:    #00E0FF;
  --c-gold:    #FFC107;
  --c-red:     #FF3A5C;
  --c-green:   #00F5A0;
  --c-purple:  #A78BFA;
  --c-text:    #C9D8E6;
  --c-muted:   #3F5060;
}

/* ── Base (mobile-first) ──────────────────────────────────────────────── */
html, body {
  overflow-x: hidden !important;
  max-width: 100vw !important;
}
html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
  font-family: 'Rajdhani', sans-serif !important;
  background-color: var(--c-bg) !important;
}
[data-testid="block-container"] {
  padding: 0.75rem 0.85rem !important;
  max-width: 100% !important;
}

/* ── Typography ───────────────────────────────────────────────────────── */
h1 {
  font-family:'Rajdhani',sans-serif!important; font-weight:700!important;
  letter-spacing:.05em!important; text-transform:uppercase!important;
  font-size: clamp(.95rem, 4.5vw, 1.35rem) !important;
}
h2,h3 { font-family:'Rajdhani',sans-serif!important; font-weight:600!important; letter-spacing:.04em!important; }

/* ── Tabs: horizontal scroll on small screens ─────────────────────────── */
[data-baseweb="tab-list"] {
  overflow-x: auto !important;
  flex-wrap: nowrap !important;
  scrollbar-width: none !important;
  -webkit-overflow-scrolling: touch;
  padding-bottom: 2px;
  gap: 2px !important;
}
[data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family:'Rajdhani',sans-serif!important; font-weight:600!important;
  font-size:.78rem!important; letter-spacing:.06em!important; text-transform:uppercase!important;
  white-space: nowrap !important;
  flex-shrink: 0 !important;
  padding: 8px 10px !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
  border-radius:5px!important; border:1px solid var(--c-border)!important;
  padding:10px!important; background:var(--c-surface)!important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  font-family:'Rajdhani',sans-serif!important; font-weight:700!important;
  letter-spacing:.08em!important; text-transform:uppercase!important;
  font-size:.74rem!important; color:var(--c-muted)!important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family:'JetBrains Mono',monospace!important; color:var(--c-cyan)!important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  border-right:1px solid rgba(0,224,255,0.07)!important;
  background-color:var(--c-surface)!important;
}
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(.material-symbols-rounded):not(.material-icons),
[data-testid="stSidebar"] label { font-family:'Rajdhani',sans-serif!important; }

/* ── Columns: mobile-first stacking ──────────────────────────────────── */
/* Mobile (<640px): full width stack */
[data-testid="stHorizontalBlock"] {
  flex-wrap: wrap !important;
  gap: 6px !important;
  row-gap: 6px !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
  min-width: 100% !important;
  flex: 1 1 100% !important;
  box-sizing: border-box !important;
}

/* ── KPI card — mobile base ───────────────────────────────────────────── */
.kpi-card {
  background:linear-gradient(150deg,var(--c-surface) 0%,var(--c-raised) 100%);
  border:1px solid var(--c-border); border-radius:5px;
  padding:12px 12px 10px; text-align:center; position:relative; overflow:hidden;
  box-shadow:0 2px 18px rgba(0,0,0,.55); margin-bottom:4px;
}
.kpi-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--c-cyan),rgba(0,224,255,.15));
  box-shadow:0 1px 8px rgba(0,224,255,.25);
}
.kpi-icon { font-size:1rem; margin-bottom:6px; opacity:.55; }
.kpi-label {
  font-size:.58rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--c-muted); margin-bottom:6px;
  font-family:'Rajdhani',sans-serif; font-weight:700;
}
.kpi-label::before{content:'[ ';opacity:.5;} .kpi-label::after{content:' ]';opacity:.5;}
.kpi-value {
  font-size: clamp(1.3rem, 5vw, 2.25rem);
  font-weight:400; font-family:'JetBrains Mono',monospace;
  color:var(--c-cyan); line-height:1.1;
  text-shadow:0 0 22px rgba(0,224,255,.28);
}
.kpi-value.amber { color:var(--c-gold); text-shadow:0 0 22px rgba(255,193,7,.24); }
.kpi-value.blue  { color:#38BDF8;        text-shadow:0 0 22px rgba(56,189,248,.22); }
.kpi-value.green { color:var(--c-green); text-shadow:0 0 22px rgba(0,245,160,.22); }
.kpi-value.red   { color:var(--c-red);   text-shadow:0 0 22px rgba(255,58,92,.24); }
.kpi-sub { font-size:.65rem; color:var(--c-muted); margin-top:5px; font-family:'JetBrains Mono',monospace; }

/* ── Section headers ─────────────────────────────────────────────────── */
.section-header {
  display:flex; align-items:center; gap:10px;
  margin:20px 0 12px; padding-bottom:8px;
  border-bottom:1px solid rgba(0,224,255,0.06);
}
.section-header-bar {
  width:3px; height:14px; flex-shrink:0;
  background:linear-gradient(180deg,var(--c-cyan),rgba(0,224,255,.2));
  border-radius:2px; box-shadow:0 0 10px rgba(0,224,255,.35);
}
.section-header-text {
  font-family:'Rajdhani',sans-serif; font-size:.82rem; font-weight:700;
  color:var(--c-text); letter-spacing:.11em; text-transform:uppercase;
}

/* ── Callout boxes ───────────────────────────────────────────────────── */
.winner-box {
  background:rgba(255,193,7,.06); border-left:4px solid var(--c-gold);
  border-radius:0 5px 5px 0; padding:12px 14px; margin:10px 0;
  font-size:.92rem; font-weight:700; color:var(--c-gold);
  font-family:'Rajdhani',sans-serif; letter-spacing:.04em; text-transform:uppercase;
}
.success-box {
  background:rgba(0,245,160,.04); border-left:3px solid var(--c-green);
  border-radius:0 4px 4px 0; padding:10px 14px; margin:8px 0;
  font-family:'Rajdhani',sans-serif; font-size:.88rem; color:var(--c-text);
}
.info-box {
  background:rgba(0,224,255,.04); border-left:3px solid var(--c-cyan);
  border-radius:0 4px 4px 0; padding:10px 14px; margin:8px 0;
  font-family:'Rajdhani',sans-serif; font-size:.88rem; color:var(--c-text);
}

/* ── Misc ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { font-family:'JetBrains Mono',monospace!important; font-size:.75rem!important; }
code,pre { font-family:'JetBrains Mono',monospace!important; }
.app-footer {
  text-align:center; padding:18px 0 8px;
  font-size:.62rem; color:#1A2838;
  font-family:'JetBrains Mono',monospace;
  border-top:1px solid rgba(0,224,255,.05); margin-top:28px; letter-spacing:.07em;
}
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
[data-testid="stDecoration"]{display:none!important;}

/* ── Tablet (≥640px): 2-column KPI grids ─────────────────────────────── */
@media (min-width: 640px) {
  [data-testid="block-container"] {
    padding: 1.25rem 1.5rem !important;
  }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    min-width: calc(50% - 6px) !important;
    flex: 1 1 calc(50% - 6px) !important;
  }
  .kpi-card { padding: 16px 14px 13px; }
  [data-testid="stTabs"] [data-baseweb="tab"] {
    font-size:.82rem!important;
    padding: 10px 14px !important;
  }
}

/* ── Desktop (≥1024px): restore original multi-column layout ──────────── */
@media (min-width: 1024px) {
  [data-testid="block-container"] {
    padding: 1.5rem 2.5rem !important;
  }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    min-width: 0 !important;
    flex: 1 1 0 !important;
  }
  .kpi-card { padding: 20px 16px 16px; }
  [data-testid="stTabs"] [data-baseweb="tab"] {
    font-size:.84rem!important;
    padding: 10px 16px !important;
  }
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

COLORS = {
    'primary': '#00E0FF',
    'accent':  '#FFC107',
    'red':     '#FF3A5C',
    'green':   '#00F5A0',
    'series':  ['#00E0FF', '#FFC107', '#00F5A0', '#FF3A5C', '#A78BFA'],
}

def kpi(label, value, icon='', color='', sub=''):
    val_cls = f'kpi-value {color}' if color else 'kpi-value'
    icon_h  = f'<div class="kpi-icon">{icon}</div>' if icon else ''
    sub_h   = f'<div class="kpi-sub">{sub}</div>' if sub else ''
    return f"""<div class="kpi-card">{icon_h}
<div class="kpi-label">{label}</div>
<div class="{val_cls}">{value}</div>{sub_h}</div>"""

def sec(text, icon=''):
    i = f'<span style="margin-right:6px;opacity:.55">{icon}</span>' if icon else ''
    return f"""<div class="section-header">
<div class="section-header-bar"></div>
<span class="section-header-text">{i}{text}</span></div>"""

def theme(fig, h=None, title=None):
    layout = dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(4,8,15,0.92)',
        font=dict(family='Rajdhani, sans-serif', color='#7A95A8', size=11),
        xaxis=dict(gridcolor='rgba(0,224,255,0.07)', showgrid=True, zeroline=False,
                   tickfont=dict(family='JetBrains Mono, monospace', color='#3F5060', size=10)),
        yaxis=dict(gridcolor='rgba(0,224,255,0.07)', showgrid=True, zeroline=False,
                   tickfont=dict(family='JetBrains Mono, monospace', color='#3F5060', size=10)),
        hoverlabel=dict(bgcolor='#070C18', font=dict(family='Rajdhani, sans-serif', color='#C9D8E6', size=13),
                        bordercolor='rgba(0,224,255,0.3)'),
        legend=dict(bgcolor='rgba(4,8,15,0.85)', bordercolor='rgba(0,224,255,0.12)',
                    borderwidth=1, font=dict(family='Rajdhani, sans-serif', color='#7A95A8', size=11),
                    orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=12, r=12, t=44, b=24),
        autosize=True,
    )
    if h:     layout['height'] = h
    if title: layout['title']  = dict(text=title,
                                       font=dict(family='Rajdhani, sans-serif', color='#C9D8E6', size=14),
                                       x=0, xanchor='left', pad=dict(l=4))
    fig.update_layout(**layout)
    return fig

# ── Datos pre-cargados ────────────────────────────────────────────────────────

# Histórico mensual: Jan 2021 – Dic 2024 (48 meses)
FECHAS_HIST = pd.date_range('2021-01-01', periods=48, freq='MS')
VENTAS_HIST = [
    16, 19, 22, 18, 17, 20, 19, 21, 23, 19, 22, 28,  # 2021
    20, 23, 27, 22, 21, 25, 23, 26, 28, 23, 27, 34,  # 2022
    23, 27, 31, 25, 24, 28, 26, 29, 31, 26, 30, 38,  # 2023
    25, 28, 32, 27, 25, 29, 27, 30, 32, 27, 31, 37,  # 2024
]

hist = pd.Series(VENTAS_HIST, index=FECHAS_HIST, name='Ventas')

# Predicciones: Jan 2025 – Jun 2025
FECHAS_PRED = pd.date_range('2025-01-01', periods=6, freq='MS')
PRED        = [28, 32, 35, 29, 27, 33]
IC_INF      = [22, 25, 28, 23, 21, 26]
IC_SUP      = [34, 39, 42, 35, 33, 40]

pred = pd.DataFrame({
    'Fecha':       FECHAS_PRED,
    'Mes':         [d.strftime('%b %Y') for d in FECHAS_PRED],
    'Predicción':  PRED,
    'IC_Inferior': IC_INF,
    'IC_Superior': IC_SUP,
})
pred['IC_Amplitud'] = pred['IC_Superior'] - pred['IC_Inferior']
pred['Ingreso_Est'] = [p * PRECIO_UNITARIO for p in PRED]

# Walk-forward validation: Ene 2024 – Dic 2024
FECHAS_WF  = pd.date_range('2024-01-01', periods=12, freq='MS')
WF_REAL    = [25, 28, 32, 27, 25, 29, 27, 30, 32, 27, 31, 37]
WF_PRED    = [23.2, 26.8, 30.1, 25.8, 24.0, 27.5, 25.4, 28.6, 30.5, 27.2, 29.6, 35.3]

wf = pd.DataFrame({
    'fecha':      FECHAS_WF,
    'real':       WF_REAL,
    'prediccion': WF_PRED,
})
wf['error_abs'] = abs(wf['real'] - wf['prediccion'])
wf['error_pct'] = wf['error_abs'] / wf['real'] * 100
MAPE = wf['error_pct'].mean()

# Métricas de calidad derivadas
ss_res  = ((wf['real'] - wf['prediccion']) ** 2).sum()
ss_tot  = ((wf['real'] - wf['real'].mean()) ** 2).sum()
R2      = 1 - ss_res / ss_tot
RMSE_WF = np.sqrt(ss_res / len(wf))
MAE_WF  = wf['error_abs'].mean()

# Crecimiento anual compuesto (CAGR 2021→2024)
hist_df_yr = pd.Series(VENTAS_HIST, index=FECHAS_HIST)
ventas_2021 = hist_df_yr[hist_df_yr.index.year == 2021].sum()
ventas_2024 = hist_df_yr[hist_df_yr.index.year == 2024].sum()
CAGR = (ventas_2024 / ventas_2021) ** (1/3) - 1

# Precio de referencia (MXN)
PRECIO_UNITARIO = 350_000

# Comparativa de modelos ML
MODELOS = ['SARIMA', 'Prophet', 'XGBoost', 'Random Forest', 'Reg. Lineal']
MAPE_ML = [MAPE,      11.2,      14.5,      16.3,            22.1]
MAE_ML  = [MAE_WF,    2.8,       3.7,        4.1,             5.6]
RMSE_ML = [RMSE_WF,   3.5,       4.6,        5.1,             6.9]
R2_ML   = [R2,        0.61,      0.48,       0.43,            0.22]

df_modelos = pd.DataFrame({
    'Modelo': MODELOS,
    'MAPE %': MAPE_ML,
    'MAE':    MAE_ML,
    'RMSE':   RMSE_ML,
    'R²':     R2_ML,
}).sort_values('MAPE %')

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
<div style="padding:18px 0 10px;border-bottom:1px solid rgba(0,224,255,0.08);margin-bottom:16px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3F5060;letter-spacing:.12em;margin-bottom:4px;">
    DEMO ACADÉMICA · ISDI
  </div>
  <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.1rem;color:#C9D8E6;letter-spacing:.08em;text-transform:uppercase;">
    Interamericana Norte
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#3F5060;margin-top:3px;">
    Chery Tiggo 2 · Predicción de demanda
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="font-family:'Rajdhani',sans-serif;font-size:.75rem;color:#3F5060;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">
  Parámetros del Modelo
</div>""", unsafe_allow_html=True)
    st.code("SARIMA(1,1,1)(1,1,1)[12]\nAIC: 234.1\nBIC: 248.3", language=None)

    st.markdown("""
<div style="font-family:'Rajdhani',sans-serif;font-size:.75rem;color:#3F5060;letter-spacing:.1em;text-transform:uppercase;margin:16px 0 8px;">
  Cobertura de datos
</div>""", unsafe_allow_html=True)
    st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#7A95A8;line-height:2;">
  Histórico: Ene 2021 – Dic 2024<br>
  Meses: 48<br>
  Horizonte: 6 meses<br>
  Validación: Walk-forward<br>
  Optimización: Optuna TPE
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Datos de ejemplo para demostración académica. No representan cifras reales de la empresa.")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<div style="padding:10px 0 16px;margin-bottom:14px;border-bottom:1px solid rgba(0,224,255,0.07);">
  <h1 style="margin:0!important;padding:0!important;color:#C9D8E6;letter-spacing:.07em;">
    Sistema de Predicción de Demanda — Tiggo 2
  </h1>
  <div style="font-family:'JetBrains Mono',monospace;font-size:clamp(.58rem,2vw,.68rem);
              color:#3F5060;margin-top:5px;letter-spacing:.04em;">
    Interamericana Norte &nbsp;·&nbsp; ISDI Troncal &nbsp;·&nbsp; Demo Académica
  </div>
</div>""", unsafe_allow_html=True)

# ── KPIs globales ─────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi("Total Ventas",      f"{sum(VENTAS_HIST):,} uds",
                "📦", sub=f"2021–2024 · ${sum(VENTAS_HIST)*PRECIO_UNITARIO/1e6:.0f}M MXN"),
            unsafe_allow_html=True)
c2.markdown(kpi("CAGR Ventas",       f"+{CAGR*100:.1f}%",
                "📈", "blue", sub="Crecim. anual compuesto 2021→2024"),
            unsafe_allow_html=True)
c3.markdown(kpi("MAPE Walk-Forward", f"{MAPE:.1f}%",
                "🎯", "green", sub=f"vs 20% baseline · R²={R2:.2f}"),
            unsafe_allow_html=True)
c4.markdown(kpi("Próximo mes",       f"{PRED[0]} uds",
                "🔮", "amber", sub=f"IC 95%: {IC_INF[0]}–{IC_SUP[0]} uds"),
            unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs(["📋 Resumen del Proyecto", "📊 Histórico", "🔮 Predicción", "🔄 Validación del Modelo", "🏆 Comparativa ML"])

# ══ Tab 2: Histórico ══════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown(sec("Serie Temporal — Ventas Mensuales Tiggo 2", "📊"), unsafe_allow_html=True)

    yoy_2324 = (ventas_2024 - hist_df_yr[hist_df_yr.index.year == 2023].sum()) / \
               hist_df_yr[hist_df_yr.index.year == 2023].sum() * 100
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Promedio mensual",  f"{hist.mean():.1f} uds",
              delta=f"+{CAGR*100:.1f}% CAGR")
    h2.metric("Máximo histórico",  f"{hist.max():.0f} uds",
              delta=f"Dic 2023")
    h3.metric("Mínimo histórico",  f"{hist.min():.0f} uds",
              delta="Ene 2021", delta_color="off")
    h4.metric("Crecim. 2023→2024", f"{yoy_2324:+.1f}%",
              delta=f"{ventas_2024-hist_df_yr[hist_df_yr.index.year==2023].sum():.0f} uds más")

    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode='lines+markers', name='Ventas Mensuales',
        line=dict(color=COLORS['primary'], width=2.5),
        marker=dict(size=5, color=COLORS['primary']),
        fill='tozeroy', fillcolor='rgba(0,224,255,0.05)',
    ))
    fig_h.add_hline(
        y=hist.mean(), line_dash='dot', line_color=COLORS['accent'],
        annotation_text=f"Media: {hist.mean():.1f}",
        annotation_position="top right",
        annotation_font_color=COLORS['accent'],
    )
    theme(fig_h, h=360, title='Ventas Mensuales — Tiggo 2 · Interamericana Norte')
    fig_h.update_layout(hovermode='x unified', xaxis_title='Fecha', yaxis_title='Unidades')
    st.plotly_chart(fig_h, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

    with st.expander("📊 Estadísticas descriptivas por año"):
        hist_df = hist.to_frame('Ventas')
        hist_df['Año'] = hist_df.index.year
        resumen_anual = hist_df.groupby('Año')['Ventas'].agg(
            Total='sum', Promedio='mean', Máximo='max', Mínimo='min'
        ).round(1)
        resumen_anual['Ingreso MXN'] = (resumen_anual['Total'] * PRECIO_UNITARIO / 1e6).round(1)
        resumen_anual['Crec. YoY'] = resumen_anual['Total'].pct_change().mul(100).round(1)
        st.dataframe(
            resumen_anual.style
                .bar(subset=['Total'], color='#4a90d9')
                .format({'Promedio': '{:.1f}', 'Ingreso MXN': '${:.1f}M',
                         'Crec. YoY': '{:+.1f}%', 'Total': '{:.0f}',
                         'Máximo': '{:.0f}', 'Mínimo': '{:.0f}'}),
            use_container_width=True
        )

# ══ Tab 3: Predicción ════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown(sec("Predicción Enero – Junio 2025", "🔮"), unsafe_allow_html=True)

    ingreso_6m = sum(PRED) * PRECIO_UNITARIO
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(kpi("Próximo mes",      f"{PRED[0]} uds",
                    "🔮", sub=f"IC 95%: {IC_INF[0]}–{IC_SUP[0]}"),
                unsafe_allow_html=True)
    p2.markdown(kpi("Total horizonte",  f"{sum(PRED)} uds",
                    "📦", "blue", sub="Ene–Jun 2025"),
                unsafe_allow_html=True)
    p3.markdown(kpi("Promedio mensual", f"{sum(PRED)/len(PRED):.1f} uds",
                    "📊", "amber", sub=f"vs {hist.mean():.1f} hist."),
                unsafe_allow_html=True)
    p4.markdown(kpi("Ingreso estimado", f"${ingreso_6m/1e6:.1f}M",
                    "💵", "green", sub="MXN · 6 meses"),
                unsafe_allow_html=True)

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode='lines', name='Histórico',
        line=dict(color=COLORS['primary'], width=2),
    ))
    fig_p.add_trace(go.Scatter(
        x=pred['Fecha'], y=pred['Predicción'],
        mode='lines+markers', name='Predicción SARIMA',
        line=dict(color=COLORS['accent'], width=2.5),
        marker=dict(size=9, symbol='circle', color=COLORS['accent'],
                    line=dict(color='#080D18', width=1.5)),
    ))
    fig_p.add_trace(go.Scatter(
        x=pred['Fecha'].tolist() + pred['Fecha'].tolist()[::-1],
        y=pred['IC_Superior'].tolist() + pred['IC_Inferior'].tolist()[::-1],
        fill='toself', fillcolor='rgba(255,193,7,0.08)',
        line=dict(color='rgba(0,0,0,0)'), name='IC 95%',
    ))
    fig_p.add_shape(
        type='line',
        x0=hist.index[-1], x1=hist.index[-1], y0=0, y1=1, yref='paper',
        line=dict(color='rgba(100,116,139,0.6)', width=1.5, dash='dot'),
    )
    theme(fig_p, h=400, title='Histórico + Predicción — Tiggo 2 · Interamericana Norte')
    fig_p.update_layout(hovermode='x unified', xaxis_title='Fecha', yaxis_title='Unidades')
    st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

    st.subheader("📋 Tabla de predicciones")
    pred_show = pred[['Mes', 'Predicción', 'IC_Inferior', 'IC_Superior',
                       'IC_Amplitud', 'Ingreso_Est']].copy()
    pred_show.columns = ['Mes', 'Pred.', 'IC Inf', 'IC Sup', 'Amplitud IC', 'Ingreso MXN']
    st.dataframe(
        pred_show.style
            .bar(subset=['Pred.'], color='#4a90d9')
            .bar(subset=['Amplitud IC'], color='rgba(255,193,7,0.35)')
            .format({'Pred.': '{:.0f}', 'IC Inf': '{:.0f}', 'IC Sup': '{:.0f}',
                     'Amplitud IC': '{:.0f}', 'Ingreso MXN': '${:,.0f}'}),
        use_container_width=True, hide_index=True,
    )

    # Recomendación de compra
    proximo = PRED[0]
    ic_inf_p, ic_sup_p = IC_INF[0], IC_SUP[0]
    prom_hist = hist.mean()
    tendencia = ((hist.iloc[-3:].mean() - prom_hist) / prom_hist) * 100

    st.markdown(sec("Recomendación de Compra", "💼"), unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(f"""<div class="info-box">
<strong>Estrategia Conservadora</strong><br>
Comprar: <strong>{int(ic_sup_p * 1.05)} unidades</strong><br>
Basado en IC superior + 5% · Minimiza riesgo de rotura de stock
</div>""", unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""<div class="success-box">
<strong>Estrategia Recomendada</strong><br>
Comprar: <strong>{int(proximo * 1.10)} unidades</strong><br>
Predicción central + 10% buffer · Equilibrio demanda / inventario
</div>""", unsafe_allow_html=True)

    if tendencia > 5:
        st.success(f"Tendencia CRECIENTE en los últimos 3 meses: +{tendencia:.1f}% vs. promedio histórico.")
    elif tendencia < -5:
        st.warning(f"Tendencia DECRECIENTE en los últimos 3 meses: {tendencia:.1f}% vs. promedio histórico.")
    else:
        st.info(f"Tendencia ESTABLE: {tendencia:+.1f}% vs. promedio histórico de {prom_hist:.1f} uds/mes.")

# ══ Tab 4: Validación ════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown(sec("Walk-Forward Validation — 2024", "🔄"), unsafe_allow_html=True)

    v1, v2, v3, v4 = st.columns(4)
    v1.markdown(kpi("MAPE",    f"{MAPE:.1f}%",    "🎯", "green",
                    sub="vs 20% baseline"),               unsafe_allow_html=True)
    v2.markdown(kpi("R²",      f"{R2:.3f}",        "📐", "blue",
                    sub=f"MAE={MAE_WF:.1f} · RMSE={RMSE_WF:.1f}"), unsafe_allow_html=True)
    v3.markdown(kpi("Mejor mes",f"{wf['error_pct'].min():.1f}%", "✅",
                    sub=wf.loc[wf['error_pct'].idxmin(),'fecha'].strftime('%b %Y')), unsafe_allow_html=True)
    v4.markdown(kpi("Peor mes", f"{wf['error_pct'].max():.1f}%", "⚠️", "amber",
                    sub=wf.loc[wf['error_pct'].idxmax(),'fecha'].strftime('%b %Y')), unsafe_allow_html=True)

    fig_wf = go.Figure()
    fig_wf.add_trace(go.Scatter(
        x=wf['fecha'], y=wf['real'],
        mode='lines+markers', name='Real',
        line=dict(color=COLORS['primary'], width=2.5),
        marker=dict(size=7, color=COLORS['primary']),
    ))
    fig_wf.add_trace(go.Scatter(
        x=wf['fecha'], y=wf['prediccion'],
        mode='lines+markers', name='Predicción (walk-forward)',
        line=dict(color=COLORS['accent'], width=2.5, dash='dot'),
        marker=dict(size=7, color=COLORS['accent'], symbol='diamond'),
    ))
    theme(fig_wf, h=360, title='Walk-Forward Validation — Real vs. Predicción (2024)')
    fig_wf.update_layout(hovermode='x unified', xaxis_title='Mes', yaxis_title='Unidades')
    st.plotly_chart(fig_wf, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

    # Error % por mes — barras
    err_colors = [COLORS['red'] if e > MAPE else COLORS['green'] for e in wf['error_pct']]
    fig_err = go.Figure(go.Bar(
        x=wf['fecha'].dt.strftime('%b'),
        y=wf['error_pct'],
        marker_color=err_colors,
        text=[f"{v:.1f}%" for v in wf['error_pct']],
        textposition='outside',
        textfont=dict(family='JetBrains Mono, monospace', color='#94A3B8', size=10),
    ))
    fig_err.add_hline(y=MAPE, line_dash='dot', line_color=COLORS['accent'],
                      annotation_text=f"MAPE={MAPE:.1f}%",
                      annotation_font_color=COLORS['accent'])
    theme(fig_err, h=280, title='Error % por mes — verde=bajo promedio, rojo=sobre promedio')
    fig_err.update_layout(showlegend=False, yaxis_title='Error %', yaxis=dict(range=[0, wf['error_pct'].max()*1.3]))
    st.plotly_chart(fig_err, use_container_width=True, config={'displayModeBar': False})

    wf_show = wf.copy()
    wf_show['fecha'] = wf_show['fecha'].dt.strftime('%B %Y')
    wf_show.columns = ['Mes', 'Real', 'Predicción', 'Error Abs.', 'Error %']
    st.dataframe(
        wf_show.style
               .bar(subset=['Error %'], color='#e05c5c')
               .format({'Real': '{:.0f}', 'Predicción': '{:.1f}',
                        'Error Abs.': '{:.2f}', 'Error %': '{:.2f}%'}),
        use_container_width=True, hide_index=True,
    )

    mae_uds = wf['error_abs'].mean()
    if MAPE <= 10:
        st.markdown(f"""<div class="success-box">
MAPE = {MAPE:.1f}% · MAE = <strong>{mae_uds:.1f} unidades</strong> — El modelo se equivoca en promedio
<strong>{mae_uds:.1f} unidades por mes</strong> sobre ventas reales de ~{wf['real'].mean():.0f} uds/mes.
Error medio inferior al 10%: apto para planificación de pedidos y compromisos de inventario.
</div>""", unsafe_allow_html=True)

# ══ Tab 5: Comparativa ML ════════════════════════════════════════════════════

with tabs[4]:
    st.markdown(sec("Comparativa de 5 Modelos — Mismo Histórico", "🏆"), unsafe_allow_html=True)

    st.markdown(f"""<div class="winner-box">
    SARIMA seleccionado — MAPE: {MAPE:.1f}% · R²: {R2:.3f} · MAE: {MAE_WF:.1f} uds
    &nbsp;·&nbsp; Optuna TPE · 150 combinaciones evaluadas
</div>""", unsafe_allow_html=True)

    # Gráfico de barras MAPE
    colors_bar = [COLORS['accent'] if m == 'SARIMA' else COLORS['series'][2]
                  for m in df_modelos['Modelo']]
    fig_cmp = go.Figure(go.Bar(
        x=df_modelos['Modelo'], y=df_modelos['MAPE %'],
        marker_color=colors_bar,
        text=[f"{v:.1f}%" for v in df_modelos['MAPE %']],
        textposition='outside',
        textfont=dict(family='JetBrains Mono, monospace', color='#94A3B8', size=11),
    ))
    theme(fig_cmp, h=340, title='MAPE por Modelo — menor es mejor')
    fig_cmp.update_layout(xaxis_title='', yaxis_title='MAPE (%)',
                          showlegend=False, yaxis=dict(range=[0, 28]))
    st.plotly_chart(fig_cmp, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

    # Tabla completa
    st.subheader("📋 Tabla de métricas")
    st.dataframe(
        df_modelos.style
                  .bar(subset=['MAPE %'], color='#e05c5c')
                  .bar(subset=['R²'], color='rgba(0,245,160,0.4)')
                  .format({'MAPE %': '{:.1f}%', 'MAE': '{:.2f}',
                           'RMSE': '{:.2f}', 'R²': '{:.3f}'})
                  .set_properties(**{'font-family': 'JetBrains Mono, monospace'}),
        use_container_width=True, hide_index=True,
    )

    # Radar — 4 ejes
    categorias = ['MAPE', 'MAE', 'RMSE', 'R²']

    def normalizar(vals, higher_is_better=False):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [1.0] * len(vals)
        if higher_is_better:
            return [(v - mn) / (mx - mn) for v in vals]
        return [1 - (v - mn) / (mx - mn) for v in vals]

    n_mape = normalizar(MAPE_ML)
    n_mae  = normalizar(MAE_ML)
    n_rmse = normalizar(RMSE_ML)
    n_r2   = normalizar(R2_ML, higher_is_better=True)

    fig_r = go.Figure()
    for i, mod in enumerate(MODELOS):
        vals = [n_mape[i], n_mae[i], n_rmse[i], n_r2[i]]
        fig_r.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categorias + [categorias[0]],
            fill='toself' if mod == 'SARIMA' else 'none',
            fillcolor='rgba(255,193,7,0.1)' if mod == 'SARIMA' else 'rgba(0,0,0,0)',
            name=mod,
            line=dict(color=COLORS['series'][i % len(COLORS['series'])],
                      width=3 if mod == 'SARIMA' else 1.5),
        ))
    theme(fig_r, h=360, title='Desempeño relativo por modelo (mayor = mejor)')
    fig_r.update_layout(polar=dict(
        bgcolor='rgba(4,8,15,0.9)',
        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                        gridcolor='rgba(0,224,255,0.08)'),
        angularaxis=dict(gridcolor='rgba(0,224,255,0.08)',
                         tickfont=dict(family='Rajdhani, sans-serif', color='#7A95A8', size=13)),
    ))
    st.plotly_chart(fig_r, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

# ══ Tab 1: Resumen del Proyecto ═══════════════════════════════════════════════

with tabs[0]:
    st.markdown(sec("El Problema de Negocio", "🎯"), unsafe_allow_html=True)
    st.markdown("""
Interamericana Norte necesita **anticipar la demanda mensual del Chery Tiggo 2** para:

- Planificar órdenes de compra al fabricante con 2–3 meses de anticipación
- Reducir el costo de inmovilización de inventario
- Evitar roturas de stock que impactan la satisfacción del cliente
- Asignar unidades entre concesionarios de forma eficiente

Sin un sistema predictivo, las decisiones se basaban en criterio subjetivo del equipo comercial, con errores de estimación superiores al 20%.
""")

    st.markdown(sec("La Solución Desarrollada", "💡"), unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown("""
**Modelo SARIMA**
- Serie temporal con estacionalidad
- Parámetros optimizados con Optuna TPE
- 150 combinaciones evaluadas
- Validación walk-forward 12 meses
""")
    with col_s2:
        st.markdown("""
**Sistema completo** *(producción)*
- Interfaz web para usuarios no técnicos
- Roles: Admin · Analista · Gerente
- Historial de entrenamientos con versionado
- Audit log automático de predicciones
""")
    with col_s3:
        st.markdown("""
**Tecnología**
- Streamlit (interfaz)
- statsmodels (SARIMA)
- Supabase (datos + auth)
- Google Gemini (asistente IA)
""")

    st.markdown(sec("Resultados Clave", "📈"), unsafe_allow_html=True)

    reduccion_error = (1 - MAPE/20) * 100
    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(kpi("MAPE obtenido",    f"{MAPE:.1f}%",
                    "🎯", "green", sub=f"R²={R2:.3f} · MAE={MAE_WF:.1f} uds"), unsafe_allow_html=True)
    r2.markdown(kpi("Reducción error",  f"{reduccion_error:.0f}%",
                    "📉", "amber", sub="vs baseline subjetivo 20%"),            unsafe_allow_html=True)
    r3.markdown(kpi("Mejor modelo",     "SARIMA",
                    "🏆", sub="vs 4 alternativas evaluadas"),                   unsafe_allow_html=True)
    r4.markdown(kpi("Horizonte pred.",  "6 meses",
                    "🔮", "blue", sub="Ene–Jun 2025"),                          unsafe_allow_html=True)

    st.markdown(sec("Impacto Económico Estimado", "💰"), unsafe_allow_html=True)

    COSTO_CAPITAL      = 0.15
    error_baseline_pct = 0.20
    error_modelo_pct   = MAPE / 100
    prom_ventas        = hist.mean()
    unidades_error_antes  = prom_ventas * error_baseline_pct
    unidades_error_modelo = prom_ventas * error_modelo_pct
    ahorro_unidades_mes   = unidades_error_antes - unidades_error_modelo
    capital_inmovilizado  = ahorro_unidades_mes * PRECIO_UNITARIO
    ahorro_pesos_mes      = capital_inmovilizado * (COSTO_CAPITAL / 12)

    e1, e2, e3, e4 = st.columns(4)
    e1.markdown(kpi("Ahorro inventario/mes",
                    f"${ahorro_pesos_mes:,.0f}",
                    "💵", "green", sub="costo de capital liberado"), unsafe_allow_html=True)
    e2.markdown(kpi("Ahorro anual estimado",
                    f"${ahorro_pesos_mes*12/1e6:.2f}M",
                    "📈", "amber", sub="MXN · proyección 12 meses"), unsafe_allow_html=True)
    e3.markdown(kpi("Unidades rescatadas/mes",
                    f"{ahorro_unidades_mes:.1f}",
                    "📦", sub=f"de {unidades_error_antes:.1f} → {unidades_error_modelo:.1f} uds error"), unsafe_allow_html=True)
    e4.markdown(kpi("Capital liberado/mes",
                    f"${capital_inmovilizado/1e6:.1f}M",
                    "🏦", "blue", sub="MXN desinmovilizados"), unsafe_allow_html=True)

    st.markdown(f"""<div class="info-box">
<strong>Supuestos del cálculo:</strong> precio unitario Tiggo 2 = $350,000 MXN &nbsp;·&nbsp;
costo de capital = 15% anual &nbsp;·&nbsp; error baseline (criterio subjetivo) = 20% &nbsp;·&nbsp;
MAPE modelo = {MAPE:.1f}%. El ahorro mensual representa la reducción en costo financiero del capital
inmovilizado por unidades sobrecompradas frente al método anterior.
</div>""", unsafe_allow_html=True)

    st.markdown(sec("Flujo de Trabajo", "🔄"), unsafe_allow_html=True)
    st.markdown("""
```
[Excel ventas] → [Limpieza + validación] → [Test ADF estacionariedad]
       ↓
[Búsqueda Optuna: 150 combinaciones SARIMA]
       ↓
[Walk-forward validation: 12 meses]
       ↓
[Aprobación del modelo → Publicación en Dashboard]
       ↓
[Gerente consulta predicciones + Asistente IA]
```
""")

    st.markdown(sec("Stack Tecnológico", "⚙️"), unsafe_allow_html=True)
    tech_cols = st.columns(5)
    techs = [
        ("Streamlit", "Interfaz web"),
        ("statsmodels", "Modelo SARIMA"),
        ("Optuna", "Optimización"),
        ("Supabase", "DB + Auth + Storage"),
        ("Gemini", "Asistente IA"),
    ]
    for i, (name, desc) in enumerate(techs):
        with tech_cols[i]:
            st.markdown(f"""
<div style="background:rgba(0,224,255,.04);border:1px solid rgba(0,224,255,.1);
            border-radius:5px;padding:14px;text-align:center;">
  <div style="font-family:'Rajdhani',sans-serif;font-weight:700;color:#C9D8E6;
              font-size:.95rem;letter-spacing:.06em;text-transform:uppercase;">
    {name}
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#3F5060;
              margin-top:4px;">
    {desc}
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown(sec("Limitaciones y Próximos Pasos", "⚠️"), unsafe_allow_html=True)

    lim1, lim2 = st.columns(2)
    with lim1:
        st.markdown("""<div class="info-box">
<strong>Limitaciones actuales</strong><br>
· 48 meses de histórico — mínimo recomendado para SARIMA estacional<br>
· El modelo asume que los patrones de estacionalidad se mantienen estables<br>
· No incorpora variables externas (tipo de cambio, precio combustible, competencia)<br>
· Validación sobre un solo año completo (2024)
</div>""", unsafe_allow_html=True)
    with lim2:
        st.markdown("""<div class="success-box">
<strong>Próximos pasos</strong><br>
· Incorporar variables exógenas (SARIMAX) para capturar choques externos<br>
· Ampliar validación a hold-out de 18 meses<br>
· Desagregar predicción por punto de venta / concesionario<br>
· Implementar reentrenamiento automático mensual con nuevos datos
</div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="app-footer">'
    'Sistema TIGGO 2 &nbsp;·&nbsp; Interamericana Norte &nbsp;·&nbsp; '
    'ISDI Troncal &nbsp;·&nbsp; Demo Académica &nbsp;·&nbsp; Datos de ejemplo'
    '</div>',
    unsafe_allow_html=True,
)
