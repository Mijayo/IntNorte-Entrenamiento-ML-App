"""
============================================================================
MÓDULO DE ESTILOS GLOBALES — Sistema TIGGO 2
Tema: Telemetría / Racing Data Dashboard
Fuentes: Rajdhani (display) + JetBrains Mono (datos)
============================================================================
"""

# ── Paleta de colores ─────────────────────────────────────────────────────────

COLORS = {
    'primary':    '#0073FF',   # vibrant orange (acción principal)
    'secondary':  '#C2FF00',   # lime-yellow
    'accent':     '#FF3A5C',   # signal red
    'success':    '#00F5A0',   # neon green
    'purple':     '#A78BFA',
    'text':       '#C9D8E6',
    'muted':      '#3F5060',
    'border':     'rgba(0,115,255,0.14)',
    # Series para gráficos
    'series': ['#0073FF', '#C2FF00', '#00F5A0', '#FF3A5C',
               '#A78BFA', '#F97316', '#38BDF8', '#FB7185'],
}


# ── Tema Plotly ───────────────────────────────────────────────────────────────

def apply_chart_theme(fig, height=None, title=None):
    """Aplica el tema oscuro telemetría a cualquier figura Plotly."""
    layout = dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(4,8,15,0.92)',
        font=dict(family='Rajdhani, sans-serif', color='#7A95A8', size=12),
        xaxis=dict(
            gridcolor='rgba(0,115,255,0.07)',
            showgrid=True, zeroline=False,
            tickfont=dict(family='JetBrains Mono, monospace', color='#3F5060', size=11),
            title_font=dict(family='Rajdhani, sans-serif', color='#7A95A8'),
        ),
        yaxis=dict(
            gridcolor='rgba(0,115,255,0.07)',
            showgrid=True, zeroline=False,
            tickfont=dict(family='JetBrains Mono, monospace', color='#3F5060', size=11),
            title_font=dict(family='Rajdhani, sans-serif', color='#7A95A8'),
        ),
        hoverlabel=dict(
            bgcolor='#070C18',
            font=dict(family='Rajdhani, sans-serif', color='#C9D8E6', size=14),
            bordercolor='rgba(0,115,255,0.3)',
        ),
        legend=dict(
            bgcolor='rgba(4,8,15,0.85)',
            bordercolor='rgba(0,115,255,0.12)',
            borderwidth=1,
            font=dict(family='Rajdhani, sans-serif', color='#7A95A8', size=12),
        ),
        margin=dict(l=20, r=20, t=50, b=30),
    )
    if height:
        layout['height'] = height
    if title:
        layout['title'] = dict(
            text=title,
            font=dict(family='Rajdhani, sans-serif', color='#C9D8E6', size=16),
            x=0, xanchor='left', pad=dict(l=4),
        )
    fig.update_layout(**layout)
    return fig


# ── Helpers HTML ──────────────────────────────────────────────────────────────

def kpi_card(label, value, icon='', color_class='', sub=''):
    """Tarjeta KPI estilo telemetría — monoespaciado + glow."""
    sub_html  = f'<div class="kpi-sub">{sub}</div>' if sub else ''
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ''
    val_cls   = f'kpi-value {color_class}' if color_class else 'kpi-value'
    return f"""
<div class="kpi-card">
  {icon_html}
  <div class="kpi-label">{label}</div>
  <div class="{val_cls}">{value}</div>
  {sub_html}
</div>"""


def section_header(text, icon=''):
    """Encabezado de sección estilo panel técnico."""
    icon_html = f'<span style="margin-right:6px;opacity:.55">{icon}</span>' if icon else ''
    return f"""
<div class="section-header">
  <div class="section-header-bar"></div>
  <span class="section-header-text">{icon_html}{text}</span>
</div>"""


# ── CSS Global ────────────────────────────────────────────────────────────────

def get_global_css():
    """CSS completo — tema Telemetría (inyectar con st.markdown)."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,700;1,300&display=swap');

:root {
  --c-bg:      #0D0C0A;
  --c-surface: #131210;
  --c-raised:  #1A1815;
  --c-border:  rgba(0,115,255,0.13);
  --c-cyan:    #0073FF;
  --c-gold:    #C2FF00;
  --c-red:     #FF3A5C;
  --c-green:   #00F5A0;
  --c-purple:  #A78BFA;
  --c-text:    #E8E0D8;
  --c-muted:   #5A4F44;
  --c-dim:     #2A2018;
}

/* ── Base ─────────────────────────────────────────────────── */
html, body, [data-testid="stApp"] {
  font-family: 'Rajdhani', sans-serif !important;
  background-color: var(--c-bg) !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
  background-color: var(--c-bg) !important;
}
h1 {
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: .06em !important;
  text-transform: uppercase !important;
}
h2, h3 {
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: .04em !important;
}

/* ── Tabs ──────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.84rem !important;
  letter-spacing: .07em !important;
  text-transform: uppercase !important;
}

/* ── Native metric containers ─────────────────────────────── */
[data-testid="metric-container"] {
  border-radius: 5px !important;
  border: 1px solid var(--c-border) !important;
  padding: 10px !important;
  background: var(--c-surface) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: .08em !important;
  text-transform: uppercase !important;
  font-size: .74rem !important;
  color: var(--c-muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 400 !important;
  color: var(--c-cyan) !important;
}

/* ── Buttons ───────────────────────────────────────────────── */
[data-testid="stButton"]>button,
[data-testid="stFormSubmitButton"]>button {
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: .1em !important;
  text-transform: uppercase !important;
  border-radius: 4px !important;
  font-size: .88rem !important;
}

/* ── Input labels ──────────────────────────────────────────── */
[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stDateInput"] label,
[data-testid="stCheckbox"] label {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: .76rem !important;
  letter-spacing: .09em !important;
  text-transform: uppercase !important;
  color: var(--c-muted) !important;
  font-weight: 600 !important;
}

/* ── Sidebar ───────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  border-right: 1px solid rgba(0,115,255,0.07) !important;
  background-color: var(--c-surface) !important;
}
[data-testid="stSidebar"] * {
  font-family: 'Rajdhani', sans-serif !important;
}
/* Preservar fuente de iconos Material */
[data-testid="stSidebar"] [data-testid="stIconMaterial"],
[data-testid="stSidebar"] span[aria-hidden="true"],
[data-testid="stSidebar"] .material-symbols-rounded,
[data-testid="stSidebar"] .material-icons {
  font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}
[data-testid="stSidebarNavSeparatorHeader"] { display: none !important; }

/* ── Dataframes ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .8rem !important;
}
code, pre {
  font-family: 'JetBrains Mono', monospace !important;
}

/* ════════════════════════════════════════════════════════════
   COMPONENTES PERSONALIZADOS
   ════════════════════════════════════════════════════════════ */

/* ── Page header ────────────────────────────────────────────── */
.page-header {
  display: flex; align-items: center; gap: 18px;
  padding: 14px 0 18px; margin-bottom: 18px;
  border-bottom: 1px solid rgba(0,115,255,0.07);
}
.page-header img {
  height: 30px; width: auto;
  filter: brightness(0) invert(1); opacity: .75;
}
.header-divider {
  width: 1px; height: 34px;
  background: linear-gradient(180deg, transparent, rgba(0,115,255,0.4), transparent);
  flex-shrink: 0;
}
.header-text h1 {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 1.35rem !important; font-weight: 700 !important;
  color: var(--c-text) !important; margin: 0 !important; padding: 0 !important;
  line-height: 1.1 !important; letter-spacing: .08em !important;
  text-transform: uppercase !important;
}
.header-sub {
  font-size: .7rem; color: var(--c-muted);
  font-family: 'JetBrains Mono', monospace; margin-top: 5px;
  letter-spacing: .04em;
}

/* ── KPI card ─────────────────────────────────────────────── */
.kpi-card {
  background: linear-gradient(150deg, var(--c-surface) 0%, var(--c-raised) 100%);
  border: 1px solid var(--c-border);
  border-radius: 5px;
  padding: 20px 16px 16px;
  text-align: center; position: relative; overflow: hidden;
  box-shadow: 0 2px 18px rgba(0,0,0,.55), 0 0 0 1px rgba(0,115,255,.03);
  margin-bottom: 6px;
  transition: border-color .25s, box-shadow .25s;
}
.kpi-card:hover {
  border-color: rgba(0,115,255,.28);
  box-shadow: 0 4px 28px rgba(0,0,0,.6), 0 0 18px rgba(0,115,255,.07);
}
.kpi-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--c-cyan), rgba(0,115,255,.15));
  box-shadow: 0 1px 8px rgba(0,115,255,.25);
}
.kpi-icon { font-size: 1.15rem; margin-bottom: 10px; opacity: .55; }
.kpi-label {
  font-size: .63rem; letter-spacing: .16em; text-transform: uppercase;
  color: var(--c-muted); margin-bottom: 8px;
  font-family: 'Rajdhani', sans-serif; font-weight: 700;
}
.kpi-label::before { content: '[ '; opacity: .5; }
.kpi-label::after  { content: ' ]'; opacity: .5; }
.kpi-value {
  font-size: 2.25rem; font-weight: 400;
  font-family: 'JetBrains Mono', monospace;
  color: var(--c-cyan); line-height: 1.1;
  text-shadow: 0 0 22px rgba(0,115,255,.28);
}
.kpi-value.amber  { color: var(--c-gold);   text-shadow: 0 0 22px rgba(255,193,7,.24); }
.kpi-value.blue   { color: #38BDF8;          text-shadow: 0 0 22px rgba(56,189,248,.22); }
.kpi-value.red    { color: var(--c-red);     text-shadow: 0 0 22px rgba(255,58,92,.24); }
.kpi-value.purple { color: var(--c-purple);  text-shadow: 0 0 22px rgba(167,139,250,.22); }
.kpi-value.green  { color: var(--c-green);   text-shadow: 0 0 22px rgba(0,245,160,.22); }
.kpi-sub {
  font-size: .68rem; color: var(--c-muted); margin-top: 6px;
  font-family: 'JetBrains Mono', monospace; letter-spacing: .02em;
}

/* ── Section header ───────────────────────────────────────── */
.section-header {
  display: flex; align-items: center; gap: 10px;
  margin: 26px 0 14px; padding-bottom: 10px;
  border-bottom: 1px solid rgba(0,115,255,0.06);
}
.section-header-bar {
  width: 3px; height: 15px; flex-shrink: 0;
  background: linear-gradient(180deg, var(--c-cyan), rgba(0,115,255,.2));
  border-radius: 2px;
  box-shadow: 0 0 10px rgba(0,115,255,.35);
}
.section-header-text {
  font-family: 'Rajdhani', sans-serif;
  font-size: .85rem; font-weight: 700;
  color: var(--c-text); letter-spacing: .12em;
  text-transform: uppercase;
}

/* ── Role badge ───────────────────────────────────────────── */
.role-badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 9px; border-radius: 3px;
  font-size: .62rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase;
  font-family: 'Rajdhani', sans-serif;
}
.admin-badge   { background: rgba(255,193,7,.07);   color: var(--c-gold);   border: 1px solid rgba(255,193,7,.22); }
.manager-badge { background: rgba(0,115,255,.07);   color: var(--c-cyan);   border: 1px solid rgba(0,115,255,.22); }
.analyst-badge { background: rgba(56,189,248,.07);  color: #38BDF8;          border: 1px solid rgba(56,189,248,.22); }
.viewer-badge  { background: rgba(63,80,96,.1);     color: #7A95A8;          border: 1px solid rgba(63,80,96,.28); }

/* ── User info sidebar card ───────────────────────────────── */
.user-info-card {
  background: rgba(0,115,255,.04);
  border: 1px solid rgba(0,115,255,.1);
  border-radius: 5px; padding: 14px 16px; margin: 10px 0;
}
.user-name {
  font-family: 'Rajdhani', sans-serif;
  font-size: .92rem; font-weight: 700;
  color: var(--c-text); margin-bottom: 2px;
  letter-spacing: .05em; text-transform: uppercase;
}
.user-handle {
  font-size: .68rem; color: var(--c-muted);
  font-family: 'JetBrains Mono', monospace; letter-spacing: .02em;
}
.session-timer {
  font-size: .65rem; color: var(--c-muted);
  margin-top: 8px; padding-top: 8px;
  border-top: 1px solid rgba(0,115,255,.06);
  font-family: 'JetBrains Mono', monospace; letter-spacing: .03em;
}

/* ── Feature cards (home page) ────────────────────────────── */
[data-testid="stHorizontalBlock"] {
  align-items: stretch !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
  display: flex !important; flex-direction: column !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"] > div:first-child {
  flex: 1 !important; display: flex !important; flex-direction: column !important;
}
.feature-card {
  background: linear-gradient(150deg, var(--c-surface) 0%, var(--c-raised) 100%);
  border: 1px solid rgba(0,115,255,.1);
  border-radius: 5px;
  padding: 28px 24px 24px;
  flex: 1 !important; min-height: 270px;
  position: relative; overflow: hidden;
  transition: border-color .25s, box-shadow .25s;
}
.feature-card:hover {
  border-color: rgba(0,115,255,.24);
  box-shadow: 0 8px 32px rgba(0,0,0,.5), 0 0 22px rgba(0,115,255,.07);
}
.feature-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.feature-card.green::before {
  background: linear-gradient(90deg, var(--c-gold), rgba(194,255,0,.2));
  box-shadow: 0 1px 10px rgba(194,255,0,.18);
}
.feature-card.blue::before  {
  background: linear-gradient(90deg, var(--c-cyan), rgba(0,115,255,.15));
  box-shadow: 0 1px 10px rgba(0,115,255,.22);
}
.feature-card.amber::before {
  background: linear-gradient(90deg, var(--c-purple), rgba(167,139,250,.2));
  box-shadow: 0 1px 10px rgba(167,139,250,.18);
}
.fc-number {
  font-family: 'JetBrains Mono', monospace;
  font-size: .68rem; font-weight: 400;
  color: var(--c-muted); letter-spacing: .1em;
  margin-bottom: 16px; display: block;
}
.feature-card-icon {
  font-size: 2rem; margin-bottom: 12px; display: block; opacity: .65;
}
.feature-card h3 {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 1.25rem !important; font-weight: 700 !important;
  color: var(--c-text) !important; margin: 0 0 10px !important;
  letter-spacing: .1em !important; text-transform: uppercase !important;
}
.feature-card p {
  font-size: .88rem; color: var(--c-muted); line-height: 1.7;
  margin: 0 0 18px; font-family: 'Rajdhani', sans-serif;
  font-weight: 400; letter-spacing: .015em;
}
.feature-card-badge {
  font-size: .62rem; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; padding: 4px 10px;
  border-radius: 3px; display: inline-block;
  font-family: 'Rajdhani', sans-serif;
}
.badge-all  { background: rgba(0,115,255,.07); color: var(--c-cyan); border: 1px solid rgba(0,115,255,.22); }
.badge-tech { background: rgba(255,193,7,.07); color: var(--c-gold); border: 1px solid rgba(255,193,7,.22); }

/* ── Alert / info boxes ────────────────────────────────────── */
.success-box {
  background: rgba(0,245,160,.04); border-left: 3px solid var(--c-green);
  border-radius: 0 4px 4px 0; padding: 12px 16px; margin: 10px 0;
  font-family: 'Rajdhani', sans-serif; font-size: .9rem; color: var(--c-text);
  letter-spacing: .015em;
}
.warning-box {
  background: rgba(255,193,7,.04); border-left: 3px solid var(--c-gold);
  border-radius: 0 4px 4px 0; padding: 12px 16px; margin: 10px 0;
  font-family: 'Rajdhani', sans-serif; font-size: .9rem; color: var(--c-text);
  letter-spacing: .015em;
}
.comparison-worse {
  background: rgba(255,58,92,.04); border-left: 3px solid var(--c-red);
  border-radius: 0 4px 4px 0; padding: 12px 16px; margin: 10px 0;
}
.winner-box {
  background: rgba(255,193,7,.06); border-left: 4px solid var(--c-gold);
  border-radius: 0 5px 5px 0; padding: 14px 18px; margin: 12px 0;
  font-size: .98rem; font-weight: 700; color: var(--c-gold);
  font-family: 'Rajdhani', sans-serif; letter-spacing: .04em;
  text-transform: uppercase;
}

/* ── Footer ───────────────────────────────────────────────── */
.app-footer {
  text-align: center; padding: 20px 0 10px;
  font-size: .67rem; color: var(--c-dim);
  font-family: 'JetBrains Mono', monospace;
  border-top: 1px solid rgba(0,115,255,.05); margin-top: 36px;
  letter-spacing: .07em;
}

/* ── Hide Streamlit chrome ─────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stDecoration"] { display: none !important; }
</style>"""


# ── CSS Login ─────────────────────────────────────────────────────────────────

def get_login_css():
    """CSS para la página de login — tema Telemetría."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&display=swap');

[data-testid="stSidebar"] { display: none !important; }

[data-testid="stTextInput"] input {
  border: 1px solid rgba(0,115,255,.2) !important;
  border-radius: 4px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .9rem !important;
  background: rgba(7,12,24,.9) !important;
  transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: #0073FF !important;
  box-shadow: 0 0 0 3px rgba(0,115,255,.09), 0 0 14px rgba(0,115,255,.14) !important;
}
[data-testid="stTextInput"] label {
  color: #3F5060 !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-size: .72rem !important;
  letter-spacing: .12em !important;
  text-transform: uppercase !important;
  font-weight: 700 !important;
}

[data-testid="stFormSubmitButton"]>button {
  background: rgba(37,99,235,.10) !important;
  border: 1px solid rgba(37,99,235,.40) !important;
  color: #3B82F6 !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 700 !important;
  font-size: .95rem !important;
  letter-spacing: .14em !important;
  text-transform: uppercase !important;
  border-radius: 4px !important;
  height: 48px !important;
  transition: background .2s, box-shadow .2s, transform .15s !important;
}
[data-testid="stFormSubmitButton"]>button:hover {
  background: rgba(37,99,235,.18) !important;
  box-shadow: 0 0 22px rgba(37,99,235,.25) !important;
  transform: translateY(-1px) !important;
}

.login-card {
  background: rgba(7,12,24,.97);
  border: 1px solid rgba(0,115,255,.16);
  border-radius: 7px;
  padding: 44px 40px 36px;
  position: relative; overflow: hidden;
  box-shadow: 0 32px 64px rgba(0,0,0,.8),
              0 0 52px rgba(0,115,255,.05),
              inset 0 1px 0 rgba(255,255,255,.02);
}
.login-card::before {
  content: ''; position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, #0073FF 0%, rgba(0,115,255,.6) 60%, rgba(194,255,0,.6) 100%);
  box-shadow: 0 0 14px rgba(0,115,255,.3);
}
.login-card::after {
  content: 'SYS:TIGGO2';
  position: absolute; top: 14px; right: 18px;
  font-family: 'JetBrains Mono', monospace;
  font-size: .58rem; color: rgba(0,115,255,.2);
  letter-spacing: .1em;
}
.login-logo {
  display: block; margin: 0 auto 22px;
  height: 30px; width: auto;
  filter: brightness(0) invert(1); opacity: .72;
}
.login-title {
  font-family: 'Rajdhani', sans-serif;
  font-size: 1.5rem; font-weight: 700; color: #C9D8E6;
  text-align: center; margin: 0 0 5px;
  letter-spacing: .12em; text-transform: uppercase;
}
.login-subtitle {
  font-family: 'JetBrains Mono', monospace;
  font-size: .7rem; color: #3F5060;
  text-align: center; margin: 0 0 30px;
  letter-spacing: .05em;
}
.login-footer-txt {
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem; color: #1A2838;
  text-align: center; margin-top: 22px;
  letter-spacing: .06em;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>"""
