"""
============================================================================
MÓDULO DE ESTILOS GLOBALES — Sistema TIGGO 2
Tema: Telemetría / Racing Data Dashboard
Fuentes: Rajdhani (display) + JetBrains Mono (datos)
============================================================================
"""

# ── Paleta de colores ─────────────────────────────────────────────────────────

COLORS = {
    'primary':    '#0073FF',   # vibrant blue (acción principal)
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


# ── Font preload links (inject once, before any CSS) ─────────────────────────

_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Rajdhani:wght@400;500;600;700'
    '&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,700;1,300'
    '&display=swap">'
)


def get_font_links() -> str:
    """Preconnect + stylesheet <link> tags — inject once per page before CSS."""
    return _FONT_LINKS


# ── CSS Global ────────────────────────────────────────────────────────────────

_GLOBAL_CSS = """
<style>
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

/* ── Card KPI (home page feature cards) ───────────────────── */
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
.card-kpi-value.lime   { color: var(--c-gold);   text-shadow: 0 0 18px rgba(194,255,0,.25); }
.card-kpi-value.violet { color: var(--c-purple);  text-shadow: 0 0 18px rgba(167,139,250,.2); }
.card-kpi-label {
  font-size: .55rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--c-muted);
  margin-top: 4px;
}

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

/* ── Responsive block container padding ───────────────────── */
[data-testid="block-container"] {
  padding-left: 1rem !important;
  padding-right: 1rem !important;
}
@media (min-width: 640px) {
  [data-testid="block-container"] {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
  }
}
@media (min-width: 1024px) {
  [data-testid="block-container"] {
    padding-left: 3rem !important;
    padding-right: 3rem !important;
  }
}

/* ── Hide Streamlit chrome ─────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stDecoration"] { display: none !important; }
</style>"""


def get_global_css() -> str:
    return _GLOBAL_CSS


# ── CSS Home ──────────────────────────────────────────────────────────────────

_HOME_CSS = """
<style>
.home-hero {
  padding: 24px 0 20px;
  margin-bottom: 8px;
  border-bottom: 1px solid rgba(0,115,255,0.10);
  position: relative;
}
@media (min-width: 640px) {
  .home-hero { padding: 44px 0 40px; }
}
.home-hero::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, rgba(0,115,255,0.5), rgba(194,255,0,0.3), transparent);
}
.hero-meta-row {
  display: flex; align-items: center; flex-wrap: wrap; gap: 8px;
  margin-bottom: 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: .58rem; letter-spacing: .18em;
  color: var(--c-muted); text-transform: uppercase;
}
@media (min-width: 640px) {
  .hero-meta-row { gap: 10px; margin-bottom: 22px; }
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
.contexto-kpis {
  display: grid !important;
  grid-template-columns: repeat(2, 1fr) !important;
  gap: 20px 12px !important;
  align-items: center !important;
}
@media (min-width: 640px) {
  .contexto-kpis {
    grid-template-columns: repeat(4, 1fr) !important;
    gap: 16px !important;
  }
}

/* ── Context panel ────────────────────────────────── */
.context-panel {
  margin-bottom: 20px; padding: 14px 16px;
  background: linear-gradient(135deg, rgba(0,115,255,0.06), rgba(194,255,0,0.04));
  border: 1px solid rgba(0,115,255,0.14); border-radius: 8px; position: relative;
}
@media (min-width: 640px) {
  .context-panel { margin-bottom: 28px; padding: 18px 24px; }
}
.context-panel-label {
  font-family: 'Rajdhani', sans-serif; font-size: .6rem; font-weight: 700;
  letter-spacing: .2em; text-transform: uppercase; color: var(--c-muted); margin-bottom: 10px;
}
.context-panel-body { display: flex; flex-direction: column; gap: 20px; }
.context-title {
  font-family: 'Rajdhani', sans-serif; font-size: 1.02rem; font-weight: 700; color: var(--c-text);
}
.context-text { color: #7A95A8; font-size: .84rem; line-height: 1.75; margin-top: 6px; }
.context-text strong { color: var(--c-gold); }
.contexto-kpis > div { text-align: center; }
.ctx-kpi-value {
  font-family: 'Rajdhani', sans-serif; font-size: 2.8rem; font-weight: 700; line-height: 1;
}
.ctx-kpi-value.blue   { color: var(--c-cyan); }
.ctx-kpi-value.lime   { color: var(--c-gold); }
.ctx-kpi-value.green  { color: var(--c-green); }
.ctx-kpi-value.violet { color: var(--c-purple); }
.ctx-kpi-sub {
  font-family: 'JetBrains Mono', monospace; font-size: .58rem;
  color: var(--c-muted); text-transform: uppercase; letter-spacing: .12em; margin-top: 3px;
}

/* ── Cards grid ───────────────────────────────────── */
.cards-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
  margin-top: 8px;
}
@media (min-width: 580px) {
  .cards-grid { grid-template-columns: repeat(2, 1fr); gap: 20px 14px; }
}
@media (min-width: 960px) {
  .cards-grid { grid-template-columns: repeat(3, 1fr); gap: 28px 16px; }
}

/* ── Home footer ──────────────────────────────────── */
.home-footer {
  margin-top: 32px; padding-top: 14px;
  border-top: 1px solid rgba(0,115,255,0.06);
  display: flex; flex-direction: column; align-items: center; gap: 6px;
  text-align: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: .55rem; letter-spacing: .16em;
  color: var(--c-muted); text-transform: uppercase; opacity: .6;
}
@media (min-width: 640px) {
  .home-footer {
    flex-direction: row; justify-content: space-between;
    align-items: center; gap: 0; text-align: left;
    margin-top: 48px; padding-top: 16px;
  }
}
</style>"""


def get_home_css() -> str:
    return _HOME_CSS


# ── CSS Login ─────────────────────────────────────────────────────────────────

_LOGIN_CSS = """
<style>
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes gradientShift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* ── Viewport fit cross-browser (Chrome · Safari · Firefox · Edge) ── */
html { height: -webkit-fill-available; }
.stApp {
  height: 100vh !important;
  height: 100dvh !important;          /* Chrome 108+, Firefox 116+, Safari 15.4+ */
  overflow: hidden !important;
}
.main {
  height: 100% !important;
  overflow: hidden !important;
}
.main .block-container {
  height: 100% !important;
  padding-top: 1.5rem !important;
  padding-bottom: 1.5rem !important;
  -webkit-box-sizing: border-box !important;
  box-sizing: border-box !important;
  display: -webkit-box !important;
  display: -webkit-flex !important;
  display: flex !important;
  -webkit-box-orient: vertical !important;
  -webkit-box-direction: normal !important;
  -webkit-flex-direction: column !important;
  flex-direction: column !important;
  -webkit-box-pack: center !important;
  -webkit-justify-content: center !important;
  justify-content: center !important;
  overflow: hidden !important;
}
/* Safari fallback: 100vh incluye la barra de direcciones */
@supports (-webkit-touch-callout: none) {
  .stApp { height: -webkit-fill-available !important; }
}
[data-testid="stVerticalBlock"] {
  gap: 6px !important;
  -webkit-gap: 6px !important;
}

[data-testid="stSidebar"] { display: none !important; }

[data-testid="stTextInput"] {
  margin-bottom: 12px !important;
}
[data-testid="stTextInput"] input {
  border: 1px solid rgba(0,115,255,.2) !important;
  border-radius: 4px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .9rem !important;
  background: rgba(7,12,24,.9) !important;
  padding: 14px 16px !important;
  height: 52px !important;
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

[data-testid="stForm"] {
  animation: fadeInUp .45s ease-out .12s both !important;
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
  height: 54px !important;
  margin-top: 8px !important;
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
  padding: clamp(24px, 4.5vh, 52px) clamp(28px, 4vw, 56px) clamp(18px, 3vh, 40px);
  position: relative; overflow: hidden;
  box-shadow: 0 32px 64px rgba(0,0,0,.8),
              0 0 52px rgba(0,115,255,.05),
              inset 0 1px 0 rgba(255,255,255,.02);
  animation: fadeInUp .45s ease-out both;
  margin-bottom: 30px !important;
}
@media (min-width: 768px) {
  .login-card { margin-bottom: 0; }
}
.login-card::before {
  content: ''; position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, #0073FF, rgba(0,115,255,.5), rgba(194,255,0,.7), rgba(0,115,255,.5), #0073FF);
  background-size: 300% 100%;
  animation: gradientShift 5s ease infinite;
  box-shadow: 0 0 14px rgba(0,115,255,.3);
}
.login-card::after {
  content: 'SYS:TIGGO2  ·  v3.1';
  position: absolute; top: 14px; right: 18px;
  font-family: 'JetBrains Mono', monospace;
  font-size: .58rem; color: rgba(0,115,255,.2);
  letter-spacing: .1em;
}
.login-logo {
  display: block; margin: 0 auto clamp(12px, 2vh, 24px);
  height: 52px; width: auto;
  filter: brightness(0) invert(1); opacity: .72;
}
.login-title {
  font-family: 'Rajdhani', sans-serif;
  font-size: 1.75rem; font-weight: 700; color: #C9D8E6;
  text-align: center; margin: 0 0 6px;
  letter-spacing: .12em; text-transform: uppercase;
}
.login-subtitle {
  font-family: 'JetBrains Mono', monospace;
  font-size: .7rem; color: #3F5060;
  text-align: center; margin: 0 0 clamp(14px, 2.2vh, 24px);
  letter-spacing: .05em;
}
.login-divider {
  height: 1px;
  background: rgba(0,115,255,.1);
  margin: 0;
}
.login-error {
  border: 1px solid rgba(239,68,68,.3) !important;
  background: rgba(239,68,68,.05) !important;
  border-radius: 4px !important;
  padding: 10px 14px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .75rem !important;
  color: #F87171 !important;
  margin-top: 10px !important;
  letter-spacing: .03em;
}
.login-warning {
  border: 1px solid rgba(234,179,8,.25) !important;
  background: rgba(234,179,8,.04) !important;
  border-radius: 4px !important;
  padding: 10px 14px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .75rem !important;
  color: #FCD34D !important;
  margin-top: 10px !important;
  letter-spacing: .03em;
}
.login-forgot {
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem;
  color: rgba(0,115,255,.28);
  text-align: center;
  margin-top: 8px;
  letter-spacing: .04em;
  cursor: default;
}
.login-footer-txt {
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem; color: #1A2838;
  text-align: center; margin-top: 14px;
  letter-spacing: .06em;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
<script>
(function () {
  function patchAutocomplete() {
    var inputs = document.querySelectorAll('[data-testid="stTextInput"] input');
    if (inputs.length < 2) { setTimeout(patchAutocomplete, 80); return; }
    inputs[0].setAttribute('autocomplete', 'username');
    inputs[1].setAttribute('autocomplete', 'current-password');
  }
  patchAutocomplete();
})();
</script>"""



def get_login_css() -> str:
    return _LOGIN_CSS
