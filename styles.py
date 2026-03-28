"""
============================================================================
MÓDULO DE ESTILOS GLOBALES — Sistema TIGGO 2
Tema: Dark Premium / Automotive Analytics
Fuentes: Exo 2 (display) + Barlow (cuerpo)
============================================================================
"""

# ── Paleta de colores ─────────────────────────────────────────────────────────

COLORS = {
    'primary':    '#20C997',   # teal (acción principal)
    'secondary':  '#0EA5E9',   # sky blue
    'accent':     '#F59E0B',   # amber / gold
    'danger':     '#EF4444',   # rojo
    'purple':     '#A855F7',
    'text':       '#E2E8F0',
    'muted':      '#64748B',
    'border':     'rgba(32,201,151,0.18)',
    # Series para gráficos
    'series': ['#20C997', '#0EA5E9', '#F59E0B', '#A855F7',
               '#EF4444', '#06D6A0', '#FB923C', '#38BDF8'],
}


# ── Tema Plotly ───────────────────────────────────────────────────────────────

def apply_chart_theme(fig, height=None, title=None):
    """Aplica el tema oscuro premium a cualquier figura Plotly."""
    layout = dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,36,0.85)',
        font=dict(family='Barlow, sans-serif', color='#94A3B8', size=12),
        xaxis=dict(
            gridcolor='rgba(30,58,95,0.45)',
            showgrid=True, zeroline=False,
            tickfont=dict(color='#64748B'),
            title_font=dict(color='#94A3B8'),
        ),
        yaxis=dict(
            gridcolor='rgba(30,58,95,0.45)',
            showgrid=True, zeroline=False,
            tickfont=dict(color='#64748B'),
            title_font=dict(color='#94A3B8'),
        ),
        hoverlabel=dict(
            bgcolor='#162030',
            font=dict(family='Barlow, sans-serif', color='#E2E8F0', size=13),
            bordercolor='rgba(32,201,151,0.3)',
        ),
        legend=dict(
            bgcolor='rgba(15,23,36,0.7)',
            bordercolor='rgba(32,201,151,0.15)',
            borderwidth=1,
            font=dict(color='#94A3B8', size=12),
        ),
        margin=dict(l=20, r=20, t=50, b=30),
    )
    if height:
        layout['height'] = height
    if title:
        layout['title'] = dict(
            text=title,
            font=dict(family='Exo 2, sans-serif', color='#CBD5E1', size=15),
            x=0, xanchor='left', pad=dict(l=4),
        )
    fig.update_layout(**layout)
    return fig


# ── Helpers HTML ──────────────────────────────────────────────────────────────

def kpi_card(label, value, icon='', color_class='', sub=''):
    """Genera HTML de una tarjeta KPI premium."""
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
    """Genera HTML de un encabezado de sección con barra decorativa."""
    icon_html = f'<span style="font-size:1rem;opacity:.7">{icon}&nbsp;</span>' if icon else ''
    return f"""
<div class="section-header">
  <div class="section-header-bar"></div>
  <span class="section-header-text">{icon_html}{text}</span>
</div>"""


# ── CSS Global ────────────────────────────────────────────────────────────────

def get_global_css():
    """CSS completo para el tema oscuro premium (inyectar con st.markdown)."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Barlow:wght@300;400;500;600&display=swap');

/* ── Tipografía base ──────────────────────────────────── */
html, body, [data-testid="stApp"] {
  font-family: 'Barlow', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
h1 { font-family:'Exo 2',sans-serif !important; font-weight:600 !important; letter-spacing:-.02em !important; }
h2 { font-family:'Exo 2',sans-serif !important; font-weight:500 !important; }
h3 { font-family:'Exo 2',sans-serif !important; font-weight:500 !important; }

/* ── Tabs ─────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family:'Barlow',sans-serif !important;
  font-weight:500 !important;
  font-size:0.88rem !important;
}

/* ── Metric cards ─────────────────────────────────────── */
[data-testid="metric-container"] {
  border-radius:12px !important;
  border:1px solid rgba(32,201,151,0.15) !important;
  padding:8px !important;
}

/* ── Botones ──────────────────────────────────────────── */
[data-testid="stButton"]>button,
[data-testid="stFormSubmitButton"]>button {
  font-family:'Barlow',sans-serif !important;
  font-weight:600 !important;
  letter-spacing:.03em !important;
  border-radius:8px !important;
}

/* ── Inputs ───────────────────────────────────────────── */
[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label {
  font-family:'Barlow',sans-serif !important;
  font-size:.82rem !important;
  letter-spacing:.04em !important;
  text-transform:uppercase !important;
  color:#64748B !important;
}

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
  border-right:1px solid rgba(32,201,151,0.1) !important;
}
[data-testid="stSidebar"] * {
  font-family:'Barlow',sans-serif !important;
}

/* ════════════════════════════════════════════════════════
   COMPONENTES PERSONALIZADOS
   ════════════════════════════════════════════════════════ */

/* ── Page header ──────────────────────────────────────── */
.page-header {
  display:flex; align-items:center; gap:18px;
  padding:14px 0 20px; margin-bottom:20px;
  border-bottom:1px solid rgba(32,201,151,0.12);
}
.page-header img {
  height:34px; width:auto;
  filter:brightness(0) invert(1); opacity:.85;
}
.header-divider {
  width:1px; height:38px;
  background:rgba(32,201,151,0.25); flex-shrink:0;
}
.header-text h1 {
  font-family:'Exo 2',sans-serif !important;
  font-size:1.45rem !important; font-weight:600 !important;
  color:#E2E8F0 !important; margin:0 !important; padding:0 !important;
  line-height:1.2 !important;
}
.header-sub {
  font-size:.8rem; color:#64748B;
  font-family:'Barlow',sans-serif; margin-top:3px;
}

/* ── KPI card ─────────────────────────────────────────── */
.kpi-card {
  background:linear-gradient(135deg,#0F1724 0%,#162030 100%);
  border:1px solid rgba(32,201,151,0.18);
  border-radius:16px; padding:22px 18px 18px;
  text-align:center; position:relative; overflow:hidden;
  box-shadow:0 4px 24px rgba(0,0,0,0.35);
  margin-bottom:6px;
}
.kpi-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,#20C997,#0EA5E9);
}
.kpi-icon { font-size:1.35rem; margin-bottom:8px; opacity:.75; }
.kpi-label {
  font-size:.68rem; letter-spacing:.12em; text-transform:uppercase;
  color:#64748B; margin-bottom:10px;
  font-family:'Barlow',sans-serif; font-weight:500;
}
.kpi-value {
  font-size:2rem; font-weight:700;
  font-family:'Exo 2',sans-serif; color:#20C997; line-height:1.1;
}
.kpi-value.amber { color:#F59E0B; }
.kpi-value.blue  { color:#0EA5E9; }
.kpi-value.red   { color:#EF4444; }
.kpi-value.purple{ color:#A855F7; }
.kpi-sub { font-size:.76rem; color:#475569; margin-top:5px; }

/* ── Section header ───────────────────────────────────── */
.section-header {
  display:flex; align-items:center; gap:10px;
  margin:26px 0 14px; padding-bottom:10px;
  border-bottom:1px solid rgba(32,201,151,0.1);
}
.section-header-bar {
  width:3px; height:18px; flex-shrink:0; border-radius:2px;
  background:linear-gradient(180deg,#20C997,#0EA5E9);
}
.section-header-text {
  font-family:'Exo 2',sans-serif; font-size:.95rem;
  font-weight:500; color:#CBD5E1;
}

/* ── Role badge ───────────────────────────────────────── */
.role-badge {
  display:inline-flex; align-items:center; gap:4px;
  padding:4px 12px; border-radius:20px;
  font-size:.68rem; font-weight:600;
  letter-spacing:.08em; text-transform:uppercase;
  font-family:'Barlow',sans-serif;
}
.admin-badge   { background:rgba(245,158,11,.1);  color:#F59E0B; border:1px solid rgba(245,158,11,.22); }
.manager-badge { background:rgba(32,201,151,.1);  color:#20C997; border:1px solid rgba(32,201,151,.22); }
.analyst-badge { background:rgba(14,165,233,.1);  color:#0EA5E9; border:1px solid rgba(14,165,233,.22); }
.viewer-badge  { background:rgba(100,116,139,.1); color:#94A3B8; border:1px solid rgba(100,116,139,.18);}

/* ── User info sidebar card ───────────────────────────── */
.user-info-card {
  background:rgba(32,201,151,.05);
  border:1px solid rgba(32,201,151,.12);
  border-radius:12px; padding:14px 16px; margin:10px 0;
}
.user-name {
  font-family:'Exo 2',sans-serif; font-size:.92rem;
  font-weight:600; color:#E2E8F0; margin-bottom:2px;
}
.user-handle { font-size:.73rem; color:#64748B; }
.session-timer {
  font-size:.7rem; color:#475569;
  margin-top:8px; padding-top:8px;
  border-top:1px solid rgba(32,201,151,.07);
}

/* ── Feature cards (home page) ────────────────────────── */
.feature-card {
  background:linear-gradient(135deg,#0F1724 0%,#162030 100%);
  border:1px solid rgba(32,201,151,.12);
  border-radius:16px; padding:30px 24px;
  height:100%; position:relative; overflow:hidden;
}
.feature-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
}
.feature-card.green::before  { background:linear-gradient(90deg,#20C997,#06D6A0); }
.feature-card.blue::before   { background:linear-gradient(90deg,#0EA5E9,#3B82F6); }
.feature-card.amber::before  { background:linear-gradient(90deg,#F59E0B,#FB923C); }
.feature-card-icon { font-size:2.2rem; margin-bottom:14px; display:block; }
.feature-card h3 {
  font-family:'Exo 2',sans-serif !important; font-size:1.1rem !important;
  font-weight:600 !important; color:#E2E8F0 !important; margin:0 0 10px !important;
}
.feature-card p {
  font-size:.86rem; color:#64748B; line-height:1.65; margin:0 0 16px;
}
.feature-card-badge {
  font-size:.67rem; font-weight:600; letter-spacing:.08em;
  text-transform:uppercase; padding:4px 10px;
  border-radius:12px; display:inline-block;
}
.badge-all  { background:rgba(32,201,151,.1); color:#20C997; }
.badge-tech { background:rgba(245,158,11,.1); color:#F59E0B; }

/* ── Custom alert boxes ───────────────────────────────── */
.success-box {
  background:rgba(32,201,151,.06); border-left:3px solid #20C997;
  border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0;
  font-family:'Barlow',sans-serif; font-size:.9rem; color:#CBD5E1;
}
.warning-box {
  background:rgba(245,158,11,.06); border-left:3px solid #F59E0B;
  border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0;
  font-family:'Barlow',sans-serif; font-size:.9rem; color:#CBD5E1;
}
.comparison-worse {
  background:rgba(239,68,68,.06); border-left:3px solid #EF4444;
  border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0;
}
.winner-box {
  background:rgba(245,158,11,.08); border-left:4px solid #F59E0B;
  border-radius:0 12px 12px 0; padding:16px 20px; margin:12px 0;
  font-size:1rem; font-weight:500; color:#FCD34D;
}

/* ── Footer ───────────────────────────────────────────── */
.app-footer {
  text-align:center; padding:20px 0 10px;
  font-size:.73rem; color:#334155;
  font-family:'Barlow',sans-serif;
  border-top:1px solid rgba(32,201,151,.07); margin-top:36px;
}

/* ── Hide Streamlit chrome ────────────────────────────── */
#MainMenu { visibility:hidden; }
footer    { visibility:hidden; }
[data-testid="stDecoration"] { display:none !important; }
</style>"""


# ── CSS Login (página de autenticación) ───────────────────────────────────────

def get_login_css():
    """CSS específico para la página de login (fondo oscuro completo)."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;500;600;700&family=Barlow:wght@300;400;500;600&display=swap');

/* Ocultar sidebar en login */
[data-testid="stSidebar"] { display:none !important; }

/* Inputs */
[data-testid="stTextInput"] input {
  border:1px solid rgba(32,201,151,.22) !important;
  border-radius:10px !important;
  font-family:'Barlow',sans-serif !important;
  font-size:.95rem !important;
  transition:border-color .2s !important;
}
[data-testid="stTextInput"] input:focus {
  border-color:#20C997 !important;
  box-shadow:0 0 0 3px rgba(32,201,151,.12) !important;
}
[data-testid="stTextInput"] label {
  color:#94A3B8 !important;
  font-family:'Barlow',sans-serif !important;
  font-size:.8rem !important;
  letter-spacing:.05em !important;
  text-transform:uppercase !important;
}

/* Submit button */
[data-testid="stFormSubmitButton"]>button {
  background:linear-gradient(135deg,#20C997 0%,#0EA5E9 100%) !important;
  color:#020812 !important;
  font-family:'Exo 2',sans-serif !important;
  font-weight:700 !important;
  font-size:1rem !important;
  letter-spacing:.06em !important;
  border:none !important;
  border-radius:10px !important;
  height:48px !important;
  transition:opacity .2s, transform .2s, box-shadow .2s !important;
}
[data-testid="stFormSubmitButton"]>button:hover {
  opacity:.9 !important;
  transform:translateY(-1px) !important;
  box-shadow:0 8px 28px rgba(32,201,151,.35) !important;
}

/* Card container */
.login-card {
  background:rgba(10,16,32,.97);
  border:1px solid rgba(32,201,151,.2);
  border-radius:20px;
  padding:46px 42px 38px;
  position:relative; overflow:hidden;
  box-shadow:0 24px 64px rgba(0,0,0,.7),
             inset 0 1px 0 rgba(255,255,255,.03);
}
.login-card::before {
  content:''; position:absolute;
  top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,#20C997,#0EA5E9,#F59E0B);
}
.login-logo {
  display:block; margin:0 auto 22px;
  height:36px; width:auto;
  filter:brightness(0) invert(1); opacity:.88;
}
.login-title {
  font-family:'Exo 2',sans-serif;
  font-size:1.35rem; font-weight:600; color:#E2E8F0;
  text-align:center; margin:0 0 5px; letter-spacing:-.01em;
}
.login-subtitle {
  font-family:'Barlow',sans-serif;
  font-size:.82rem; color:#475569;
  text-align:center; margin:0 0 32px;
}
.login-footer-txt {
  font-family:'Barlow',sans-serif;
  font-size:.7rem; color:#334155;
  text-align:center; margin-top:24px;
}

#MainMenu { visibility:hidden; }
footer    { visibility:hidden; }
</style>"""
