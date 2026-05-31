"""
============================================================================
PÁGINA: PANEL DE ADMINISTRACIÓN
============================================================================
Acceso restringido: solo rol admin.
Secciones:
  · Usuarios — lista de cuentas configuradas (sin contraseñas)
  · Audit Log — actividad reciente del sistema
  · Gestión de modelos — aprobar y eliminar runs
============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import core.supabase_io as sio
from core.auth_system import (guard_page, show_user_info, show_header)
from core.styles import kpi_card, section_header, apply_chart_theme, COLORS

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Administración — TIGGO 2", page_icon="⚙️",
    layout="wide", initial_sidebar_state="expanded",
)

# ── Auth ──────────────────────────────────────────────────────────────────────

guard_page("⚙️ Administración — TIGGO 2", roles=["admin"])

# ── Header ────────────────────────────────────────────────────────────────────

show_header(
    "Panel de Administración — TIGGO 2",
    "Usuarios · Auditoría · Gestión de modelos",
)
show_user_info()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_usuarios, tab_audit, tab_modelos = st.tabs([
    "👥 Usuarios", "📜 Audit Log", "🤖 Gestión de modelos",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — USUARIOS
# ══════════════════════════════════════════════════════════════════════════════

with tab_usuarios:

    st.markdown(section_header("Usuarios configurados", "👥"), unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(167,139,250,0.07);border:1px solid rgba(167,139,250,0.2);
            border-radius:10px;padding:12px 18px;margin-bottom:16px;">
<span style="color:#94A3B8;font-size:0.9rem;">
Los usuarios se gestionan en <code>.streamlit/secrets.toml</code>.
Para agregar, modificar o eliminar una cuenta, edita ese archivo y haz redeploy.
Las contraseñas <strong>nunca</strong> se muestran aquí.
</span>
</div>
""", unsafe_allow_html=True)

    _users_cfg = st.secrets.get("users", {})

    if not _users_cfg:
        st.warning("No hay usuarios configurados en secrets.toml")
    else:
        _rows = []
        _role_icons = {
            "admin": "👑 Admin",
            "manager": "💼 Gerente",
            "analyst": "📊 Analista",
            "financiero": "💰 Financiero",
            "viewer": "👁 Viewer",
        }
        for username, cfg in _users_cfg.items():
            _rows.append({
                "Usuario": username,
                "Nombre": cfg.get("name", "—"),
                "Rol": _role_icons.get(cfg.get("role", ""), cfg.get("role", "—")),
                "Email": cfg.get("email", "—"),
                "Auth Supabase": "✅" if cfg.get("email") else "🔑 Local",
                "Permisos": ", ".join(
                    k for k, v in cfg.get("permissions", {}).items() if v
                ) or "—",
            })
        _df_users = pd.DataFrame(_rows)

        # KPIs
        ku1, ku2, ku3 = st.columns(3)
        ku1.markdown(kpi_card("Total usuarios",  len(_rows),                                       "👥",   "blue"), unsafe_allow_html=True)
        ku2.markdown(kpi_card("Con Supabase Auth", sum(1 for r in _rows if r["Auth Supabase"]=="✅"), "🔐"), unsafe_allow_html=True)
        ku3.markdown(kpi_card("Admins",           sum(1 for r in _rows if "Admin" in r["Rol"]),    "👑",   "amber"), unsafe_allow_html=True)

        st.dataframe(_df_users, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════

with tab_audit:

    st.markdown(section_header("Registro de actividad", "📜"), unsafe_allow_html=True)

    _limit = st.slider("Entradas a mostrar", min_value=10, max_value=500, value=50, step=10)

    with st.spinner("Cargando audit log..."):
        _audit_data = sio.get_audit_log(limit=_limit)

    if not _audit_data:
        st.info("No hay entradas en el audit log todavía.")
    else:
        _df_audit = pd.DataFrame(_audit_data)

        # KPIs
        _acciones = _df_audit["accion"].value_counts()
        al1, al2, al3, al4 = st.columns(4)
        al1.markdown(kpi_card("Total acciones",  len(_df_audit),                         "📊", "blue"), unsafe_allow_html=True)
        al2.markdown(kpi_card("Logins",          _acciones.get("LOGIN", 0),              "🔑"), unsafe_allow_html=True)
        al3.markdown(kpi_card("Entrenamientos",  _acciones.get("SAVE_TRAINING", 0),      "🤖", "amber"), unsafe_allow_html=True)
        al4.markdown(kpi_card("Aprobaciones",    _acciones.get("APPROVE_MODEL", 0),      "✅"), unsafe_allow_html=True)

        # Filtro por acción
        _acc_opciones = ["Todas"] + sorted(_df_audit["accion"].unique().tolist())
        _filtro_acc = st.selectbox("Filtrar por acción", _acc_opciones)
        _df_show = _df_audit if _filtro_acc == "Todas" else _df_audit[_df_audit["accion"] == _filtro_acc]

        # Formatear
        _cols_show = [c for c in ["timestamp", "usuario", "accion", "run_name", "detalle"]
                      if c in _df_show.columns]
        st.dataframe(_df_show[_cols_show], use_container_width=True, hide_index=True)

        # Gráfico de actividad por día
        if "timestamp" in _df_audit.columns:
            try:
                _df_audit["ts"] = pd.to_datetime(_df_audit["timestamp"])
                _by_day = _df_audit.groupby(_df_audit["ts"].dt.date).size().reset_index()
                _by_day.columns = ["Fecha", "Acciones"]

                _fig_act = go.Figure()
                _fig_act.add_trace(go.Bar(
                    x=_by_day["Fecha"], y=_by_day["Acciones"],
                    marker=dict(color=COLORS["primary"], opacity=0.85),
                ))
                apply_chart_theme(_fig_act, height=280, title="Actividad diaria del sistema")
                _fig_act.update_layout(xaxis_title="Fecha", yaxis_title="Nº acciones")
                st.plotly_chart(_fig_act, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GESTIÓN DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════

with tab_modelos:

    st.markdown(section_header("Gestión de modelos entrenados", "🤖"), unsafe_allow_html=True)

    with st.spinner("Cargando runs disponibles..."):
        available_runs = sio.get_available_runs()
        active_run     = sio.get_default_run(available_runs)

    if not available_runs:
        st.info("No hay modelos entrenados aún.")
    else:
        km1, km2 = st.columns(2)
        km1.markdown(kpi_card("Modelos disponibles", len(available_runs), "🤖", "blue"), unsafe_allow_html=True)
        km2.markdown(kpi_card("Modelo activo", sio.format_run_label(active_run) if active_run else "—", "🟢"), unsafe_allow_html=True)

        # Tabla de runs
        _runs_df = sio.get_runs_df()
        if not _runs_df.empty:
            _cols = [c for c in ["run_name", "created_at", "usuario", "mape_wf", "aic", "activo"]
                     if c in _runs_df.columns]
            _runs_show = _runs_df[_cols].copy()
            if "run_name" in _runs_show:
                _runs_show["run_name"] = _runs_show["run_name"].apply(sio.format_run_label)
            if "activo" in _runs_show:
                _runs_show["activo"] = _runs_show["activo"].map({True: "🟢 Activo", False: "—"})
            st.dataframe(_runs_show, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Cambiar modelo activo")

        _non_active = [r for r in available_runs if r != active_run]

        if not _non_active:
            st.info("Solo hay un modelo disponible y ya está activo.")
        else:
            _sel_approve = st.selectbox(
                "Selecciona el run a activar",
                options=_non_active,
                format_func=sio.format_run_label,
            )
            if st.button("✅ Activar este modelo", type="primary"):
                sio.approve_model(_sel_approve, st.session_state.username)
                st.success(f"Modelo **{sio.format_run_label(_sel_approve)}** activado en producción.")
                st.rerun()

        st.markdown("---")
        st.subheader("Eliminar modelo")
        st.warning("⚠️ Esta acción es irreversible. Los artefactos se borran de Storage y DB.")

        _non_active_del = [r for r in available_runs if r != active_run]
        if not _non_active_del:
            st.info("No puedes eliminar el modelo activo. Activa otro primero.")
        else:
            _sel_delete = st.selectbox(
                "Selecciona el run a eliminar",
                options=_non_active_del,
                format_func=sio.format_run_label,
                key="sel_delete",
            )
            _confirm = st.checkbox(
                f"Confirmo que quiero eliminar **{sio.format_run_label(_sel_delete)}** permanentemente"
            )
            if st.button("🗑️ Eliminar modelo seleccionado", type="secondary", disabled=not _confirm):
                sio.delete_run(_sel_delete, st.session_state.username)
                st.success(f"Run **{sio.format_run_label(_sel_delete)}** eliminado.")
                st.rerun()
