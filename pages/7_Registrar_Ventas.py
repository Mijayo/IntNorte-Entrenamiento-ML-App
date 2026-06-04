"""
============================================================================
PÁGINA: REGISTRAR VENTAS REALES
============================================================================
Feedback loop: permite ingresar las ventas reales de cada mes una vez
cerrado el período. Los datos se usan para:
  · Comparar forecast del modelo activo vs ventas reales
  · Detectar drift automáticamente (error > 15% activa alerta)
  · Mostrar el scoreboard acumulado: ¿cuánto acertó el modelo en producción?

Acceso: admin · analyst
============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

import core.supabase_io as sio
from core.auth_system import (guard_page, show_user_info, show_header)
from core.styles import kpi_card, section_header, apply_chart_theme, COLORS

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Registrar Ventas — TIGGO 2", page_icon="📋",
    layout="wide", initial_sidebar_state="expanded",
)

# ── Auth ──────────────────────────────────────────────────────────────────────

guard_page("📋 Registrar Ventas — TIGGO 2", roles=["admin", "analyst"])

# ── Selector de run (sidebar) ─────────────────────────────────────────────────

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
    help="Selecciona el modelo contra el que comparar las ventas reales.",
)

is_latest = sio.get_default_run(available_runs) == selected_run
st.sidebar.caption("🟢 Activo en producción" if is_latest else "🔵 Versión histórica")

# ── Cargar datos del run ──────────────────────────────────────────────────────

with st.spinner("Cargando datos del modelo..."):
    metricas, pred_total, _gs, _wf, hist_total, _exog = sio.load_precargados(selected_run)

show_header(
    "Registrar Ventas Reales — TIGGO 2",
    f"Feedback loop  |  Modelo: {sio.format_run_label(selected_run)} {'🟢' if is_latest else '🔵'}",
)
show_user_info()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["➕ Registrar mes", "📡 Comparativa en producción"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — REGISTRAR MES
# ══════════════════════════════════════════════════════════════════════════════

with tab1:

    st.markdown(section_header("Registrar Ventas Reales del Mes", "➕"), unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(0,115,255,0.07);border:1px solid rgba(0,115,255,0.22);
            border-radius:10px;padding:14px 20px;margin-bottom:20px;">
<span style="font-size:1.0rem;font-weight:600;color:#4D9FFF;">¿Para qué sirve esto?</span><br>
<span style="color:#94A3B8;font-size:0.9rem;">
Cada mes, una vez cerrado el período de ventas, ingresa aquí las unidades reales vendidas.
El sistema las compara automáticamente con lo que el modelo había predicho para ese mes,
calculando el error real en producción — no sobre datos de entrenamiento.
Si el error supera el 15%, se activa una alerta para reentrenar.
</span>
</div>
""", unsafe_allow_html=True)

    # ── Formulario de registro ─────────────────────────────────────────────────

    with st.form("form_registrar_venta", border=True):
        fc1, fc2 = st.columns([1, 1])
        with fc1:
            mes_sel = st.selectbox(
                "Mes",
                options=list(range(1, 13)),
                format_func=lambda m: ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                                       "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"][m-1],
                index=max(0, datetime.now().month - 2),
            )
        with fc2:
            anio_sel = st.number_input(
                "Año",
                min_value=2020, max_value=datetime.now().year,
                value=datetime.now().year if datetime.now().month > 1 else datetime.now().year - 1,
            )
        ventas_input = st.number_input(
            "Unidades vendidas (real)",
            min_value=0, max_value=10_000, value=0, step=1,
        )
        submitted = st.form_submit_button("💾 Guardar registro", use_container_width=True)

    if submitted:
        if ventas_input == 0:
            st.warning("⚠️ Ingresa al menos 1 unidad. Si las ventas fueron 0, ingresa 1 y ajusta.")
        else:
            _fecha_str = f"{int(anio_sel):04d}-{mes_sel:02d}-01"
            try:
                sio.save_venta_real(_fecha_str, int(ventas_input), st.session_state.username)
                st.success(f"✅ Registro guardado: {_fecha_str} → {ventas_input:,} unidades")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error al guardar: {e}")

    # ── Historial de registros ─────────────────────────────────────────────────

    st.markdown(section_header("Historial de Registros", "📋"), unsafe_allow_html=True)

    ventas_reales = sio.get_ventas_reales()

    if not ventas_reales:
        st.info("Aún no hay ventas reales registradas. Usa el formulario de arriba para agregar el primer mes.")
    else:
        _df_hist = pd.DataFrame(ventas_reales)
        _df_hist["fecha"] = pd.to_datetime(_df_hist["fecha"])
        _df_hist = _df_hist.sort_values("fecha", ascending=False).reset_index(drop=True)
        _df_hist["Mes"] = _df_hist["fecha"].dt.strftime("%B %Y")
        _df_hist_show = _df_hist[["Mes", "ventas", "usuario", "timestamp"]].copy()
        _df_hist_show.columns = ["Mes", "Unidades reales", "Registrado por", "Timestamp"]

        st.dataframe(_df_hist_show, use_container_width=True, hide_index=True)

        # Eliminación de registros (admin only)
        if st.session_state.role == "admin":
            with st.expander("🗑️ Eliminar un registro"):
                _fecha_opts = {
                    row["Mes"]: row["fecha"].strftime("%Y-%m-%d")
                    for _, row in _df_hist.iterrows()
                }
                _del_label = st.selectbox("Selecciona el mes a eliminar", options=list(_fecha_opts.keys()))
                if st.button("Eliminar registro seleccionado", type="secondary"):
                    sio.delete_venta_real(_fecha_opts[_del_label], st.session_state.username)
                    st.success(f"Registro de {_del_label} eliminado.")
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARATIVA EN PRODUCCIÓN
# ══════════════════════════════════════════════════════════════════════════════

with tab2:

    st.markdown(section_header("Comparativa en Producción — Forecast vs Real", "📡"),
                unsafe_allow_html=True)

    ventas_reales = sio.get_ventas_reales()

    if not ventas_reales:
        st.info(
            "Aún no hay ventas reales registradas. "
            "Regresa aquí después de ingresar al menos un mes real en la pestaña **Registrar mes**."
        )
    else:
        _vr_df = pd.DataFrame(ventas_reales)
        _vr_df["fecha"] = pd.to_datetime(_vr_df["fecha"]).dt.to_period("M").dt.to_timestamp()

        # Unir con predicciones — normalizar a inicio de mes para que
        # pred_total (freq='ME' → fin de mes) coincida con las fechas registradas (día 1)
        _all_pred = pred_total[["Fecha", "Predicción", "IC_Inferior", "IC_Superior"]].copy()
        _all_pred.columns = ["fecha", "prediccion", "ic_inf", "ic_sup"]
        _all_pred["fecha"] = _all_pred["fecha"].dt.to_period("M").dt.to_timestamp()

        _merged = _vr_df.merge(_all_pred, on="fecha", how="inner")

        if _merged.empty:
            # Intentar con walk-forward (datos históricos cubiertos por wf)
            st.info(
                "Las fechas registradas no coinciden con el horizonte de predicción del modelo "
                f"({sio.format_run_label(selected_run)}). "
                "Prueba seleccionando el modelo activo o registra ventas de los meses que este modelo predijo."
            )
        else:
            _merged = _merged.copy()
            _merged["error_abs"] = abs(_merged["ventas"] - _merged["prediccion"])
            _merged["error_pct"] = _merged["error_abs"] / _merged["ventas"] * 100
            _merged["dentro_ic"] = (
                (_merged["ventas"] >= _merged["ic_inf"]) &
                (_merged["ventas"] <= _merged["ic_sup"])
            )

            # ── KPIs de producción ─────────────────────────────────────────────
            _mape_prod = _merged["error_pct"].mean()
            _ic_rate   = _merged["dentro_ic"].mean() * 100
            _mejor_mes = _merged.loc[_merged["error_pct"].idxmin(), "fecha"].strftime("%b %Y")
            _peor_mes  = _merged.loc[_merged["error_pct"].idxmax(), "fecha"].strftime("%b %Y")
            _mape_col  = "red" if _mape_prod > 15 else ("amber" if _mape_prod > 10 else "")

            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.markdown(kpi_card("MAPE producción",  f"{_mape_prod:.1f}%",   "🎯", _mape_col), unsafe_allow_html=True)
            kc2.markdown(kpi_card("Dentro del IC 95%", f"{_ic_rate:.0f}%",    "📐", "blue"), unsafe_allow_html=True)
            kc3.markdown(kpi_card("Mejor mes",         _mejor_mes,            "✅"), unsafe_allow_html=True)
            kc4.markdown(kpi_card("Peor mes",          _peor_mes,             "⚠️", "red"), unsafe_allow_html=True)

            # Alerta de drift
            if _mape_prod > 15:
                st.error(
                    f"⚠️ **Drift detectado:** MAPE en producción = {_mape_prod:.1f}% (umbral: 15%). "
                    "El modelo no está prediciendo bien los datos reales. "
                    "**Acción recomendada:** reentrenar con los datos más recientes."
                )
            elif _mape_prod > 10:
                st.warning(
                    f"ℹ️ MAPE en producción = {_mape_prod:.1f}% — aceptable pero cerca del umbral."
                )
            else:
                st.success(
                    f"✅ El modelo está prediciendo bien en producción — MAPE: {_mape_prod:.1f}%"
                )

            # ── Gráfico real vs predicción ─────────────────────────────────────
            _fig = go.Figure()

            # Banda IC 95%
            _fig.add_trace(go.Scatter(
                x=_merged["fecha"].tolist() + _merged["fecha"].tolist()[::-1],
                y=_merged["ic_sup"].tolist() + _merged["ic_inf"].tolist()[::-1],
                fill="toself", fillcolor="rgba(255,58,92,0.08)",
                line=dict(color="rgba(0,0,0,0)"), name="IC 95%",
            ))
            # Predicción
            _fig.add_trace(go.Scatter(
                x=_merged["fecha"], y=_merged["prediccion"],
                mode="lines+markers", name="Predicción modelo",
                line=dict(color=COLORS["accent"], width=2.5, dash="dot"),
                marker=dict(size=8, color=COLORS["accent"], symbol="diamond",
                            line=dict(color="#080D18", width=1.5)),
                hovertemplate="%{x|%b %Y}<br>Predicción: %{y:.0f} uds<extra></extra>",
            ))
            # Real
            _fig.add_trace(go.Scatter(
                x=_merged["fecha"], y=_merged["ventas"],
                mode="lines+markers", name="Real registrado",
                line=dict(color=COLORS["success"], width=2.5),
                marker=dict(size=10, color=COLORS["success"],
                            line=dict(color="#080D18", width=1.5)),
                hovertemplate="%{x|%b %Y}<br>Real: %{y:.0f} uds<extra></extra>",
            ))

            apply_chart_theme(_fig, height=440,
                              title="Forecast del modelo vs ventas reales en producción")
            _fig.update_layout(hovermode="x unified",
                               xaxis_title="Mes", yaxis_title="Unidades")
            st.plotly_chart(_fig, use_container_width=True, config={"displayModeBar": False})

            # ── Tabla detallada ────────────────────────────────────────────────
            st.markdown(section_header("Detalle por mes", "📋"), unsafe_allow_html=True)
            _tbl = _merged[["fecha", "ventas", "prediccion", "error_abs", "error_pct", "dentro_ic"]].copy()
            _tbl["fecha"] = _tbl["fecha"].dt.strftime("%B %Y")
            _tbl.columns = ["Mes", "Real", "Predicción", "Error Abs", "Error %", "Dentro IC 95%"]
            _tbl["Dentro IC 95%"] = _tbl["Dentro IC 95%"].map({True: "✅ Sí", False: "❌ No"})

            st.dataframe(
                _tbl.style
                    .background_gradient(subset=["Error %"], cmap="RdYlGn_r")
                    .format({"Real": "{:.0f}", "Predicción": "{:.1f}",
                             "Error Abs": "{:.2f}", "Error %": "{:.1f}%"}),
                use_container_width=True, hide_index=True,
            )
