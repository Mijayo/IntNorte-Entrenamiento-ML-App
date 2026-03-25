"""
============================================================================
PÁGINA: COMPARACIÓN PROPHET vs SARIMA
============================================================================
Entrena ambos modelos sobre el mismo histórico y compara métricas,
gráficas y velocidad. Diseñado para entender por qué Prophet puede
superar a SARIMA en series de ventas con estacionalidad compleja.
============================================================================
"""

import io
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

import supabase_io as sio
from auth_system import (init_session_state, show_login_page, show_user_info,
                         check_session_timeout, has_permission, show_header)

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Prophet vs SARIMA", page_icon="⚔️", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .metric-prophet {background-color:#d4edda;border-left:4px solid #28a745;
                     padding:12px;border-radius:5px;margin:6px 0;}
    .metric-sarima  {background-color:#cce5ff;border-left:4px solid #004085;
                     padding:12px;border-radius:5px;margin:6px 0;}
    .winner-box     {background-color:#fff3cd;border-left:4px solid #ffc107;
                     padding:15px;border-radius:5px;margin:10px 0;font-size:1.1em;}
    </style>
""", unsafe_allow_html=True)

# ── Auth ──────────────────────────────────────────────────────────────────────

init_session_state()
if check_session_timeout():
    st.warning("⏱️ Tu sesión ha expirado.")
    st.stop()
if not st.session_state.authenticated:
    show_login_page("⚔️ Prophet vs SARIMA")
    st.stop()
if not has_permission('entrenar_modelos'):
    st.error("❌ No tienes permiso para acceder a esta página")
    st.stop()

show_header("Prophet vs SARIMA", "Comparación de modelos de series de tiempo")
show_user_info()

# ── Funciones auxiliares ──────────────────────────────────────────────────────

def metrics(real, pred, label):
    real, pred = np.array(real), np.array(pred)
    mae  = mean_absolute_error(real, pred)
    rmse = np.sqrt(mean_squared_error(real, pred))
    mape = np.mean(np.abs((real - pred) / (real + 0.1))) * 100
    return {"Modelo": label, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}


def entrenar_sarima(train_series, test_len, order, seasonal_order):
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False, maxiter=200, method='lbfgs')
    pred = res.forecast(steps=test_len)
    return pred.values, res.aic


def entrenar_prophet(train_series, test_len, country_holidays):
    df_p = pd.DataFrame({"ds": train_series.index, "y": train_series.values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode='multiplicative')
    if country_holidays:
        m.add_country_holidays(country_name='MX')
    m.fit(df_p)
    future_dates = pd.date_range(
        start=train_series.index[-1] + pd.offsets.MonthBegin(1),
        periods=test_len, freq='MS'
    )
    future = pd.DataFrame({"ds": future_dates})
    forecast = m.predict(future)
    return forecast["yhat"].values, m


def plot_comparacion(train, test, pred_sarima, pred_prophet):
    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Scatter(
        x=train.index, y=train.values, name="Histórico (train)",
        line=dict(color="#555555", width=2)
    ))
    # Real test
    fig.add_trace(go.Scatter(
        x=test.index, y=test.values, name="Real (test)",
        line=dict(color="#000000", width=2, dash="dot")
    ))
    # SARIMA
    fig.add_trace(go.Scatter(
        x=test.index, y=pred_sarima, name="SARIMA",
        line=dict(color="#1C7293", width=2.5)
    ))
    # Prophet
    fig.add_trace(go.Scatter(
        x=test.index, y=pred_prophet, name="Prophet",
        line=dict(color="#E84855", width=2.5)
    ))

    fig.update_layout(
        title="Predicciones en período de prueba",
        xaxis_title="Fecha", yaxis_title="Ventas",
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_error_bars(test, pred_sarima, pred_prophet):
    error_s = np.abs(test.values - pred_sarima)
    error_p = np.abs(test.values - pred_prophet)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=test.index, y=error_s, name="Error SARIMA",
                         marker_color="#1C7293", opacity=0.75))
    fig.add_trace(go.Bar(x=test.index, y=error_p, name="Error Prophet",
                         marker_color="#E84855", opacity=0.75))
    fig.update_layout(
        title="Error absoluto por mes",
        barmode="group", template="plotly_white", height=350,
        yaxis_title="Error (unidades)", xaxis_title="Fecha"
    )
    return fig


def plot_prophet_components(prophet_model, train_series, test_len):
    future_dates = pd.date_range(
        start=train_series.index[0],
        periods=len(train_series) + test_len, freq='MS'
    )
    future = pd.DataFrame({"ds": future_dates})
    forecast = prophet_model.predict(future)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Tendencia", "Estacionalidad anual"),
                        shared_xaxes=True)
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["trend"],
                             line=dict(color="#E84855", width=2), name="Tendencia"), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yearly"],
                             line=dict(color="#FF8C42", width=2), name="Estacionalidad"), row=2, col=1)
    fig.update_layout(title="Descomposición Prophet", template="plotly_white",
                      height=400, showlegend=False)
    return fig


# ── UI Principal ──────────────────────────────────────────────────────────────

st.markdown("""
> **¿Por qué comparar?** Prophet (Meta) y SARIMA son los dos enfoques más populares
> para series temporales mensuales. Esta página los enfrenta sobre los mismos datos
> para que puedas decidir cuál usar en producción.
""")

# ── Paso 1: Cargar datos ──────────────────────────────────────────────────────

st.header("1. Fuente de datos", divider="blue")

fuente = st.radio("Elige cómo cargar el histórico:",
                  ["Cargar desde un run guardado en Supabase",
                   "Subir archivo Excel manualmente"],
                  horizontal=True)

ventas_series = None  # pd.Series con índice mensual

if fuente == "Cargar desde un run guardado en Supabase":
    runs = sio.get_available_runs()
    if not runs:
        st.warning("No hay runs guardados. Entrena primero un modelo SARIMA.")
        st.stop()

    run_sel = st.selectbox(
        "Selecciona un run:",
        runs,
        format_func=sio.format_run_label
    )

    if st.button("Cargar histórico", type="primary"):
        with st.spinner("Descargando datos de Supabase..."):
            try:
                # load_precargados retorna (metricas, pred_total, grid_search, walk_forward, hist_total)
                _, _, _, _, hist = sio.load_precargados(run_sel)
                hist = hist.sort_index()
                hist.index = hist.index.to_period('M').to_timestamp('MS')
                st.session_state["ventas_cmp"] = hist
                st.success(f"Histórico cargado: {len(hist)} meses ({hist.index[0].strftime('%b %Y')} → {hist.index[-1].strftime('%b %Y')})")
            except Exception as e:
                st.error(f"Error al cargar: {e}")

    if "ventas_cmp" in st.session_state:
        ventas_series = st.session_state["ventas_cmp"]

else:  # carga manual
    uploaded = st.file_uploader("Excel con columna de fechas y columna de ventas", type=["xlsx", "xls"])
    if uploaded:
        try:
            df_raw = pd.read_excel(uploaded, engine="openpyxl")
            st.dataframe(df_raw.head(), use_container_width=True)
            cols = df_raw.columns.tolist()
            col_fecha = st.selectbox("Columna de fecha:", cols)
            col_ventas = st.selectbox("Columna de ventas:", [c for c in cols if c != col_fecha])
            if st.button("Usar estos datos", type="primary"):
                df_raw[col_fecha] = pd.to_datetime(df_raw[col_fecha])
                serie = (df_raw.groupby(col_fecha)[col_ventas]
                         .sum()
                         .resample('MS').sum()
                         .sort_index())
                st.session_state["ventas_cmp"] = serie
                st.success(f"Serie lista: {len(serie)} meses")
        except Exception as e:
            st.error(f"Error procesando archivo: {e}")

    if "ventas_cmp" in st.session_state:
        ventas_series = st.session_state["ventas_cmp"]

# ── A partir de aquí necesitamos datos ───────────────────────────────────────

if ventas_series is None:
    st.info("Carga o sube el histórico para continuar.")
    st.stop()

# ── Paso 2: Configuración ─────────────────────────────────────────────────────

st.header("2. Configuración de la comparación", divider="blue")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Partición train/test")
    n_test = st.slider("Meses para test (hold-out):", min_value=3, max_value=12, value=6)
    st.caption(f"Train: {len(ventas_series) - n_test} meses | Test: {n_test} meses")

with col2:
    st.subheader("Opciones de Prophet")
    usar_holidays = st.checkbox("Incluir festivos de México (MX)", value=True)
    st.caption("Prophet añade efectos de días festivos nacionales automáticamente.")

st.subheader("Parámetros SARIMA (orden manual)")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    p = st.number_input("p (AR)", min_value=0, max_value=5, value=1)
    d = st.number_input("d (I)", min_value=0, max_value=2, value=1)
    q = st.number_input("q (MA)", min_value=0, max_value=5, value=1)
with col_s2:
    P = st.number_input("P (SAR)", min_value=0, max_value=3, value=1)
    D = st.number_input("D (SI)", min_value=0, max_value=2, value=1)
    Q = st.number_input("Q (SMA)", min_value=0, max_value=3, value=1)
with col_s3:
    st.markdown("&nbsp;")
    st.markdown(f"**Seasonal period:** 12 meses")
    st.markdown("_(fijo para series mensuales)_")

# ── Paso 3: Entrenar ──────────────────────────────────────────────────────────

st.header("3. Entrenar y comparar", divider="blue")

if st.button("⚔️ Iniciar comparación", type="primary", use_container_width=True):

    train = ventas_series.iloc[:-n_test]
    test  = ventas_series.iloc[-n_test:]

    col_prog1, col_prog2 = st.columns(2)

    # --- SARIMA ---
    with col_prog1:
        st.markdown("**SARIMA**")
        with st.spinner("Entrenando SARIMA..."):
            import time
            t0 = time.time()
            try:
                pred_s, aic_s = entrenar_sarima(
                    train, n_test,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, 12)
                )
                t_sarima = time.time() - t0
                met_s = metrics(test.values, pred_s, "SARIMA")
                met_s["AIC"] = round(aic_s, 1)
                met_s["Tiempo (s)"] = round(t_sarima, 2)
                st.success(f"Listo en {t_sarima:.1f}s")
            except Exception as e:
                st.error(f"Error SARIMA: {e}")
                st.stop()

    # --- Prophet ---
    with col_prog2:
        st.markdown("**Prophet**")
        with st.spinner("Entrenando Prophet..."):
            t0 = time.time()
            try:
                pred_p, prophet_model = entrenar_prophet(train, n_test, usar_holidays)
                t_prophet = time.time() - t0
                met_p = metrics(test.values, pred_p, "Prophet")
                met_p["AIC"] = "—"
                met_p["Tiempo (s)"] = round(t_prophet, 2)
                st.success(f"Listo en {t_prophet:.1f}s")
            except Exception as e:
                st.error(f"Error Prophet: {e}")
                st.stop()

    # ── Resultados ────────────────────────────────────────────────────────────

    st.header("4. Resultados", divider="blue")

    # Tabla comparativa
    df_met = pd.DataFrame([met_s, met_p]).set_index("Modelo")
    st.dataframe(
        df_met.style.highlight_min(
            subset=["MAE", "RMSE", "MAPE (%)"],
            axis=0, color="#d4edda"
        ).highlight_min(
            subset=["Tiempo (s)"],
            axis=0, color="#fff3cd"
        ),
        use_container_width=True
    )

    # Ganador
    winner = "Prophet" if met_p["MAPE (%)"] < met_s["MAPE (%)"] else "SARIMA"
    diff   = abs(met_p["MAPE (%)"] - met_s["MAPE (%)"])
    icon   = "🏆" if diff > 1 else "🤝"
    st.markdown(
        f'<div class="winner-box">{icon} <b>Ganador por MAPE:</b> {winner} '
        f'({diff:.1f}pp de diferencia)</div>',
        unsafe_allow_html=True
    )

    # Métricas individuales en columnas
    col_m1, col_m2, col_m3 = st.columns(3)
    for col, key in zip([col_m1, col_m2, col_m3], ["MAE", "RMSE", "MAPE (%)"]):
        with col:
            delta = round(met_p[key] - met_s[key], 2) if isinstance(met_p[key], float) else 0
            st.metric(
                label=key,
                value=f"Prophet: {met_p[key]}",
                delta=f"vs SARIMA ({met_s[key]})",
                delta_color="inverse"
            )

    st.divider()

    # Gráfica de predicciones
    st.plotly_chart(plot_comparacion(train, test, pred_s, pred_p), use_container_width=True)

    # Gráfica de errores
    st.plotly_chart(plot_error_bars(test, pred_s, pred_p), use_container_width=True)

    # Descomposición Prophet
    st.subheader("Descomposición de Prophet")
    st.caption("Prophet separa la predicción en tendencia + estacionalidad, "
               "lo que facilita interpretar qué está pasando con las ventas.")
    st.plotly_chart(plot_prophet_components(prophet_model, train, n_test), use_container_width=True)

    # ── Tabla de predicciones ─────────────────────────────────────────────────

    st.subheader("Tabla detallada — período de test")
    df_tabla = pd.DataFrame({
        "Fecha": test.index.strftime("%b %Y"),
        "Real": test.values.astype(int),
        "SARIMA": np.round(pred_s).astype(int),
        "Error SARIMA": np.abs(test.values - pred_s).round(1),
        "Prophet": np.round(pred_p).astype(int),
        "Error Prophet": np.abs(test.values - pred_p).round(1),
    })
    st.dataframe(df_tabla.set_index("Fecha"), use_container_width=True)

    # ── Explicación didáctica ─────────────────────────────────────────────────

    with st.expander("📚 ¿Qué significan estos resultados?"):
        st.markdown(f"""
        ### MAPE (Mean Absolute Percentage Error)
        - **SARIMA:** {met_s['MAPE (%)']:.1f}% → en promedio, el modelo se equivoca un {met_s['MAPE (%)']:.1f}% respecto al valor real.
        - **Prophet:** {met_p['MAPE (%)']:.1f}% → en promedio, se equivoca un {met_p['MAPE (%)']:.1f}%.
        - Regla práctica: MAPE < 10% es excelente, < 20% es aceptable en ventas.

        ### ¿Por qué Prophet puede ganar?
        1. **Estacionalidad multiplicativa** — si las ventas suben en diciembre, Prophet escala el efecto
           proporcionalmente a la tendencia. SARIMA usa diferenciación, que asume efectos aditivos.
        2. **Festivos de México** — Prophet añade automáticamente el efecto de días como
           Día de Muertos, Navidad, Año Nuevo sobre las ventas.
        3. **Changepoints** — si hubo un cambio de modelo o una crisis (ej. COVID),
           Prophet detecta y adapta la tendencia. SARIMA no lo hace de forma automática.

        ### ¿Cuándo SARIMA puede ganar?
        - Series muy cortas (< 36 meses) donde Prophet no tiene suficiente historia.
        - Cuando se tienen variables exógenas relevantes (stock, precio) — SARIMAX
          las incorpora nativamente. Prophet las añade como `add_regressor()`.
        - Series muy estables sin cambios de régimen.
        """)

    # Guardar resultados en session_state para referencia
    st.session_state["cmp_resultados"] = {
        "sarima": met_s, "prophet": met_p,
        "pred_sarima": pred_s, "pred_prophet": pred_p,
        "test": test, "train": train
    }
