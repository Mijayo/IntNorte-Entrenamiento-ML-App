"""
============================================================================
PÁGINA: CONCESIONARIOS — Análisis histórico + predicciones por tienda
Metodología de predicción: shares históricos aplicados sobre predicción SARIMA
============================================================================
"""

import io
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

import core.supabase_io as sio
from core.auth_system import (guard_page, show_user_info, show_header, has_permission)
from core.styles import kpi_card, section_header, apply_chart_theme, COLORS

# ── Contorno simplificado de Perú (GeoJSON inline) ───────────────────────────

_PERU_BORDER_GEOJSON = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-80.30, -3.40], [-80.64, -1.77], [-80.51,  0.00],
                [-78.89, -0.27], [-75.24, -0.11], [-75.54,  0.13],
                [-73.65, -1.26], [-72.12, -2.12], [-70.85, -4.22],
                [-70.39, -9.67], [-72.65,-10.98], [-70.64,-11.01],
                [-69.58,-13.98], [-69.50,-16.50], [-69.51,-17.50],
                [-69.85,-18.30], [-70.41,-18.35], [-75.71,-16.20],
                [-76.19,-14.00], [-77.61,-11.01], [-79.57, -8.46],
                [-80.68, -6.24], [-81.18, -4.78], [-80.30, -3.40],
            ]]
        },
        "properties": {}
    }]
}

# ── Coordenadas de ciudades peruanas ─────────────────────────────────────────

_COORDS_PERU = {
    'lima':      (-12.0464, -77.0428),
    'callao':    (-12.0565, -77.1197),
    'piura':     (-5.1945,  -80.6328),
    'chiclayo':  (-6.7714,  -79.8409),
    'tarapoto':  (-6.4870,  -76.3710),
    'cajamarca': (-7.1638,  -78.5128),
    'trujillo':  (-8.1116,  -79.0287),
    'arequipa':  (-16.4090, -71.5375),
    'cusco':     (-13.5319, -71.9675),
    'iquitos':   (-3.7489,  -73.2500),
    'huancayo':  (-12.0653, -75.2049),
    'puno':      (-15.8422, -70.0199),
}


def _coords_concesionario(nombre: str):
    """Devuelve (lat, lon) buscando nombre de ciudad dentro del nombre del concesionario."""
    n = nombre.lower()
    for ciudad, coords in _COORDS_PERU.items():
        if ciudad in n:
            return coords
    return None


# Mapeo ciudad → nombre de departamento (normalizado sin acentos)
_CIUDAD_TO_DPTO = {
    'lima':      'Lima',
    'callao':    'Lima',
    'piura':     'Piura',
    'chiclayo':  'Lambayeque',
    'tarapoto':  'San Martin',
    'cajamarca': 'Cajamarca',
    'trujillo':  'La Libertad',
    'arequipa':  'Arequipa',
    'cusco':     'Cusco',
    'iquitos':   'Loreto',
    'huancayo':  'Junin',
    'puno':      'Puno',
}


def _norm_dpto(s: str) -> str:
    """Normaliza a ASCII uppercase para comparación insensible a acentos."""
    import unicodedata
    return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('ascii').upper()


@st.cache_data(show_spinner=False)
def _get_peru_geojson():
    import urllib.request, json
    # GADM 4.1 — fuente oficial, property NAME_1 = nombre del departamento
    urls = [
        "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_PER_1.json",
        "https://raw.githubusercontent.com/juliansotelo/peruviandatasets/master/peruviandatasets/geo/departamentos.json",
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                data = json.loads(r.read())
                if data.get('features'):
                    return data
        except Exception:
            continue
    return None


# ── Datos precargados ─────────────────────────────────────────────────────────

def _normalizar_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas de fecha y modelo en un DataFrame de ventas."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    if len(raw) > 0 and raw.iloc[0].apply(lambda x: isinstance(x, str)).all():
        raw = raw.iloc[1:].reset_index(drop=True)
    for fc in ['FECHA_VENTA', 'FECHA-VENTA', 'FECHA VENTA']:
        if fc in raw.columns:
            raw[fc] = pd.to_datetime(raw[fc], errors='coerce')
            if fc != 'FECHA_VENTA':
                raw = raw.rename(columns={fc: 'FECHA_VENTA'})
            break
    for mc in ['MODELO2', 'MODELO3', 'MODELO']:
        if mc in raw.columns:
            raw = raw.rename(columns={mc: 'MODELO_NORM'})
            break
    return raw


@st.cache_data(show_spinner=False)
def _cargar_precargado() -> pd.DataFrame | None:
    # 1. Supabase Storage (fuente primaria — funciona en cualquier despliegue)
    try:
        df_v, _ = sio.load_datos_precargados()
        return _normalizar_df(df_v)
    except Exception:
        pass
    # 2. Fallback: archivo local (entornos de desarrollo)
    path = Path(__file__).parent.parent / "data" / "processed" / "veh_ml_features.xlsx"
    if not path.exists():
        return None
    return _normalizar_df(pd.read_excel(path, sheet_name="Hoja1", engine='openpyxl'))


def _procesar_excel(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normaliza columnas de fecha y modelo. Retorna (df, errores)."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    if len(raw) > 0 and raw.iloc[0].apply(lambda x: isinstance(x, str)).all():
        raw = raw.iloc[1:].reset_index(drop=True)
    errors: list[str] = []
    for fc in ['FECHA_VENTA', 'FECHA-VENTA', 'FECHA VENTA']:
        if fc in raw.columns:
            raw[fc] = pd.to_datetime(raw[fc], errors='coerce')
            bad = raw[fc].isna().sum()
            if bad:
                errors.append(f"⚠️ {bad} fechas no parseables en `{fc}`.")
            if fc != 'FECHA_VENTA':
                raw = raw.rename(columns={fc: 'FECHA_VENTA'})
            break
    else:
        errors.append("❌ Columna de fecha no encontrada.")
    for mc in ['MODELO2', 'MODELO3', 'MODELO']:
        if mc in raw.columns:
            raw = raw.rename(columns={mc: 'MODELO_NORM'})
            break
    return raw, errors


# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Concesionarios TIGGO 2", page_icon="🏪",
    layout="wide", initial_sidebar_state="expanded"
)

# ── Auth ──────────────────────────────────────────────────────────────────────

guard_page("🏪 Concesionarios TIGGO 2", roles=['admin', 'analyst', 'manager'])

# ── Selector de versión del modelo (sidebar) ──────────────────────────────────

available_runs = sio.get_available_runs()
if not available_runs:
    st.error("❌ No hay modelos entrenados. Ejecuta primero la página de Entrenamiento.")
    st.stop()

default_run = sio.get_default_run(available_runs)
selected_run = st.sidebar.selectbox(
    "📦 Versión del modelo",
    options=available_runs,
    index=available_runs.index(default_run) if default_run in available_runs else 0,
    format_func=sio.format_run_label,
)
is_latest = sio.get_default_run(available_runs) == selected_run
st.sidebar.caption("🟢 Activo en producción" if is_latest else "🔵 Versión histórica")

# ── Cargar predicción SARIMA total ────────────────────────────────────────────

with st.spinner("Cargando modelo SARIMA..."):
    _metricas, pred_total, _grid, _wf, hist_total, _exog = sio.load_precargados(selected_run)

# ── Header ────────────────────────────────────────────────────────────────────

show_header(
    "Concesionarios — Análisis y Predicciones",
    f"Desglose por tienda  |  Modelo: {sio.format_run_label(selected_run)} {'🟢' if is_latest else '🔵'}"
)
show_user_info()

# ── Fuente de datos ───────────────────────────────────────────────────────────

_precargado = _cargar_precargado()
_tiene_custom = 'df_concesionarios' in st.session_state

with st.expander("📂 Fuente de datos", expanded=False):
    if _tiene_custom:
        c1, c2 = st.columns([4, 1])
        n_custom = len(st.session_state['df_concesionarios'])
        c1.success(f"📁 Archivo personalizado activo — {n_custom:,} registros")
        with c2:
            if st.button("↩ Usar precargados"):
                del st.session_state['df_concesionarios']
                st.rerun()
    elif _precargado is not None:
        n_pre = len(_precargado)
        n_ch  = len(_precargado[_precargado['MARCA'] == 'CHERY']) if 'MARCA' in _precargado.columns else n_pre
        st.info(f"📦 Datos precargados — **veh_ml_features.xlsx** · {n_pre:,} registros · {n_ch:,} CHERY")
    else:
        st.warning("⚠️ No hay datos precargados en Supabase Storage ni en disco local. Carga un Excel para continuar.")

    st.caption("Sube tu propio Excel para reemplazar los datos precargados. Columnas mínimas: MARCA · MODELO/MODELO3 · FECHA-VENTA · CONCESIONARIO")
    con_file = st.file_uploader("Excel personalizado de ventas", type=['xlsx', 'xls'], key="conc_page_uploader")
    if con_file and not _tiene_custom:
        with st.spinner("Procesando..."):
            try:
                file_bytes = con_file.read()
                raw = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
                raw, errors = _procesar_excel(raw)
                for msg in errors:
                    (st.error if msg.startswith("❌") else st.warning)(msg)
                if not any(m.startswith("❌") for m in errors):
                    st.session_state['df_concesionarios'] = raw
                    st.session_state['_conc_pending_bytes'] = file_bytes
                    n_ch = len(raw[raw['MARCA'] == 'CHERY']) if 'MARCA' in raw.columns else len(raw)
                    st.success(f"✅ {len(raw):,} registros · {n_ch:,} CHERY")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error al leer el archivo: {e}")

    # Admin: guardar en Supabase Storage para que sea el precargado permanente
    if has_permission('admin') and st.session_state.get('_conc_pending_bytes'):
        if st.button("☁️ Guardar como precargado en Supabase", type="primary"):
            with st.spinner("Subiendo a Supabase Storage..."):
                try:
                    sio.upload_ventas_precargado(st.session_state['_conc_pending_bytes'])
                    del st.session_state['_conc_pending_bytes']
                    _cargar_precargado.clear()
                    st.success("✅ veh_ml_features.xlsx guardado en Supabase. Ahora es el archivo precargado para todos los usuarios.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error al subir a Supabase: {e}")

# Resolver fuente activa
if _tiene_custom:
    _df_source = st.session_state['df_concesionarios']
elif _precargado is not None:
    _df_source = _precargado
else:
    st.error("❌ No hay datos disponibles. Carga un Excel usando el expander de arriba.")
    st.stop()

# ── Preparar DataFrame ────────────────────────────────────────────────────────

df_raw = _df_source.copy()
if 'MARCA' in df_raw.columns:
    df_raw = df_raw[df_raw['MARCA'] == 'CHERY']

# Detectar columnas
conc_col   = next((c for c in ['CONCESIONARIO', 'DET_CC', 'AGE', 'SUCURSAL'] if c in df_raw.columns), None)
modelo_col = 'MODELO_NORM' if 'MODELO_NORM' in df_raw.columns else None
fecha_col  = 'FECHA_VENTA' if 'FECHA_VENTA' in df_raw.columns else None

if not conc_col or len(df_raw) == 0:
    st.error("⚠️ No se encontró columna CONCESIONARIO o no hay registros CHERY.")
    st.stop()

# ── Filtros inline ────────────────────────────────────────────────────────────

df = df_raw.copy()
_f1, _f2, _f3 = st.columns([2, 2, 4])

if fecha_col:
    years_all = sorted(df[fecha_col].dt.year.dropna().unique().astype(int), reverse=True)
    years_sel = _f1.multiselect("", years_all, default=years_all,
                                placeholder="Año", label_visibility="collapsed")
    if years_sel:
        df = df[df[fecha_col].dt.year.isin(years_sel)]
else:
    years_sel = []

if modelo_col:
    modelos_all = ['Todos'] + sorted(df[modelo_col].dropna().unique())
    modelo_sel = _f2.selectbox("", modelos_all, label_visibility="collapsed")
    if modelo_sel != 'Todos':
        df = df[df[modelo_col] == modelo_sel]
else:
    modelo_sel = 'Todos'

concs_all = sorted(df[conc_col].dropna().unique())
concs_sel = _f3.multiselect("", concs_all, default=concs_all,
                            placeholder="Concesionarios", label_visibility="collapsed")
if concs_sel:
    df = df[df[conc_col].isin(concs_sel)]

if len(df) == 0:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# ── KPIs globales ─────────────────────────────────────────────────────────────

ventas_por_conc = df.groupby(conc_col).size().sort_values(ascending=False)
top_conc        = ventas_por_conc.index[0]
top_modelo      = df[modelo_col].value_counts().index[0] if modelo_col else '—'
last_month_str  = df[fecha_col].max().strftime('%b %Y') if fecha_col else '—'

k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi_card("Total Ventas CHERY", f"{len(df):,}", "📦"), unsafe_allow_html=True)
k2.markdown(kpi_card("Concesionarios", len(ventas_por_conc), "🏪", "blue"), unsafe_allow_html=True)
k3.markdown(kpi_card("Top Concesionario", top_conc, "🥇", "amber"), unsafe_allow_html=True)
k4.markdown(kpi_card("Último Dato", last_month_str, "📅"), unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_hist, tab_evo, tab_pred, tab_tabla = st.tabs([
    "📊 Resumen", "📈 Evolución Mensual", "🔮 Predicciones por Tienda", "📋 Detalle"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESUMEN HISTÓRICO
# ══════════════════════════════════════════════════════════════════════════════

with tab_hist:
    st.markdown(section_header("Ventas Históricas por Concesionario", "📊"), unsafe_allow_html=True)

    # Barras horizontales
    df_bar = ventas_por_conc.reset_index()
    df_bar.columns = ['Concesionario', 'Ventas']
    df_bar['% Total'] = (df_bar['Ventas'] / df_bar['Ventas'].sum() * 100).round(1)
    df_bar['Label'] = df_bar.apply(lambda r: f"{r['Ventas']:,}  ({r['% Total']:.1f}%)", axis=1)

    fig_bar = go.Figure(go.Bar(
        y=df_bar['Concesionario'], x=df_bar['Ventas'],
        orientation='h',
        text=df_bar['Label'],
        textposition='outside',
        textfont=dict(color='#94A3B8', size=11),
        marker=dict(
            color=COLORS['series'][:len(df_bar)],
            line=dict(width=0),
        ),
    ))
    apply_chart_theme(fig_bar, height=max(280, 60 + len(df_bar) * 52),
                      title='Ventas Totales por Concesionario')
    fig_bar.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Unidades vendidas',
        margin=dict(r=160),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    # Mapa geográfico de concesionarios
    st.markdown(section_header("Ubicación Geográfica de Concesionarios", "🗺️"), unsafe_allow_html=True)

    _map_rows = []
    for conc, ventas in ventas_por_conc.items():
        coords = _coords_concesionario(conc)
        if coords:
            _map_rows.append({
                'Concesionario': conc,
                'Ventas': int(ventas),
                'lat': coords[0],
                'lon': coords[1],
                '% Total': round(ventas / ventas_por_conc.sum() * 100, 1),
            })

    if _map_rows:
        df_map = pd.DataFrame(_map_rows)
        _geo = _get_peru_geojson()

        # Detectar la clave de nombre de departamento en el GeoJSON
        _prop_key = None
        if _geo and _geo.get('features'):
            _fp = _geo['features'][0].get('properties', {})
            _prop_key = next(
                (k for k in ['DEPARTAMEN', 'NOMBDEP', 'NAME_1', 'NOMBRE', 'NOM_DEP', 'name'] if k in _fp),
                None
            )

        # Paleta extendida para el mapa (>8 colores garantiza unicidad)
        _MAP_COLORS = [
            '#0073FF', '#C2FF00', '#00F5A0', '#FF3A5C',
            '#A78BFA', '#F97316', '#38BDF8', '#FB7185',
            '#FBBF24', '#34D399', '#E879F9', '#F43F5E',
            '#22D3EE', '#84CC16', '#FB923C', '#818CF8',
        ]

        if _geo and _prop_key:
            # ── Choropleth por departamentos ──────────────────────────────────
            _all_depts = [f['properties'][_prop_key] for f in _geo['features']]
            # Índice normalizado (sin acentos, uppercase) → nombre real en GeoJSON
            _all_norm = {_norm_dpto(d): d for d in _all_depts}

            # Mapear cada concesionario a su departamento GeoJSON
            _conc_dpto: dict[str, str] = {}
            for row in _map_rows:
                for ciudad, dpto_key in _CIUDAD_TO_DPTO.items():
                    if ciudad in row['Concesionario'].lower():
                        matched = _all_norm.get(_norm_dpto(dpto_key))
                        if matched:
                            _conc_dpto[row['Concesionario']] = matched
                        break

            # Asignar color index a cada departamento destacado (orden desc. ventas)
            _highlighted: dict[str, int] = {}
            for row in sorted(_map_rows, key=lambda r: r['Ventas'], reverse=True):
                dpto = _conc_dpto.get(row['Concesionario'])
                if dpto and dpto not in _highlighted:
                    _highlighted[dpto] = len(_highlighted)

            fig_map = go.Figure()

            # Traza base: departamentos sin concesionario (gris oscuro)
            _bg_depts = [d for d in _all_depts if d not in _highlighted]
            if _bg_depts:
                fig_map.add_trace(go.Choroplethmapbox(
                    geojson=_geo,
                    locations=_bg_depts,
                    z=[0] * len(_bg_depts),
                    featureidkey=f'properties.{_prop_key}',
                    colorscale=[[0, '#1A2742'], [1, '#1A2742']],
                    zmin=0, zmax=1,
                    showscale=False,
                    showlegend=False,
                    marker=dict(opacity=0.80, line=dict(color='rgba(160,190,220,0.20)', width=0.6)),
                    hovertemplate='<b>%{location}</b><extra></extra>',
                ))

            # Una traza por cada departamento destacado (color propio)
            for dpto, cidx in _highlighted.items():
                color = _MAP_COLORS[cidx % len(_MAP_COLORS)]
                fig_map.add_trace(go.Choroplethmapbox(
                    geojson=_geo,
                    locations=[dpto],
                    z=[1],
                    featureidkey=f'properties.{_prop_key}',
                    colorscale=[[0, color], [1, color]],
                    zmin=0, zmax=1,
                    showscale=False,
                    showlegend=False,
                    marker=dict(opacity=0.85, line=dict(color='rgba(200,220,240,0.35)', width=0.8)),
                    hovertemplate='<b>%{location}</b><extra></extra>',
                ))

            # Puntos + nombres encima
            for i, row in enumerate(_map_rows):
                color = _MAP_COLORS[
                    _highlighted.get(_conc_dpto.get(row['Concesionario'], ''), i)
                    % len(_MAP_COLORS)
                ]
                fig_map.add_trace(go.Scattermapbox(
                    lat=[row['lat']], lon=[row['lon']],
                    mode='markers+text',
                    marker=dict(size=13, color=color, opacity=1.0),
                    text=[f"  {row['Concesionario'].split()[-1].upper()}"],
                    textposition='middle right',
                    textfont=dict(color='white', size=12, family='Arial Black'),
                    name=row['Concesionario'],
                    hovertemplate=(
                        f"<b>{row['Concesionario']}</b><br>"
                        f"Ventas: {row['Ventas']:,}<br>"
                        f"Share: {row['% Total']}%<extra></extra>"
                    ),
                ))

            fig_map.update_layout(
                mapbox_style='carto-darkmatter',
                mapbox_zoom=4.5,
                mapbox_center={'lat': -9.19, 'lon': -75.0},
                paper_bgcolor='#080D18',
                height=520,
                margin=dict(l=0, r=0, t=36, b=0),
                legend=dict(
                    bgcolor='rgba(8,13,24,0.85)',
                    bordercolor='rgba(255,255,255,0.12)',
                    borderwidth=1,
                    font=dict(color='#C9D8E6', size=12),
                    title_text='Concesionario',
                    x=0.01, y=0.01,
                    xanchor='left', yanchor='bottom',
                ),
            )
        else:
            # Fallback: scatter sobre open-street-map si falla el fetch GeoJSON
            fig_map = px.scatter_mapbox(
                df_map, lat='lat', lon='lon',
                size='Ventas', color='Concesionario',
                hover_name='Concesionario',
                hover_data={'Ventas': True, '% Total': True, 'lat': False, 'lon': False},
                color_discrete_sequence=COLORS['series'][:len(df_map)],
                size_max=55, zoom=5.5,
                center={'lat': -9.19, 'lon': -75.0},
                mapbox_style='open-street-map', height=520,
            )
            fig_map.update_layout(
                paper_bgcolor='#080D18', plot_bgcolor='#080D18',
                margin=dict(l=0, r=0, t=36, b=0),
                legend=dict(
                    bgcolor='rgba(8,13,24,0.85)', bordercolor='rgba(255,255,255,0.12)',
                    borderwidth=1, font=dict(color='#C9D8E6', size=12),
                    x=0.01, y=0.01, xanchor='left', yanchor='bottom',
                ),
                mapbox_layers=[{
                    "sourcetype": "geojson", "source": _PERU_BORDER_GEOJSON,
                    "type": "line", "color": "#F59E0B", "line": {"width": 2},
                }],
            )

        st.plotly_chart(fig_map, use_container_width=True, config={
            'displayModeBar': True,
            'modeBarButtonsToKeep': ['zoomInMapbox', 'zoomOutMapbox', 'resetViewMapbox'],
            'displaylogo': False,
        })
    else:
        st.info(
            "ℹ️ No se identificaron ciudades en los nombres de los concesionarios. "
            "Para mostrar el mapa, el nombre debe incluir la ciudad (ej. 'Lima', 'Piura', 'Chiclayo')."
        )

    # Distribución de modelos por concesionario
    if modelo_col:
        st.markdown(section_header("Distribución de Modelos por Concesionario", "🚗"), unsafe_allow_html=True)
        df_mod = df.groupby([conc_col, modelo_col]).size().reset_index(name='Ventas')
        fig_mod = px.bar(
            df_mod, x=conc_col, y='Ventas', color=modelo_col,
            barmode='stack',
            color_discrete_sequence=COLORS['series'],
        )
        apply_chart_theme(fig_mod, height=400, title='Mix de Modelos por Concesionario')
        fig_mod.update_layout(
            xaxis_tickangle=-20, xaxis_title='', yaxis_title='Unidades',
            legend_title='Modelo',
        )
        st.plotly_chart(fig_mod, use_container_width=True, config={'displayModeBar': False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVOLUCIÓN MENSUAL
# ══════════════════════════════════════════════════════════════════════════════

with tab_evo:
    st.markdown(section_header("Evolución Mensual por Concesionario", "📈"), unsafe_allow_html=True)

    if not fecha_col:
        st.warning("Se requiere columna FECHA-VENTA para este gráfico.")
    else:
        df_ts = (
            df.groupby([pd.Grouper(key=fecha_col, freq='ME'), conc_col])
            .size().reset_index(name='Ventas')
        )

        fig_evo = px.line(
            df_ts, x=fecha_col, y='Ventas', color=conc_col,
            markers=True,
            color_discrete_sequence=COLORS['series'],
        )
        apply_chart_theme(fig_evo, height=480, title='Ventas Mensuales por Concesionario')
        fig_evo.update_layout(
            hovermode='x unified',
            xaxis_title='Mes', yaxis_title='Unidades',
            legend_title='Concesionario',
        )
        st.plotly_chart(fig_evo, use_container_width=True, config={'displayModeBar': False})

        # Share mensual (100% stacked area)
        st.markdown(section_header("Share Mensual por Concesionario (%)", "📐"), unsafe_allow_html=True)
        df_pivot = df_ts.pivot_table(index=fecha_col, columns=conc_col, values='Ventas', fill_value=0)
        df_pct   = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100

        fig_share = go.Figure()
        for i, col_name in enumerate(df_pct.columns):
            fig_share.add_trace(go.Scatter(
                x=df_pct.index, y=df_pct[col_name],
                mode='lines', name=col_name,
                stackgroup='one',
                line=dict(color=COLORS['series'][i % len(COLORS['series'])], width=0),
                fillcolor='rgba({},{},{},0.75)'.format(*bytes.fromhex(COLORS['series'][i % len(COLORS['series'])].lstrip('#'))),
                hovertemplate='%{y:.1f}%<extra>' + col_name + '</extra>',
            ))
        apply_chart_theme(fig_share, height=320, title='Share de Mercado Mensual (%)')
        fig_share.update_layout(
            hovermode='x unified',
            xaxis_title='Mes', yaxis_title='%',
            yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig_share, use_container_width=True, config={'displayModeBar': False})

        # Crecimiento MoM
        if len(df_pivot) >= 2:
            st.markdown(section_header("Crecimiento Mes a Mes (%)", "🔺"), unsafe_allow_html=True)
            df_mom = df_pivot.pct_change() * 100
            fig_mom = go.Figure()
            for i, col_name in enumerate(df_mom.columns):
                fig_mom.add_trace(go.Bar(
                    x=df_mom.index, y=df_mom[col_name],
                    name=col_name,
                    marker_color=COLORS['series'][i % len(COLORS['series'])],
                ))
            apply_chart_theme(fig_mom, height=320, title='Variación MoM por Concesionario')
            fig_mom.update_layout(
                barmode='group',
                hovermode='x unified',
                xaxis_title='Mes', yaxis_title='%',
                yaxis_ticksuffix='%',
            )
            st.plotly_chart(fig_mom, use_container_width=True, config={'displayModeBar': False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICCIONES POR CONCESIONARIO
# ══════════════════════════════════════════════════════════════════════════════

with tab_pred:
    st.markdown(section_header("Predicciones por Concesionario", "🔮"), unsafe_allow_html=True)

    # ── Banner metodología ────────────────────────────────────────────────────
    st.markdown("""
<div style="background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.25);
            border-radius:10px;padding:16px 20px;margin-bottom:18px;">
<span style="font-size:1.05rem;font-weight:600;color:#A78BFA;">Metodología — Asignación por shares históricos</span><br>
<span style="color:#94A3B8;font-size:0.92rem;">
El modelo SARIMA predice el <strong style="color:#C9D8E6;">total nacional</strong> de ventas TIGGO 2.
Para desglosar por concesionario se calcula el <strong style="color:#C9D8E6;">share de los últimos 12 meses</strong>
de cada tienda y se aplica como ponderación sobre la predicción total y sus intervalos de confianza.<br>
<strong style="color:#A78BFA;">Supuesto:</strong> la distribución relativa entre concesionarios se mantiene estable
en el horizonte de predicción. Si hay cambios estructurales (apertura/cierre de tiendas, campañas locales),
ajusta los shares manualmente en la tabla de abajo.
</span>
</div>
""", unsafe_allow_html=True)

    if not fecha_col:
        st.warning("Se requiere columna FECHA-VENTA para calcular predicciones por concesionario.")
    else:
        # ── Calcular shares ───────────────────────────────────────────────────
        last_date    = df_raw[fecha_col].max()
        cutoff_12m   = last_date - pd.DateOffset(months=12)
        # Filtrar últimos 12 meses de todos los concesionarios (sin filtro de conc)
        df_12m = df_raw[df_raw[fecha_col] >= cutoff_12m]
        if 'MARCA' in df_raw.columns:
            df_12m = df_12m[df_12m['MARCA'] == 'CHERY']
        if modelo_col and modelo_sel != 'Todos':
            df_12m = df_12m[df_12m[modelo_col] == modelo_sel]

        n_12m = len(df_12m)
        if n_12m == 0:
            st.warning("No hay datos en los últimos 12 meses para calcular shares.")
            st.stop()

        shares_raw  = df_12m.groupby(conc_col).size()
        shares_pct  = (shares_raw / shares_raw.sum())

        # Filtrar solo concesionarios seleccionados en sidebar
        if concs_sel:
            shares_pct = shares_pct[shares_pct.index.isin(concs_sel)]
            shares_pct = shares_pct / shares_pct.sum()   # renormalizar

        # ── Editor de shares ──────────────────────────────────────────────────
        with st.expander("⚙️ Ajustar shares manualmente"):
            st.caption("Edita el share (%) para simular cambios en la distribución. Deben sumar 100%.")
            shares_edit_df = pd.DataFrame({
                'Concesionario': shares_pct.index,
                'Share (%) — últimos 12 m': (shares_pct.values * 100).round(2),
                'Share ajustado (%)': (shares_pct.values * 100).round(2),
            })
            edited = st.data_editor(shares_edit_df, hide_index=True, use_container_width=True,
                                    column_config={
                                        'Share (%) — últimos 12 m': st.column_config.NumberColumn(disabled=True, format="%.2f%%"),
                                        'Share ajustado (%)': st.column_config.NumberColumn(min_value=0, max_value=100, format="%.2f%%"),
                                    })
            total_adj = edited['Share ajustado (%)'].sum()
            if abs(total_adj - 100) > 0.5:
                st.warning(f"Los shares suman {total_adj:.1f}% — deben sumar 100%.")
            else:
                shares_pct = pd.Series(
                    edited['Share ajustado (%)'].values / 100,
                    index=edited['Concesionario'].values
                )

        # ── Construir tabla de predicciones por concesionario ─────────────────
        pred_rows = []
        for conc, share in shares_pct.items():
            for _, row in pred_total.iterrows():
                pred_rows.append({
                    'Fecha':          row['Fecha'],
                    'Mes':            row['Mes'],
                    'Concesionario':  conc,
                    'Share (%)':      round(share * 100, 1),
                    'Predicción':     row['Predicción'] * share,
                    'IC_Inferior':    row['IC_Inferior'] * share,
                    'IC_Superior':    row['IC_Superior'] * share,
                })
        df_pred_conc = pd.DataFrame(pred_rows)

        # Regenerar Mes en español (el modelo guardado puede tenerlo en inglés)
        _MESES_ES = ['Enero','Febrero','Marzo','Abril','Mayo','Junio',
                     'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
        df_pred_conc['Mes'] = df_pred_conc['Fecha'].apply(
            lambda d: f"{_MESES_ES[d.month - 1]} {d.year}"
        )

        # ── KPIs del próximo mes ──────────────────────────────────────────────
        _hoy = pd.Timestamp.today().normalize()
        _fechas_futuras = df_pred_conc[df_pred_conc['Fecha'] >= _hoy]['Fecha']
        _next_date = _fechas_futuras.min() if not _fechas_futuras.empty else df_pred_conc['Fecha'].max()
        pred_next   = df_pred_conc[df_pred_conc['Fecha'] == _next_date]
        pred_total_next = int(pred_next['Predicción'].sum())

        st.markdown(f"### Predicción próximo mes — **{pred_next['Mes'].iloc[0]}**")

        cols_kpi = st.columns(len(pred_next))
        for idx, (_, row) in enumerate(pred_next.sort_values('Predicción', ascending=False).iterrows()):
            cols_kpi[idx].markdown(
                kpi_card(
                    row['Concesionario'],
                    f"{row['Predicción']:.0f} uds",
                    "🏪",
                    "blue" if idx == 0 else ("amber" if idx == 1 else ""),
                    sub=f"IC 95%: {row['IC_Inferior']:.0f}–{row['IC_Superior']:.0f}",
                ),
                unsafe_allow_html=True,
            )

        # ── Gráfico: histórico + predicción por concesionario ─────────────────
        st.markdown(section_header("Histórico + Predicción por Concesionario", "📊"), unsafe_allow_html=True)

        if fecha_col:
            df_hist_ts = (
                df.groupby([pd.Grouper(key=fecha_col, freq='ME'), conc_col])
                .size().reset_index(name='Ventas')
            )
        else:
            df_hist_ts = pd.DataFrame()

        fig_main = go.Figure()

        # Línea vertical: inicio predicción
        if not df_hist_ts.empty:
            last_hist_date = df_hist_ts[fecha_col].max()
            fig_main.add_shape(
                type="line",
                x0=last_hist_date, x1=last_hist_date,
                y0=0, y1=1, yref="paper",
                line=dict(color='rgba(100,116,139,0.5)', width=1.5, dash="dot"),
            )
            fig_main.add_annotation(
                x=last_hist_date, y=1, yref="paper",
                text="Predicción ▶", showarrow=False,
                font=dict(color='#64748B', size=10),
                xshift=8, xanchor='left',
            )

        _IC_FILLS = [
            'rgba(0,115,255,0.08)', 'rgba(194,255,0,0.08)', 'rgba(0,245,160,0.08)',
            'rgba(255,58,92,0.08)', 'rgba(167,139,250,0.08)', 'rgba(249,115,22,0.08)',
            'rgba(56,189,248,0.08)', 'rgba(251,113,133,0.08)',
        ]

        for i, conc in enumerate(shares_pct.index):
            color    = COLORS['series'][i % len(COLORS['series'])]
            ic_fill  = _IC_FILLS[i % len(_IC_FILLS)]

            # Histórico
            if not df_hist_ts.empty:
                hist_conc = df_hist_ts[df_hist_ts[conc_col] == conc]
                if not hist_conc.empty:
                    fig_main.add_trace(go.Scatter(
                        x=hist_conc[fecha_col], y=hist_conc['Ventas'],
                        mode='lines+markers', name=f'{conc} — Real',
                        line=dict(color=color, width=2),
                        marker=dict(size=5, color=color),
                        legendgroup=conc,
                    ))

            # Predicción + IC band
            pred_conc = df_pred_conc[df_pred_conc['Concesionario'] == conc]

            # Banda IC
            fig_main.add_trace(go.Scatter(
                x=pred_conc['Fecha'].tolist() + pred_conc['Fecha'].tolist()[::-1],
                y=pred_conc['IC_Superior'].tolist() + pred_conc['IC_Inferior'].tolist()[::-1],
                fill='toself',
                fillcolor=ic_fill,
                line=dict(color='rgba(0,0,0,0)'),
                name=f'{conc} — IC 95%',
                legendgroup=conc,
                showlegend=False,
            ))

            # Línea de predicción
            fig_main.add_trace(go.Scatter(
                x=pred_conc['Fecha'], y=pred_conc['Predicción'],
                mode='lines+markers', name=f'{conc} — Predicción',
                line=dict(color=color, width=2.5, dash='dot'),
                marker=dict(size=9, symbol='diamond', color=color,
                            line=dict(color='#080D18', width=1.5)),
                legendgroup=conc,
                hovertemplate=(
                    f'<b>{conc}</b><br>'
                    'Predicción: %{y:.0f} uds<br>'
                    'Fecha: %{x|%b %Y}<extra></extra>'
                ),
            ))

        apply_chart_theme(fig_main, height=540,
                          title='Histórico + Predicción SARIMA — Por Concesionario')
        fig_main.update_layout(
            hovermode='x unified',
            xaxis_title='Fecha', yaxis_title='Unidades',
            legend=dict(groupclick='toggleitem'),
        )
        st.plotly_chart(fig_main, use_container_width=True, config={'displayModeBar': False})

        # ── Barras agrupadas: horizonte completo ──────────────────────────────
        st.markdown(section_header("Horizonte de Predicción — Barras por Mes", "📅"), unsafe_allow_html=True)

        fig_hor = go.Figure()
        for i, conc in enumerate(shares_pct.index):
            pred_conc = df_pred_conc[df_pred_conc['Concesionario'] == conc]
            fig_hor.add_trace(go.Bar(
                x=pred_conc['Mes'], y=pred_conc['Predicción'].round(1),
                name=conc,
                marker_color=COLORS['series'][i % len(COLORS['series'])],
                text=pred_conc['Predicción'].round(0).astype(int),
                textposition='inside',
                textfont=dict(size=10, color='#080D18'),
                hovertemplate=(
                    f'<b>{conc}</b><br>%{{x}}<br>'
                    'Predicción: %{y:.0f} uds<br>'
                    'IC 95%: %{customdata[0]:.0f}–%{customdata[1]:.0f}<extra></extra>'
                ),
                customdata=np.column_stack([
                    pred_conc['IC_Inferior'].round(0).values,
                    pred_conc['IC_Superior'].round(0).values,
                ]),
            ))
        apply_chart_theme(fig_hor, height=420,
                          title='Predicción Mensual por Concesionario — Horizonte Completo')
        fig_hor.update_layout(
            barmode='stack',
            xaxis_title='Mes', yaxis_title='Unidades',
            legend_title='Concesionario',
        )
        st.plotly_chart(fig_hor, use_container_width=True, config={'displayModeBar': False})

        # ── Tabla resumen ─────────────────────────────────────────────────────
        st.markdown(section_header("Tabla de Predicciones", "📋"), unsafe_allow_html=True)

        df_tabla_pred = df_pred_conc[['Mes', 'Concesionario', 'Share (%)', 'Predicción', 'IC_Inferior', 'IC_Superior']].copy()
        df_tabla_pred['Predicción']  = df_tabla_pred['Predicción'].round(1)
        df_tabla_pred['IC_Inferior'] = df_tabla_pred['IC_Inferior'].round(1)
        df_tabla_pred['IC_Superior'] = df_tabla_pred['IC_Superior'].round(1)
        df_tabla_pred = df_tabla_pred.sort_values(['Mes', 'Predicción'], ascending=[True, False])

        # Agregar fila de totales por mes
        totales = df_pred_conc.groupby('Mes').agg(
            Predicción=('Predicción', 'sum'),
            IC_Inferior=('IC_Inferior', 'sum'),
            IC_Superior=('IC_Superior', 'sum'),
        ).reset_index()
        totales['Concesionario'] = 'TOTAL'
        totales['Share (%)'] = 100.0
        totales['Predicción']  = totales['Predicción'].round(1)
        totales['IC_Inferior'] = totales['IC_Inferior'].round(1)
        totales['IC_Superior'] = totales['IC_Superior'].round(1)

        df_tabla_final = pd.concat(
            [df_tabla_pred, totales[df_tabla_pred.columns]],
            ignore_index=True
        ).sort_values(['Mes', 'Concesionario'])

        st.dataframe(
            df_tabla_final.style
                .background_gradient(subset=['Predicción'], cmap='Blues')
                .format({'Predicción': '{:.1f}', 'IC_Inferior': '{:.1f}',
                         'IC_Superior': '{:.1f}', 'Share (%)': '{:.1f}%'}),
            use_container_width=True, hide_index=True,
        )

        if has_permission('exportar'):
            csv_out = df_tabla_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Exportar predicciones CSV",
                csv_out,
                f"pred_concesionarios_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DETALLE HISTÓRICO
# ══════════════════════════════════════════════════════════════════════════════

with tab_tabla:
    st.markdown(section_header("Ranking y Detalle por Concesionario", "📋"), unsafe_allow_html=True)

    df_rank = ventas_por_conc.reset_index()
    df_rank.columns = ['Concesionario', 'Ventas']
    df_rank['% Total']     = (df_rank['Ventas'] / df_rank['Ventas'].sum() * 100).round(1)
    df_rank['Acumulado %'] = df_rank['% Total'].cumsum().round(1)
    st.dataframe(
        df_rank.style
               .background_gradient(subset=['Ventas'], cmap='Blues')
               .format({'% Total': '{:.1f}%', 'Acumulado %': '{:.1f}%'}),
        use_container_width=True, hide_index=True,
    )

    if fecha_col:
        st.markdown(section_header("Ventas Mensuales por Concesionario", "📅"), unsafe_allow_html=True)
        df_monthly = (
            df.groupby([pd.Grouper(key=fecha_col, freq='ME'), conc_col])
            .size().unstack(fill_value=0)
        )
        df_monthly.index = df_monthly.index.strftime('%b %Y')
        st.dataframe(
            df_monthly.style.background_gradient(cmap='Blues', axis=None),
            use_container_width=True,
        )

        if has_permission('exportar'):
            csv_hist = df_monthly.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Exportar histórico CSV",
                csv_hist,
                f"historico_concesionarios_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
            )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="app-footer">Sistema TIGGO 2 &nbsp;·&nbsp; ISDI &nbsp;·&nbsp; Concesionarios</div>',
    unsafe_allow_html=True,
)
