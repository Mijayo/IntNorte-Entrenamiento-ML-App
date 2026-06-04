"""
============================================================================
MÓDULO: I/O CON SUPABASE STORAGE + DATABASE
Bucket : configurado en st.secrets["supabase"]["bucket"]
DB     : tabla training_runs en Supabase PostgreSQL
============================================================================
"""

import gzip
import io
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import matplotlib.figure
import pandas as pd
import streamlit as st
from supabase import create_client

from .logger import get_logger

log = get_logger("supabase_io")

_TABLE         = "training_runs"
_AUDIT_TABLE   = "audit_log"
_VENTAS_TABLE  = "ventas_reales"

_ARTIFACT_NAMES = [
    "modelo_total_mejorado.pkl.gz",
    "prediccion_total_mejorada.xlsx",
    "grid_search_results.xlsx",
    "walk_forward_validation.xlsx",
    "historico_total_mejorado.xlsx",
    "historico_exog.xlsx",
    "metricas_mejoradas.json",
    "acf_plot.png",
    "pacf_plot.png",
    "llm_cache.json",
    "cml_resultados.json",
]


# ── Clientes ─────────────────────────────────────────────────────────────────

@st.cache_resource
def get_client():
    """Cliente con anon key — usado para Supabase Auth (login/logout)."""
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"]
    )


@st.cache_resource
def _get_service_client():
    """Cliente con service role key — bypasea RLS para Storage y DB.
    Seguro en Streamlit porque es server-side y nunca llega al navegador."""
    cfg = st.secrets["supabase"]
    service_key = cfg.get("service_key")
    if not service_key:
        raise RuntimeError(
            "supabase.service_key no está configurado en secrets.toml. "
            "Las operaciones de Storage y DB requieren la service role key."
        )
    return create_client(cfg["url"], service_key)


def _bucket() -> str:
    return st.secrets["supabase"]["bucket"]


def _db():
    """Acceso directo a la tabla training_runs (service role — sin RLS)."""
    return _get_service_client().table(_TABLE)


def _audit():
    """Acceso directo a la tabla audit_log (service role — sin RLS)."""
    return _get_service_client().table(_AUDIT_TABLE)


def _ventas():
    """Acceso directo a la tabla ventas_reales (service role — sin RLS)."""
    return _get_service_client().table(_VENTAS_TABLE)


# ── Primitivas de I/O (Storage) ──────────────────────────────────────────────

def _upload(path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """Sube bytes a Supabase Storage (sobrescribe atómicamente con upsert)."""
    _get_service_client().storage.from_(_bucket()).upload(
        path, data, {"content-type": content_type, "upsert": "true"}
    )
    log.debug("Upload OK: %s (%d bytes)", path, len(data))


def _download(path: str) -> bytes:
    """Descarga bytes de Supabase Storage."""
    data = _get_service_client().storage.from_(_bucket()).download(path)
    log.debug("Download OK: %s (%d bytes)", path, len(data))
    return data


# ── Datos precargados (Storage) ──────────────────────────────────────────────

_PRELOADED_STORAGE_VENTAS    = "preloaded/veh_ml_features.xlsx"
_PRELOADED_STORAGE_STOCK     = "preloaded/Stock Vehiculos.xlsx"

_EXCEL_CT = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@st.cache_data(ttl=3600, show_spinner=False)
def load_datos_precargados() -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Descarga los Excel de datos precargados desde Supabase Storage.

    Paths en el bucket:
      - preloaded/veh_ml_features.xlsx   (sheet 'Hoja1')
      - preloaded/Stock Vehiculos.xlsx   (sheet 'Stock Actual')
    """
    raw_v = _download(_PRELOADED_STORAGE_VENTAS)
    raw_s = _download(_PRELOADED_STORAGE_STOCK)
    df_v = pd.read_excel(io.BytesIO(raw_v), sheet_name="Hoja1", engine="openpyxl")
    df_s = pd.read_excel(io.BytesIO(raw_s), sheet_name="Stock Actual", engine="openpyxl")
    return df_v, df_s


def upload_datos_precargados(ventas_bytes: bytes, stock_bytes: bytes) -> None:
    """Sube los Excel de datos precargados a Supabase Storage (admin)."""
    _upload(_PRELOADED_STORAGE_VENTAS, ventas_bytes, _EXCEL_CT)
    _upload(_PRELOADED_STORAGE_STOCK,  stock_bytes,  _EXCEL_CT)
    load_datos_precargados.clear()
    log.info("Datos precargados actualizados en Supabase Storage")


def upload_ventas_precargado(ventas_bytes: bytes) -> None:
    """Sube veh_ml_features.xlsx a Supabase Storage (admin). Invalida el caché."""
    _upload(_PRELOADED_STORAGE_VENTAS, ventas_bytes, _EXCEL_CT)
    load_datos_precargados.clear()
    log.info("veh_ml_features.xlsx actualizado en Supabase Storage")


# ── Gestión de runs ──────────────────────────────────────────────────────────

def _run_exists(run_name: str) -> bool:
    """Comprueba si un run tiene artefactos en Supabase Storage."""
    try:
        files = _get_service_client().storage.from_(_bucket()).list(run_name)
        return len(files) > 0
    except Exception:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def get_available_runs() -> list[str]:
    """Lista de runs disponibles (más reciente primero).
    Fuente primaria: tabla DB. Fallback: training_log.json.
    Filtra runs sin artefactos en Storage con una sola llamada al bucket."""
    try:
        rows = (
            _db()
            .select("run_name")
            .order("created_at", desc=True)
            .execute()
            .data
        )
        runs = [r["run_name"] for r in rows]
    except Exception as e:
        log.warning("DB no disponible, usando training_log.json: %s", e)
        try:
            log_data = json.loads(_download("training_log.json"))
            seen: dict[str, bool] = {}
            for entry in reversed(log_data):
                rn = entry.get("run_name")
                if rn and rn not in seen:
                    seen[rn] = True
            runs = list(seen.keys())
        except Exception as e2:
            log.debug("training_log.json tampoco disponible: %s", e2)
            return []

    # Una sola llamada al bucket en lugar de N+1 llamadas individuales
    try:
        items = _get_service_client().storage.from_(_bucket()).list("")
        existing = {item["name"] for item in items if item.get("name")}
        return [rn for rn in runs if rn in existing]
    except Exception as e:
        log.warning("No se pudo listar bucket, usando _run_exists: %s", e)
        return [rn for rn in runs if _run_exists(rn)]


@st.cache_data(ttl=300, show_spinner=False)
def get_default_run(runs: list[str]) -> str | None:
    """Run activo: primero busca activo=TRUE en DB, luego latest.txt, luego el más reciente."""
    try:
        rows = _db().select("run_name").eq("activo", True).limit(1).execute().data
        if rows:
            candidate = rows[0]["run_name"]
            if candidate in runs:
                return candidate
    except Exception as e:
        log.debug("No se pudo consultar activo en DB: %s", e)
    try:
        candidate = _download("latest.txt").decode().strip()
        if candidate in runs:
            return candidate
    except Exception as e:
        log.debug("latest.txt no disponible: %s", e)
    return runs[0] if runs else None


def approve_model(run_name: str, usuario: str | None = None) -> None:
    """Activa un run como modelo de producción.
    Marca activo=TRUE en DB y actualiza latest.txt en Storage."""
    try:
        # Activar el target primero — si falla el segundo UPDATE, al menos hay un modelo activo
        _db().update({"activo": True}).eq("run_name", run_name).execute()
        _db().update({"activo": False}).neq("run_name", run_name).execute()
        log.info("Modelo activado en DB: run='%s'", run_name)
    except Exception as e:
        log.warning("No se pudo actualizar activo en DB: %s", e)
    _upload("latest.txt", run_name.encode(), "text/plain")
    log_audit(usuario, "APPROVE_MODEL", run_name=run_name)
    st.cache_data.clear()
    log.info("Modelo activado en producción: run='%s'", run_name)


def delete_run(run_name: str, usuario: str | None = None) -> None:
    """Elimina un run de DB y sus artefactos en Storage."""
    # 1. Borrar artefactos de Storage
    try:
        paths = [f"{run_name}/{name}" for name in _ARTIFACT_NAMES]
        _get_service_client().storage.from_(_bucket()).remove(paths)
        log.info("Artefactos de Storage eliminados: run='%s'", run_name)
    except Exception as e:
        log.warning("No se pudieron eliminar artefactos de Storage para run='%s': %s", run_name, e)

    # 2. Borrar de DB
    try:
        _db().delete().eq("run_name", run_name).execute()
        log_audit(usuario, "DELETE_RUN", run_name=run_name)
        log.info("Run '%s' eliminado de DB", run_name)
    except Exception as e:
        log.error("No se pudo eliminar run '%s' de DB: %s", run_name, e)


@st.cache_data(ttl=300, show_spinner=False)
def get_runs_df() -> pd.DataFrame:
    """DataFrame con todos los runs y sus métricas para análisis comparativo."""
    try:
        rows = (
            _db()
            .select("*")
            .order("created_at", desc=True)
            .execute()
            .data
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["created_at"] = pd.to_datetime(df["created_at"])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        log.warning("No se pudo cargar runs de DB: %s", e)
        return pd.DataFrame()


def format_run_label(run_name: str) -> str:
    """Formatea 20260322_143000 → 22/03/2026  14:30."""
    try:
        dt = datetime.strptime(run_name, "%Y%m%d_%H%M%S")
        return dt.strftime("%d/%m/%Y  %H:%M")
    except ValueError:
        return run_name


# ── Guardar artefactos (Storage) ─────────────────────────────────────────────

def save_to_dashboard(
    run_name: str,
    modelo,
    predicciones: pd.DataFrame,
    grid_results: pd.DataFrame,
    walk_forward: pd.DataFrame,
    historico: pd.Series,
    metricas: dict,
    acf_fig: matplotlib.figure.Figure,
    pacf_fig: matplotlib.figure.Figure,
    exog_data: "pd.Series | pd.DataFrame | None" = None,
) -> None:
    """Sube todos los artefactos del run a Supabase Storage.

    Parameters
    ----------
    exog_data : pd.Series | pd.DataFrame | None
        Variable exógena (ventas_otros) usada en SARIMAX. Se guarda como
        ``historico_exog.xlsx`` para que Comparativa ML pueda reproducir el
        mismo modelo que entrenamiento con la misma información de entrada.
        Si es None (exog descartada por baja correlación), el artefacto no
        se crea y Comparativa degradará automáticamente a SARIMA puro.
    """
    p = f"{run_name}/"
    log.info("Guardando artefactos del run '%s' en Supabase", run_name)
    failed: list[str] = []

    # Modelo (pickle comprimido)
    try:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            pickle.dump(modelo, gz)
        _upload(p + "modelo_total_mejorado.pkl.gz", buf.getvalue())
    except Exception as e:
        log.error("Run '%s': fallo al guardar modelo: %s", run_name, e)
        failed.append("modelo_total_mejorado.pkl.gz")

    # Archivos Excel
    excel_ct = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    for df, name, with_index in [
        (predicciones,  "prediccion_total_mejorada.xlsx",  False),
        (grid_results,  "grid_search_results.xlsx",        False),
        (walk_forward,  "walk_forward_validation.xlsx",    False),
        (historico.to_frame() if hasattr(historico, "to_frame") else historico,
                        "historico_total_mejorado.xlsx",   True),
    ]:
        try:
            buf = io.BytesIO()
            df.to_excel(buf, index=with_index, engine="openpyxl")
            _upload(p + name, buf.getvalue(), excel_ct)
        except Exception as e:
            log.error("Run '%s': fallo al guardar '%s': %s", run_name, name, e)
            failed.append(name)

    # Variable exógena (ventas_otros) si estuvo disponible en el entrenamiento
    if exog_data is not None:
        try:
            exog_df = exog_data.to_frame() if isinstance(exog_data, pd.Series) else exog_data
            buf = io.BytesIO()
            exog_df.to_excel(buf, index=True, engine="openpyxl")
            _upload(p + "historico_exog.xlsx", buf.getvalue(), excel_ct)
            log.debug("Run '%s': exog guardada (%d meses)", run_name, len(exog_df))
        except Exception as e:
            log.error("Run '%s': fallo al guardar exog: %s", run_name, e)
            failed.append("historico_exog.xlsx")

    # Métricas JSON
    try:
        _upload(
            p + "metricas_mejoradas.json",
            json.dumps(metricas, indent=2, ensure_ascii=False).encode(),
            "application/json"
        )
    except Exception as e:
        log.error("Run '%s': fallo al guardar métricas: %s", run_name, e)
        failed.append("metricas_mejoradas.json")

    # Imágenes ACF/PACF
    for fig, name in [(acf_fig, "acf_plot.png"), (pacf_fig, "pacf_plot.png")]:
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            _upload(p + name, buf.getvalue(), "image/png")
        except Exception as e:
            log.error("Run '%s': fallo al guardar '%s': %s", run_name, name, e)
            failed.append(name)

    if failed:
        log.error("Run '%s': artefactos con error: %s", run_name, failed)
        st.warning(f"Algunos artefactos no se guardaron correctamente: {', '.join(failed)}")

    n_esperados = 9 if exog_data is not None else 8
    n_guardados = n_esperados - len(failed)
    st.cache_data.clear()
    log.info("Run '%s' guardado (%d/%d artefactos)", run_name, n_guardados, n_esperados)


# ── Cargar datos del dashboard ───────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_precargados(
    run_name: str,
) -> "tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series | None]":
    """Descarga y parsea todos los artefactos de un run en paralelo (cacheado 10 min).

    Returns
    -------
    metricas, pred_total, grid_search, walk_forward, hist_total, exog_total
        ``exog_total`` es la serie ``ventas_otros`` guardada en el momento del
        entrenamiento, o ``None`` si el run es anterior a esta versión o si la
        variable exógena fue descartada por baja correlación.
    """
    p = f"{run_name}/"

    # Descargar todos los archivos en paralelo
    file_paths = {
        "metricas": p + "metricas_mejoradas.json",
        "pred":     p + "prediccion_total_mejorada.xlsx",
        "grid":     p + "grid_search_results.xlsx",
        "walk":     p + "walk_forward_validation.xlsx",
        "hist":     p + "historico_total_mejorado.xlsx",
        "exog":     p + "historico_exog.xlsx",
    }
    raw: dict[str, bytes | None] = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_download, path): key for key, path in file_paths.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                raw[key] = future.result()
            except Exception as e:
                raw[key] = None
                if key != "exog":
                    log.warning("Run '%s': no se pudo descargar '%s': %s", run_name, key, e)

    metricas: dict = json.loads(raw["metricas"])

    pred_total = pd.read_excel(io.BytesIO(raw["pred"]), engine="openpyxl")
    pred_total["Fecha"] = pd.to_datetime(pred_total["Fecha"])

    grid_search = pd.read_excel(io.BytesIO(raw["grid"]), engine="openpyxl")

    walk_forward = pd.read_excel(io.BytesIO(raw["walk"]), engine="openpyxl")
    walk_forward["fecha"] = pd.to_datetime(walk_forward["fecha"])

    hist_total = pd.read_excel(
        io.BytesIO(raw["hist"]), engine="openpyxl", index_col=0
    )
    hist_total.index = pd.to_datetime(hist_total.index)
    hist_total = hist_total.squeeze()

    exog_total = None
    if raw.get("exog") is not None:
        try:
            exog_df = pd.read_excel(
                io.BytesIO(raw["exog"]), engine="openpyxl", index_col=0
            )
            exog_df.index = pd.to_datetime(exog_df.index)
            exog_total = exog_df.squeeze()
            log.debug("Run '%s': exog cargada (%d meses)", run_name, len(exog_total))
        except Exception as e:
            log.debug("Run '%s': exog no disponible — %s", run_name, e)

    return metricas, pred_total, grid_search, walk_forward, hist_total, exog_total


@st.cache_data(ttl=600, show_spinner=False)
def load_acf_pacf_images(run_name: str) -> tuple[bytes | None, bytes | None]:
    """Descarga imágenes ACF/PACF como bytes para st.image."""
    try:
        acf  = _download(f"{run_name}/acf_plot.png")
        pacf = _download(f"{run_name}/pacf_plot.png")
        return acf, pacf
    except Exception as e:
        log.debug("ACF/PACF no disponibles para run='%s': %s", run_name, e)
        return None, None


# ── Modelo actual (para comparación ML) ─────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_current_model() -> dict | None:
    """Carga métricas del modelo activo. Devuelve None si no existe."""
    run_name = None
    # 1. Buscar en DB (activo=TRUE)
    try:
        rows = _db().select("run_name").eq("activo", True).limit(1).execute().data
        if rows:
            run_name = rows[0]["run_name"]
    except Exception as e:
        log.debug("No se pudo consultar activo en DB: %s", e)
    # 2. Fallback: latest.txt en Storage
    if not run_name:
        try:
            run_name = _download("latest.txt").decode().strip()
        except Exception as e:
            log.debug("latest.txt no disponible: %s", e)
    if not run_name:
        return None
    try:
        return json.loads(_download(f"{run_name}/metricas_mejoradas.json"))
    except Exception as e:
        log.debug("No se pudo cargar métricas del modelo activo '%s': %s", run_name, e)
        return None


# ── Historial de entrenamientos ──────────────────────────────────────────────

def save_training_log(entry: dict) -> None:
    """Persiste una entrada de entrenamiento en DB (primario) y training_log.json (backup)."""
    # 1. Upsert en PostgreSQL
    try:
        row = {
            "run_name":                   entry.get("run_name"),
            "timestamp":                  entry.get("timestamp"),
            "usuario":                    entry.get("usuario"),
            "modelo":                     entry.get("modelo"),
            "marca":                      entry.get("marca"),
            "fecha_inicio":               entry.get("fecha_inicio"),
            "horizonte":                  entry.get("horizonte"),
            "max_ventas":                 entry.get("max_ventas"),
            "sarima_order":               json.dumps(entry.get("sarima_order")),
            "sarima_seasonal":            json.dumps(entry.get("sarima_seasonal")),
            "aic":                        entry.get("aic"),
            "mape_wf":                    entry.get("mape_wf"),
            "meses_datos":                entry.get("meses_datos"),
            "combinaciones_validas":      entry.get("combinaciones_validas"),
            "combinaciones_descartadas":  entry.get("combinaciones_descartadas"),
        }
        _db().upsert(row, on_conflict="run_name").execute()
        log.info("Training log guardado en DB: run='%s'", entry.get("run_name"))
    except Exception as e:
        log.error("No se pudo guardar en DB: %s", e)
        st.warning(f"No se pudo guardar en base de datos: {e}")

    # 2. Backup en training_log.json (best-effort, fuente secundaria)
    # NOTA: este patrón read-modify-write puede perder entradas si dos entrenamientos
    # ocurren simultáneamente. La DB (paso 1) es la fuente de verdad.
    try:
        try:
            existing = json.loads(_download("training_log.json"))
        except Exception:
            existing = []
        existing.append(entry)
        _upload(
            "training_log.json",
            json.dumps(existing, indent=2, ensure_ascii=False).encode(),
            "application/json"
        )
    except Exception as e:
        log.warning("No se pudo actualizar training_log.json: %s", e)


def load_training_log() -> list[dict]:
    """Carga el historial completo. Fuente primaria: DB. Fallback: training_log.json."""
    try:
        rows = _db().select("*").order("created_at", desc=True).execute().data
        if rows:
            return rows
    except Exception as e:
        log.warning("DB no disponible en load_training_log: %s", e)
    try:
        return json.loads(_download("training_log.json"))
    except Exception as e:
        log.debug("training_log.json no disponible: %s", e)
        return []


# ── Caché LLM persistente por run ───────────────────────────────────────────

def save_llm_cache(run_name: str, cache: dict) -> None:
    """Persiste el caché de respuestas Gemini de un run en Supabase Storage."""
    try:
        _upload(
            f"{run_name}/llm_cache.json",
            json.dumps(cache, indent=2, ensure_ascii=False).encode(),
            "application/json"
        )
    except Exception as e:
        log.warning("No se pudo guardar llm_cache para run='%s': %s", run_name, e)


def load_llm_cache(run_name: str) -> dict:
    """Descarga el caché de respuestas Gemini de un run. Devuelve {} si no existe."""
    try:
        return json.loads(_download(f"{run_name}/llm_cache.json"))
    except Exception as e:
        log.debug("llm_cache no disponible para run='%s': %s", run_name, e)
        return {}


# ── Audit Log ────────────────────────────────────────────────────────────────

def log_audit(usuario: str | None, accion: str, run_name: str | None = None, detalle: dict | None = None) -> None:
    """Registra una acción de usuario en la tabla audit_log.
    Falla silenciosamente si la tabla no existe o hay error de red."""
    try:
        _audit().insert({
            "usuario":  usuario,
            "accion":   accion,
            "run_name": run_name,
            "detalle":  detalle or {},
        }).execute()
        log.debug("Audit log: usuario='%s' accion='%s' run='%s'", usuario, accion, run_name)
    except Exception as e:
        log.warning("No se pudo escribir audit_log: %s", e)


def get_audit_log(limit: int = 100) -> list[dict]:
    """Devuelve las últimas `limit` entradas del audit_log ordenadas por timestamp desc."""
    try:
        return (
            _audit()
            .select("*")
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
            .data
        )
    except Exception as e:
        log.warning("No se pudo leer audit_log: %s", e)
        return []


# ── Frescura del modelo ──────────────────────────────────────────────────────

def get_model_age_days(run_name: str) -> int | None:
    """Días transcurridos desde que se entrenó el modelo.
    Parsea el formato de run_name: YYYYMMDD_HHMMSS."""
    try:
        dt = datetime.strptime(run_name, "%Y%m%d_%H%M%S")
        return (datetime.now() - dt).days
    except ValueError:
        return None


# ── Ventas reales (feedback loop) ────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def get_ventas_reales() -> list[dict]:
    """Registros de ventas reales desde la tabla DB (cacheado 2 min)."""
    try:
        return (
            _ventas()
            .select("fecha, ventas, usuario, timestamp")
            .order("fecha", desc=False)
            .execute()
            .data
        )
    except Exception as e:
        log.warning("No se pudo leer ventas_reales de DB: %s", e)
        return []


def save_venta_real(fecha: str, ventas: int, usuario: str) -> None:
    """Upsert de un registro mensual en la tabla ventas_reales.
    Si ya existe una entrada para esa fecha, la sobreescribe (upsert on fecha)."""
    _ventas().upsert(
        {"fecha": fecha, "ventas": ventas, "usuario": usuario},
        on_conflict="fecha",
    ).execute()
    log_audit(usuario, "REGISTRAR_VENTA", detalle={"fecha": fecha, "ventas": ventas})
    log.info("Venta real guardada: fecha='%s' ventas=%d", fecha, ventas)
    st.cache_data.clear()


def delete_venta_real(fecha: str, usuario: str) -> None:
    """Elimina el registro de venta real de una fecha concreta."""
    _ventas().delete().eq("fecha", fecha).execute()
    log_audit(usuario, "ELIMINAR_VENTA", detalle={"fecha": fecha})
    log.info("Venta real eliminada: fecha='%s'", fecha)
    st.cache_data.clear()


# ── Comparativa ML — persistencia por run ────────────────────────────────────

def save_cml_resultados(run_name: str, metricas_df: "pd.DataFrame", ganador: str) -> None:
    """Persiste los resultados de Comparativa ML en Storage como JSON por run.
    Formato: {run_name}/cml_resultados.json — se sobreescribe en cada ejecución."""
    try:
        payload = {
            "ganador":   ganador,
            "metricas":  metricas_df.to_dict(orient="index"),
            "timestamp": datetime.now().isoformat(),
        }
        _upload(
            f"{run_name}/cml_resultados.json",
            json.dumps(payload, indent=2, ensure_ascii=False).encode(),
            "application/json",
        )
        log.info("CML resultados guardados: run='%s' ganador='%s'", run_name, ganador)
    except Exception as e:
        log.warning("No se pudo guardar cml_resultados para run='%s': %s", run_name, e)


@st.cache_data(ttl=600, show_spinner=False)
def load_cml_resultados(run_name: str) -> "dict | None":
    """Carga los resultados de Comparativa ML guardados para un run.
    Devuelve dict con 'metricas' (DataFrame) y 'ganador' (str), o None si no existen."""
    try:
        raw = json.loads(_download(f"{run_name}/cml_resultados.json"))
        return {
            "metricas": pd.DataFrame(raw["metricas"]).T,
            "ganador":  raw["ganador"],
        }
    except Exception as e:
        log.debug("cml_resultados no disponibles para run='%s': %s", run_name, e)
        return None


# ── Exportación Excel ────────────────────────────────────────────────────────

def build_export_excel(
    pred_total: pd.DataFrame,
    walk_forward: pd.DataFrame,
    metricas: dict,
    hist_total: pd.Series,
) -> bytes:
    """Genera un workbook Excel multi-hoja para descarga.
    Hojas: Predicciones · Walk-Forward · Histórico · Métricas."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pred_total.to_excel(writer, sheet_name="Predicciones", index=False)
        walk_forward.to_excel(writer, sheet_name="Walk-Forward", index=False)
        hist_df = hist_total.reset_index()
        hist_df.columns = ["Fecha", "Ventas"]
        hist_df.to_excel(writer, sheet_name="Histórico", index=False)
        orden    = metricas.get("mejor_modelo", {}).get("order", [])
        seasonal = metricas.get("mejor_modelo", {}).get("seasonal_order", [])
        pd.DataFrame({
            "Métrica": ["Modelo", "AIC", "BIC", "MAPE walk-forward (%)"],
            "Valor":   [
                f"SARIMA{orden}{seasonal}",
                metricas.get("mejor_modelo", {}).get("aic"),
                metricas.get("mejor_modelo", {}).get("bic"),
                metricas.get("walk_forward_validation", {}).get("mape"),
            ],
        }).to_excel(writer, sheet_name="Métricas", index=False)
    return buf.getvalue()


def build_proyeccion_excel(df_financiero: pd.DataFrame, cols: list[str]) -> bytes:
    """Excel para la proyección financiera (una hoja)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_financiero[cols].to_excel(writer, sheet_name="Proyección Ingresos", index=False)
    return buf.getvalue()
