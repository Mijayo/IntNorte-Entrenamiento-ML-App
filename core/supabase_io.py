"""
============================================================================
MÓDULO: I/O CON SUPABASE STORAGE
Reemplaza todas las operaciones de filesystem local.
Bucket: configurado en st.secrets["supabase"]["bucket"]
============================================================================
"""

import gzip
import io
import json
import pickle
from datetime import datetime

import matplotlib.figure
import pandas as pd
import streamlit as st
from supabase import create_client

from .logger import get_logger

log = get_logger("supabase_io")


# ── Cliente ─────────────────────────────────────────────────────────────────

@st.cache_resource
def get_client():
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"]
    )


def _bucket() -> str:
    return st.secrets["supabase"]["bucket"]


# ── Primitivas de I/O ────────────────────────────────────────────────────────

def _upload(path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """Sube bytes a Supabase Storage (sobreescribe si existe)."""
    sb = get_client()
    try:
        sb.storage.from_(_bucket()).remove([path])
    except Exception as e:
        # El fichero puede no existir aún (primera subida); no es un error fatal.
        log.debug("Pre-remove skipped for '%s': %s", path, e)
    sb.storage.from_(_bucket()).upload(
        path, data, {"content-type": content_type}
    )
    log.debug("Upload OK: %s (%d bytes)", path, len(data))


def _download(path: str) -> bytes:
    """Descarga bytes de Supabase Storage."""
    data = get_client().storage.from_(_bucket()).download(path)
    log.debug("Download OK: %s (%d bytes)", path, len(data))
    return data


# ── Gestión de runs ──────────────────────────────────────────────────────────

def _run_exists(run_name: str) -> bool:
    """Comprueba si un run tiene artefactos en Supabase Storage."""
    try:
        files = get_client().storage.from_(_bucket()).list(run_name)
        return len(files) > 0
    except Exception:
        return False


def get_available_runs() -> list[str]:
    """Lista de runs disponibles (más reciente primero) desde training_log.
    Filtra runs cuyos artefactos ya no existen en Supabase Storage."""
    try:
        log_data = json.loads(_download("training_log.json"))
        seen: dict[str, bool] = {}
        for entry in reversed(log_data):
            rn = entry.get("run_name")
            if rn and rn not in seen:
                seen[rn] = True
        # Solo devolver runs que realmente existen en el bucket
        return [rn for rn in seen.keys() if _run_exists(rn)]
    except Exception as e:
        log.debug("training_log.json no disponible, devolviendo lista vacía: %s", e)
        return []


def get_default_run(runs: list[str]) -> str | None:
    """Run activo según latest.txt, o el más reciente si no existe."""
    try:
        candidate = _download("latest.txt").decode().strip()
        if candidate in runs:
            return candidate
    except Exception as e:
        log.debug("latest.txt no disponible, usando run más reciente: %s", e)
    return runs[0] if runs else None


def approve_model(run_name: str) -> None:
    """Activa un run como modelo de producción (actualiza latest.txt)."""
    _upload("latest.txt", run_name.encode(), "text/plain")
    log.info("Modelo activado en producción: run='%s'", run_name)


def format_run_label(run_name: str) -> str:
    """Formatea 20260322_143000 → 22/03/2026  14:30."""
    try:
        dt = datetime.strptime(run_name, "%Y%m%d_%H%M%S")
        return dt.strftime("%d/%m/%Y  %H:%M")
    except ValueError:
        return run_name


# ── Guardar artefactos ───────────────────────────────────────────────────────

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
) -> None:
    """Sube todos los artefactos del run a Supabase Storage."""
    p = f"{run_name}/"
    log.info("Guardando artefactos del run '%s' en Supabase", run_name)

    # Modelo PKL (comprimido con gzip para reducir tamaño)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        pickle.dump(modelo, gz)
    _upload(p + "modelo_total_mejorado.pkl.gz", buf.getvalue())

    # Excel
    excel_ct = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    for df, name, with_index in [
        (predicciones,                                          "prediccion_total_mejorada.xlsx",  False),
        (grid_results,                                          "grid_search_results.xlsx",        False),
        (walk_forward,                                          "walk_forward_validation.xlsx",    False),
        (historico.to_frame() if hasattr(historico, "to_frame") else historico,
                                                                "historico_total_mejorado.xlsx",   True),
    ]:
        buf = io.BytesIO()
        df.to_excel(buf, index=with_index, engine="openpyxl")
        _upload(p + name, buf.getvalue(), excel_ct)

    # JSON métricas
    _upload(
        p + "metricas_mejoradas.json",
        json.dumps(metricas, indent=2, ensure_ascii=False).encode(),
        "application/json"
    )

    # PNG plots
    for fig, name in [(acf_fig, "acf_plot.png"), (pacf_fig, "pacf_plot.png")]:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        _upload(p + name, buf.getvalue(), "image/png")

    log.info("Run '%s' guardado correctamente (%d artefactos)", run_name, 8)


# ── Cargar datos del dashboard ───────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_precargados(run_name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Descarga y parsea todos los artefactos de un run (cacheado 10 min)."""
    p = f"{run_name}/"

    metricas: dict = json.loads(_download(p + "metricas_mejoradas.json"))

    pred_total = pd.read_excel(
        io.BytesIO(_download(p + "prediccion_total_mejorada.xlsx")), engine="openpyxl"
    )
    pred_total["Fecha"] = pd.to_datetime(pred_total["Fecha"])

    grid_search = pd.read_excel(
        io.BytesIO(_download(p + "grid_search_results.xlsx")), engine="openpyxl"
    )

    walk_forward = pd.read_excel(
        io.BytesIO(_download(p + "walk_forward_validation.xlsx")), engine="openpyxl"
    )
    walk_forward["fecha"] = pd.to_datetime(walk_forward["fecha"])

    hist_total = pd.read_excel(
        io.BytesIO(_download(p + "historico_total_mejorado.xlsx")),
        engine="openpyxl", index_col=0
    )
    hist_total.index = pd.to_datetime(hist_total.index)
    hist_total = hist_total.squeeze()

    return metricas, pred_total, grid_search, walk_forward, hist_total


def load_acf_pacf_images(run_name: str) -> tuple[bytes | None, bytes | None]:
    """Descarga imágenes ACF/PACF como bytes para st.image."""
    try:
        acf = _download(f"{run_name}/acf_plot.png")
        pacf = _download(f"{run_name}/pacf_plot.png")
        return acf, pacf
    except Exception as e:
        log.debug("ACF/PACF no disponibles para run='%s': %s", run_name, e)
        return None, None


# ── Modelo actual (para comparación) ────────────────────────────────────────

def load_current_model() -> dict | None:
    """Carga métricas del modelo activo (latest.txt). Devuelve None si no existe."""
    try:
        run_name = _download("latest.txt").decode().strip()
        return json.loads(_download(f"{run_name}/metricas_mejoradas.json"))
    except Exception as e:
        log.debug("No se pudo cargar el modelo actual (puede ser el primer run): %s", e)
        return None


# ── Historial de entrenamientos ──────────────────────────────────────────────

def save_training_log(entry: dict) -> None:
    """Añade una entrada al historial en Supabase."""
    try:
        try:
            existing = json.loads(_download("training_log.json"))
        except Exception as e:
            log.debug("training_log.json no existe aún, creando nuevo: %s", e)
            existing = []
        existing.append(entry)
        _upload(
            "training_log.json",
            json.dumps(existing, indent=2, ensure_ascii=False).encode(),
            "application/json"
        )
        log.info("Training log actualizado: run='%s'", entry.get("run_name"))
    except Exception as e:
        log.error("No se pudo guardar el historial: %s", e)
        st.warning(f"No se pudo guardar el historial: {e}")


def load_training_log() -> list[dict]:
    """Carga el historial completo de entrenamientos."""
    try:
        return json.loads(_download("training_log.json"))
    except Exception as e:
        log.debug("training_log.json no disponible: %s", e)
        return []


# ── Caché LLM persistente por run ───────────────────────────────────────────

def save_llm_cache(run_name: str, cache: dict) -> None:
    """Persiste el caché de respuestas Gemini de un run en Supabase."""
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
