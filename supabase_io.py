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

import pandas as pd
import streamlit as st
from supabase import create_client


# ── Cliente ─────────────────────────────────────────────────────────────────

@st.cache_resource
def get_client():
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"]
    )


def _bucket():
    return st.secrets["supabase"]["bucket"]


# ── Primitivas de I/O ────────────────────────────────────────────────────────

def _upload(path: str, data: bytes, content_type: str = "application/octet-stream"):
    """Sube bytes a Supabase Storage (sobreescribe si existe)"""
    sb = get_client()
    try:
        sb.storage.from_(_bucket()).remove([path])
    except Exception:
        pass
    sb.storage.from_(_bucket()).upload(
        path, data, {"content-type": content_type}
    )


def _download(path: str) -> bytes:
    """Descarga bytes de Supabase Storage"""
    return get_client().storage.from_(_bucket()).download(path)


# ── Gestión de runs ──────────────────────────────────────────────────────────

def get_available_runs() -> list:
    """Lista de runs disponibles (más reciente primero) desde training_log"""
    try:
        log = json.loads(_download("training_log.json"))
        seen = {}
        for entry in reversed(log):
            rn = entry.get("run_name")
            if rn and rn not in seen:
                seen[rn] = True
        return list(seen.keys())
    except Exception:
        return []


def get_default_run(runs: list) -> str | None:
    """Run activo según latest.txt, o el más reciente si no existe"""
    try:
        candidate = _download("latest.txt").decode().strip()
        if candidate in runs:
            return candidate
    except Exception:
        pass
    return runs[0] if runs else None


def approve_model(run_name: str):
    """Activa un run como modelo de producción (actualiza latest.txt)"""
    _upload("latest.txt", run_name.encode(), "text/plain")


def format_run_label(run_name: str) -> str:
    """Formatea 20260322_143000 → 22/03/2026  14:30"""
    try:
        dt = datetime.strptime(run_name, "%Y%m%d_%H%M%S")
        return dt.strftime("%d/%m/%Y  %H:%M")
    except ValueError:
        return run_name


# ── Guardar artefactos ───────────────────────────────────────────────────────

def save_to_dashboard(run_name, modelo, predicciones, grid_results,
                      walk_forward, historico, metricas, acf_fig, pacf_fig):
    """Sube todos los artefactos del run a Supabase Storage"""
    p = f"{run_name}/"

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


# ── Cargar datos del dashboard ───────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_precargados(run_name: str):
    """Descarga y parsea todos los artefactos de un run (cacheado 10 min)"""
    p = f"{run_name}/"

    metricas = json.loads(_download(p + "metricas_mejoradas.json"))

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


def load_acf_pacf_images(run_name: str):
    """Descarga imágenes ACF/PACF como bytes para st.image"""
    try:
        acf = _download(f"{run_name}/acf_plot.png")
        pacf = _download(f"{run_name}/pacf_plot.png")
        return acf, pacf
    except Exception:
        return None, None


# ── Modelo actual (para comparación) ────────────────────────────────────────

def load_current_model():
    """Carga métricas del modelo activo (latest.txt)"""
    try:
        run_name = _download("latest.txt").decode().strip()
        return json.loads(_download(f"{run_name}/metricas_mejoradas.json"))
    except Exception:
        return None


# ── Historial de entrenamientos ──────────────────────────────────────────────

def save_training_log(entry: dict):
    """Añade una entrada al historial en Supabase"""
    try:
        try:
            log = json.loads(_download("training_log.json"))
        except Exception:
            log = []
        log.append(entry)
        _upload(
            "training_log.json",
            json.dumps(log, indent=2, ensure_ascii=False).encode(),
            "application/json"
        )
    except Exception as e:
        st.warning(f"No se pudo guardar el historial: {e}")


def load_training_log() -> list:
    """Carga el historial completo de entrenamientos"""
    try:
        return json.loads(_download("training_log.json"))
    except Exception:
        return []
