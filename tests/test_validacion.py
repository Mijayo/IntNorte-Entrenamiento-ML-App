"""
============================================================================
TESTS UNITARIOS — utils_validacion
Ejecutar: pytest tests/ -v
============================================================================
"""

import sys
import os

# Permite importar módulos del proyecto sin instalarlo como paquete
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

# Las funciones de validación no dependen de Streamlit en su lógica pura;
# se importan directamente y se testean las rutas de validación.
from core.utils_validacion import (
    REQUIRED_COLUMNS,
    MIN_MONTHS_DATA,
    MAX_MISSING_PCT,
    get_dataset_summary,
    validate_dataframe,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n_months: int = 40, marca: str = "CHERY", modelo: str = "TIGGO 2") -> pd.DataFrame:
    """DataFrame mínimo válido: n_months meses consecutivos, 10 ventas/mes."""
    dates = pd.date_range("2021-01-15", periods=n_months, freq="ME")
    rows = []
    for d in dates:
        for _ in range(10):
            rows.append({"FECHA-VENTA": d, "MARCA": marca, "MODELO3": modelo})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: validate_dataframe — columnas
# ─────────────────────────────────────────────────────────────────────────────

class TestColumnas:
    def test_columnas_presentes(self):
        df = _make_df()
        valid, _, _, errors = validate_dataframe(df, "test")
        assert valid is True
        assert not any("Faltan columnas" in e for e in errors)

    def test_falta_modelo3(self):
        df = _make_df().drop(columns=["MODELO3"])
        valid, _, _, errors = validate_dataframe(df, "test")
        assert valid is False
        assert any("MODELO3" in e for e in errors)

    def test_falta_marca(self):
        df = _make_df().drop(columns=["MARCA"])
        valid, _, _, errors = validate_dataframe(df, "test")
        assert valid is False
        assert any("MARCA" in e for e in errors)

    def test_falta_fecha_venta(self):
        df = _make_df().drop(columns=["FECHA-VENTA"])
        valid, _, _, errors = validate_dataframe(df, "test")
        assert valid is False
        assert any("FECHA-VENTA" in e for e in errors)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: validate_dataframe — período temporal
# ─────────────────────────────────────────────────────────────────────────────

class TestPeriodoTemporal:
    def test_suficientes_meses(self):
        df = _make_df(n_months=MIN_MONTHS_DATA)
        valid, _, _, errors = validate_dataframe(df, "test")
        assert valid is True
        assert not any("insuficientes" in e.lower() for e in errors)

    def test_insuficientes_meses(self):
        df = _make_df(n_months=MIN_MONTHS_DATA - 1)
        valid, _, _, errors = validate_dataframe(df, "test")
        assert valid is False
        assert any("insuficientes" in e.lower() or "mínimo" in e.lower() for e in errors)

    def test_exactamente_36_meses_es_valido(self):
        df = _make_df(n_months=36)
        valid, _, _, errors = validate_dataframe(df, "test")
        assert valid is True


# ─────────────────────────────────────────────────────────────────────────────
# Tests: validate_dataframe — fechas inválidas
# ─────────────────────────────────────────────────────────────────────────────

class TestFechasInvalidas:
    def test_fechas_validas(self):
        df = _make_df()
        valid, _, warnings, errors = validate_dataframe(df, "test")
        assert not any("fechas inválidas" in e.lower() for e in errors)

    def test_pocas_fechas_invalidas_solo_warning(self):
        df = _make_df(n_months=40)
        # Introducir 1 fecha inválida: < MAX_MISSING_PCT → debe ser warning, no error
        df.loc[0, "FECHA-VENTA"] = "NOT_A_DATE"
        _, _, warnings, errors = validate_dataframe(df, "test")
        assert not any("Demasiadas fechas" in e for e in errors)

    def test_muchas_fechas_invalidas_es_error(self):
        df = _make_df(n_months=40)
        # Corromper el 50% de las fechas → supera MAX_MISSING_PCT
        n_corrupt = len(df) // 2
        df.loc[:n_corrupt, "FECHA-VENTA"] = "INVALID"
        valid, _, _, errors = validate_dataframe(df, "test")
        assert valid is False
        assert any("fechas inválidas" in e.lower() or "demasiadas" in e.lower() for e in errors)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: validate_dataframe — datos faltantes
# ─────────────────────────────────────────────────────────────────────────────

class TestDatosFaltantes:
    def test_sin_faltantes_sin_warning(self):
        df = _make_df()
        _, _, warnings, _ = validate_dataframe(df, "test")
        assert not any(">5% faltantes" in w for w in warnings)

    def test_muchos_nulos_genera_warning(self):
        df = _make_df()
        # Poner nulos en >5% de MODELO3
        n_nulos = int(len(df) * 0.10)
        df.loc[:n_nulos, "MODELO3"] = np.nan
        _, _, warnings, _ = validate_dataframe(df, "test")
        assert any("faltantes" in w.lower() for w in warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: get_dataset_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDatasetSummary:
    def test_claves_presentes(self):
        df = _make_df()
        summary = get_dataset_summary(df)
        assert "total_registros" in summary
        assert "meses_unicos" in summary
        assert "modelos_unicos" in summary
        assert "marcas_unicas" in summary

    def test_conteo_correcto(self):
        df = _make_df(n_months=40)
        summary = get_dataset_summary(df)
        assert summary["total_registros"] == 400   # 40 meses × 10 ventas
        assert summary["meses_unicos"] == 40
        assert summary["modelos_unicos"] == 1
        assert summary["marcas_unicas"] == 1

    def test_df_vacio_no_explota(self):
        df = pd.DataFrame(columns=["FECHA-VENTA", "MARCA", "MODELO3"])
        summary = get_dataset_summary(df)
        assert summary["total_registros"] == 0

    def test_memoria_mb_positiva(self):
        df = _make_df()
        summary = get_dataset_summary(df)
        assert summary["memoria_mb"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: opciones de frontera
# ─────────────────────────────────────────────────────────────────────────────

class TestCasosFrontera:
    def test_df_una_fila(self):
        df = pd.DataFrame([{
            "FECHA-VENTA": "2024-01-01",
            "MARCA": "CHERY",
            "MODELO3": "TIGGO 2"
        }])
        valid, _, _, errors = validate_dataframe(df, "test")
        assert valid is False   # 1 mes < MIN_MONTHS_DATA

    def test_required_columns_constante(self):
        assert "FECHA-VENTA" in REQUIRED_COLUMNS
        assert "MODELO3" in REQUIRED_COLUMNS
        assert "MARCA" in REQUIRED_COLUMNS

    def test_thresholds_valores_razonables(self):
        assert MIN_MONTHS_DATA >= 24
        assert 1 <= MAX_MISSING_PCT <= 20
