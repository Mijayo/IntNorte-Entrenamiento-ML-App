"""
============================================================================
MÓDULO: LOGGING CENTRALIZADO
Provee un logger configurado que escribe simultáneamente a consola y a un
archivo rotativo en logs/app.log.  Todas las páginas importan get_logger()
en lugar de usar st.info/warning directamente para eventos de sistema.
============================================================================
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_LOG_FILE = os.path.join(_LOGS_DIR, "app.log")
_MAX_BYTES = 2 * 1024 * 1024   # 2 MB por fichero
_BACKUP_COUNT = 3               # conservar los 3 últimos


def _build_handler_file() -> RotatingFileHandler:
    os.makedirs(_LOGS_DIR, exist_ok=True)
    handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def _build_handler_console() -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)-8s  [%(name)s]  %(message)s")
    )
    return handler


def get_logger(name: str = "tiggo2") -> logging.Logger:
    """
    Devuelve un Logger configurado.  Las llamadas repetidas con el mismo
    ``name`` devuelven el mismo objeto (comportamiento estándar de logging).

    Uso::

        from logger import get_logger
        log = get_logger(__name__)

        log.info("Entrenamiento iniciado por %s", username)
        log.warning("MAPE %.1f%% supera el umbral del 20%%", mape)
        log.error("Fallo al subir artefactos: %s", exc)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.addHandler(_build_handler_file())
    logger.addHandler(_build_handler_console())
    logger.propagate = False

    return logger
