"""Logging structuré basé sur loguru."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")
logger.add(LOG_DIR / "plu_rag.log", level="DEBUG", rotation="10 MB", retention="10 days", encoding="utf-8")

__all__ = ["logger"]
