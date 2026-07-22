"""Parsing bas niveau des PDF avec PyMuPDF (fitz).

Fournit l'extraction de texte natif (rapide, pour les PDF non scannés) et
sert de base à l'extraction d'images (voir image_extractor.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from src.logger import logger


@dataclass
class PageContent:
    page_number: int
    text: str
    has_native_text: bool


def extract_native_text(pdf_path: Path) -> list[PageContent]:
    """Extrait le texte natif de chaque page. Retourne has_native_text=False
    pour les pages qui semblent scannées (peu/pas de texte détecté), afin de
    savoir lesquelles envoyer à l'OCR."""
    pages: list[PageContent] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            has_native = len(text) > 40  # heuristique simple
            pages.append(PageContent(page_number=i + 1, text=text, has_native_text=has_native))
    logger.debug(f"{pdf_path.name}: {len(pages)} pages, "
                 f"{sum(p.has_native_text for p in pages)} avec texte natif")
    return pages


def page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count
