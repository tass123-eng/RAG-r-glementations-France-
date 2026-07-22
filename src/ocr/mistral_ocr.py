"""Client pour l'API Mistral OCR.

Utilisé pour les pages sans texte natif (PDF scannés) — voir pdf_parser.py
pour la détection des pages concernées.
"""
from __future__ import annotations

import base64
from pathlib import Path

from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.logger import logger


class MistralOCRClient:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.mistral_api_key:
            logger.warning("MISTRAL_API_KEY non configurée — MistralOCRClient ne pourra pas appeler l'API.")
        self._client = Mistral(api_key=settings.mistral_api_key) if settings.mistral_api_key else None
        self.model = settings.mistral_ocr_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
    def ocr_pdf(self, pdf_path: Path) -> str:
        """Envoie un PDF entier à l'API Mistral OCR et retourne le texte
        markdown concaténé de toutes les pages."""
        if self._client is None:
            raise RuntimeError("MISTRAL_API_KEY manquante.")

        with open(pdf_path, "rb") as f:
            b64_pdf = base64.b64encode(f.read()).decode("utf-8")

        response = self._client.ocr.process(
            model=self.model,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64_pdf}",
            },
        )

        pages_markdown = [page.markdown for page in response.pages]
        logger.info(f"OCR terminé pour {pdf_path.name}: {len(pages_markdown)} page(s)")
        return "\n\n".join(pages_markdown)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
    def ocr_page_image(self, image_b64: str, mime_type: str = "image/png") -> str:
        """OCR d'une seule page rendue en image."""
        if self._client is None:
            raise RuntimeError("MISTRAL_API_KEY manquante.")

        response = self._client.ocr.process(
            model=self.model,
            document={
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{image_b64}",
            },
        )
        return "\n\n".join(page.markdown for page in response.pages)
