"""Nettoyage de texte issu de l'OCR/extraction PDF."""
from __future__ import annotations

import re


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Recolle les mots coupés par un tiret en fin de ligne (artefact OCR/PDF)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


def remove_boilerplate(text: str, patterns: list[str] | None = None) -> str:
    """Retire les en-têtes/pieds de page récurrents (numéros de page seuls,
    mentions type 'Page X/Y')."""
    default_patterns = [
        r"^\s*Page\s+\d+\s*/\s*\d+\s*$",
        r"^\s*\d+\s*$",
    ]
    patterns = patterns or default_patterns
    lines = text.split("\n")
    cleaned = [ln for ln in lines if not any(re.match(p, ln) for p in patterns)]
    return "\n".join(cleaned)
