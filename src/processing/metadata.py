"""Extraction de métadonnées à partir du texte et du contexte du document."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ZONE_PATTERN = re.compile(r"\bzone\s+([UAN][A-Za-z0-9]{0,3})\b", re.IGNORECASE)

CHAPTER_KEYWORDS = {
    "hauteur": ["hauteur", "gabarit"],
    "implantation": ["implantation", "recul", "retrait"],
    "stationnement": ["stationnement", "parking"],
    "emprise au sol": ["emprise au sol", "coefficient d'emprise"],
    "espaces verts": ["espace vert", "espace libre", "plantation"],
    "aspect extérieur": ["aspect extérieur", "façade", "toiture", "matériaux"],
    "accès et voirie": ["accès", "voirie", "desserte"],
}


@dataclass
class DocumentMetadata:
    commune: str | None = None
    source: str = "PLU"
    page: int | None = None
    zone: str | None = None
    chapter: str | None = None
    type: str = "texte"  # texte | image | tableau
    filename: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "commune": self.commune or "",
            "source": self.source,
            "page": self.page or 0,
            "zone": self.zone or "",
            "chapter": self.chapter or "",
            "type": self.type,
            "filename": self.filename or "",
        }
        d.update(self.extra)
        return d


def detect_zone(text: str) -> str | None:
    match = ZONE_PATTERN.search(text)
    return match.group(1).upper() if match else None


def detect_chapter(text: str) -> str | None:
    lower = text.lower()
    for chapter, keywords in CHAPTER_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return chapter
    return None


def build_metadata(text: str, commune: str | None, source: str, page: int,
                    filename: str, content_type: str = "texte") -> DocumentMetadata:
    return DocumentMetadata(
        commune=commune,
        source=source,
        page=page,
        zone=detect_zone(text),
        chapter=detect_chapter(text),
        type=content_type,
        filename=filename,
    )
