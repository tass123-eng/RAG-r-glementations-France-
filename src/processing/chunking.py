"""Découpage de texte en chunks avec chevauchement, en respectant autant
que possible les frontières de phrases/paragraphes."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import get_yaml_config


@dataclass
class Chunk:
    text: str
    index: int


def _split_sentences(text: str) -> list[str]:
    # Découpage simple par phrase, suffisant pour du texte réglementaire français
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ÿ])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """Regroupe des phrases jusqu'à ~chunk_size caractères, avec un
    chevauchement d'~overlap caractères entre chunks consécutifs."""
    cfg = get_yaml_config().get("chunking", {})
    chunk_size = chunk_size or cfg.get("chunk_size", 800)
    overlap = overlap or cfg.get("chunk_overlap", 150)

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunk_str = " ".join(current)
            chunks.append(Chunk(text=chunk_str, index=len(chunks)))

            # Construit le chevauchement à partir de la fin du chunk précédent
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len >= overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(Chunk(text=" ".join(current), index=len(chunks)))

    return chunks
