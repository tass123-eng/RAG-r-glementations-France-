"""Embeddings multilingues BAAI/bge-m3 via sentence-transformers.

BGE-M3 est choisi pour sa robustesse en français et sur le vocabulaire
juridique/technique, comparé à des modèles anglais généralistes.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.config import get_settings
from src.logger import logger


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().embedding_model
        self._model = None  # chargement paresseux (le modèle est lourd)

    def _load(self):
        if self._model is None:
            logger.info(f"Chargement du modèle d'embedding: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0].tolist()


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
