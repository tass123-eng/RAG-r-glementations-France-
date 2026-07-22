"""Reranking par cross-encoder. Si le modèle n'est pas configuré ou ne peut
pas être chargé (pas de réseau, pas de GPU adapté...), on dégrade
gracieusement en conservant l'ordre de la fusion RRF plutôt que de planter."""
from __future__ import annotations

from src.config import get_settings
from src.logger import logger
from src.processing.hybrid_search import FusedResult


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().reranker_model
        self._model = None
        self._unavailable = not self.model_name

    def _load(self):
        if self._model is None and not self._unavailable:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Chargement du reranker: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Reranker indisponible ({e}), fallback sur l'ordre RRF.")
                self._unavailable = True
        return self._model

    def rerank(self, query: str, results: list[FusedResult], top_k: int = 10) -> list[FusedResult]:
        if not results:
            return results

        model = self._load()
        if model is None:
            return results[:top_k]

        pairs = [(query, r.text) for r in results]
        scores = model.predict(pairs)
        reranked = sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)
        return [r for r, _ in reranked[:top_k]]
