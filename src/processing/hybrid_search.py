"""Recherche hybride : fusion BM25 (lexical) + recherche vectorielle (sémantique)
via Reciprocal Rank Fusion (RRF), plus robuste qu'une simple moyenne pondérée
car insensible aux échelles de score différentes entre BM25 et cosine."""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.config import get_yaml_config
from src.logger import logger
from src.processing.vector_store import SearchResult, VectorStore


@dataclass
class FusedResult:
    id: str
    text: str
    metadata: dict
    rrf_score: float


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridSearcher:
    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self._bm25: BM25Okapi | None = None
        self._corpus: list[SearchResult] = []
        self._bm25_where: dict | None = None

    def build_bm25_index(self, where: dict | None = None) -> None:
        """Construit (ou reconstruit) l'index BM25 en mémoire à partir de
        tous les documents de la collection. À rappeler après une ingestion."""
        self._corpus = self.store.get_all_documents(where=where)
        self._bm25_where = where.copy() if where else None
        if not self._corpus:
            logger.warning("Aucun document disponible pour construire l'index BM25.")
            self._bm25 = None
            return
        tokenized = [_tokenize(r.text) for r in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"Index BM25 construit sur {len(self._corpus)} document(s)")

    def _bm25_search(self, query: str, top_k: int, where: dict | None = None) -> list[SearchResult]:
        if self._bm25 is None or self._bm25_where != where:
            self.build_bm25_index(where=where)
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(id=self._corpus[i].id, text=self._corpus[i].text,
                         metadata=self._corpus[i].metadata, score=float(scores[i]))
            for i in ranked_indices
        ]

    def search(self, query: str, where: dict | None = None,
               top_k_vector: int | None = None, top_k_bm25: int | None = None,
               top_k_final: int | None = None, rrf_k: int | None = None) -> list[FusedResult]:
        cfg = get_yaml_config().get("search", {})
        top_k_vector = top_k_vector or cfg.get("top_k_vector", 30)
        top_k_bm25 = top_k_bm25 or cfg.get("top_k_bm25", 30)
        top_k_final = top_k_final or cfg.get("top_k_final", 10)
        rrf_k = rrf_k or cfg.get("rrf_k", 60)

        vector_results = self.store.query(query, top_k=top_k_vector, where=where)
        bm25_results = self._bm25_search(query, top_k=top_k_bm25, where=where)

        # Reciprocal Rank Fusion : score = somme(1 / (k + rang)) sur chaque méthode
        rrf_scores: dict[str, float] = {}
        payload: dict[str, SearchResult] = {}

        for rank, res in enumerate(vector_results):
            rrf_scores[res.id] = rrf_scores.get(res.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            payload[res.id] = res

        for rank, res in enumerate(bm25_results):
            rrf_scores[res.id] = rrf_scores.get(res.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            payload.setdefault(res.id, res)

        fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k_final]
        return [
            FusedResult(id=doc_id, text=payload[doc_id].text, metadata=payload[doc_id].metadata, rrf_score=score)
            for doc_id, score in fused
        ]
