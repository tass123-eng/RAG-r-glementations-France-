"""Wrapper ChromaDB : upsert de chunks avec métadonnées, requêtes filtrées."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings
from src.logger import logger
from src.processing.embeddings import get_embedding_model


@dataclass
class SearchResult:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # similarité (plus haut = mieux)


class VectorStore:
    def __init__(self, collection_name: str) -> None:
        persist_dir = get_settings().chroma_persist_dir
        self._client = chromadb.PersistentClient(path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._embedder = get_embedding_model()

    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        embeddings = self._embedder.embed(texts).tolist()
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logger.info(f"{len(ids)} chunk(s) indexé(s) dans '{self._collection.name}'")

    def query(self, query_text: str, top_k: int = 30, where: dict[str, Any] | None = None) -> list[SearchResult]:
        query_embedding = self._embedder.embed_one(query_text)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where or None,
        )

        results: list[SearchResult] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            # Chroma retourne une distance cosine ; on la convertit en score de similarité
            similarity = 1 - distances[i]
            results.append(SearchResult(id=doc_id, text=docs[i], metadata=metas[i], score=similarity))

        return results

    def get_all_documents(self, where: dict[str, Any] | None = None) -> list[SearchResult]:
        """Récupère tous les documents (pour construire l'index BM25)."""
        result = self._collection.get(where=where or None)
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        return [SearchResult(id=ids[i], text=docs[i], metadata=metas[i], score=0.0) for i in range(len(ids))]

    def count(self) -> int:
        return self._collection.count()
