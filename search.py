"""CLI de recherche / RAG.

Usage :
    python search.py "Quelle est la hauteur maximale en zone UA ?" --commune Nantes
"""
from __future__ import annotations

import argparse

from src.processing.rag import RAGPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interroge la base PLU/réglementaire")
    parser.add_argument("question", help="Question en langage naturel")
    parser.add_argument("--commune", default=None, help="Commune ou adresse pour filtrer la recherche")
    parser.add_argument("--collection", default="plu_documents")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    pipeline = RAGPipeline(collection_name=args.collection)
    result = pipeline.ask(args.question, address_or_commune=args.commune, top_k=args.top_k)

    print("\n=== Réponse ===\n")
    print(result.answer)
    print("\n=== Sources ===\n")
    for src in result.sources:
        meta = src.metadata
        print(f"- [{meta.get('commune', '')} | zone {meta.get('zone', 'N/A')} | "
              f"p.{meta.get('page', '?')} | {meta.get('filename', '')}] score={src.rrf_score:.4f}")
