"""Orchestrateur RAG complet :

Question -> API BAN (adresse -> commune) -> filtre Chroma par commune
         -> recherche hybride (BM25 + vecteur) -> reranking -> Mistral Large -> réponse + sources
"""
from __future__ import annotations

from dataclasses import dataclass

from mistralai import Mistral

from src.ban import BANClient
from src.config import get_settings
from src.logger import logger
from src.processing.hybrid_search import FusedResult, HybridSearcher
from src.processing.reranker import Reranker
from src.processing.vector_store import VectorStore

ANSWER_PROMPT = """Tu es un assistant expert en urbanisme et réglementation de la \
construction en France. Réponds à la question de l'utilisateur UNIQUEMENT à partir \
des extraits fournis ci-dessous. Si l'information n'est pas présente dans les \
extraits, dis-le clairement plutôt que d'inventer. Cite la zone, la commune et la \
page quand c'est pertinent.

Extraits :
{context}

Question : {question}

Réponse :"""


@dataclass
class RAGAnswer:
    answer: str
    sources: list[FusedResult]


class RAGPipeline:
    def __init__(self, collection_name: str = "plu_documents") -> None:
        settings = get_settings()
        self.store = VectorStore(collection_name)
        self.searcher = HybridSearcher(self.store)
        self.reranker = Reranker()
        self.ban = BANClient()
        self._llm = Mistral(api_key=settings.mistral_api_key) if settings.mistral_api_key else None
        self._llm_model = settings.mistral_llm_model

    def _resolve_commune_filter(self, address_or_commune: str | None) -> dict | None:
        if not address_or_commune:
            return None
        commune = self.ban.commune_from_address(address_or_commune) or address_or_commune
        return {"commune": commune}

    def ask(self, question: str, address_or_commune: str | None = None, top_k: int = 10) -> RAGAnswer:
        where = self._resolve_commune_filter(address_or_commune)

        fused = self.searcher.search(question, where=where, top_k_final=top_k * 3)
        top_results = self.reranker.rerank(question, fused, top_k=top_k)

        if not top_results:
            return RAGAnswer(answer="Aucune information pertinente trouvée dans la base indexée.", sources=[])

        context = "\n\n".join(
            f"[{r.metadata.get('commune', '')} | zone {r.metadata.get('zone', 'N/A')} | "
            f"p.{r.metadata.get('page', '?')}] {r.text}"
            for r in top_results
        )

        if self._llm is None:
            logger.warning("MISTRAL_API_KEY manquante — retour des extraits sans synthèse LLM.")
            return RAGAnswer(answer=context, sources=top_results)

        response = self._llm.chat.complete(
            model=self._llm_model,
            messages=[{"role": "user", "content": ANSWER_PROMPT.format(context=context, question=question)}],
        )
        answer = response.choices[0].message.content
        return RAGAnswer(answer=answer, sources=top_results)
