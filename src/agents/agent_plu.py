"""Agent spécialisé PLU : répond aux questions de règlement local d'urbanisme,
toujours filtré par commune quand elle est identifiable."""
from __future__ import annotations

from dataclasses import dataclass

from mistralai import Mistral

from src.agents.prompts import PLU_SYSTEM_PROMPT
from src.config import get_settings
from src.logger import logger
from src.processing.hybrid_search import FusedResult, HybridSearcher
from src.processing.reranker import Reranker
from src.processing.vector_store import VectorStore


@dataclass
class AgentAnswer:
    answer: str
    sources: list[FusedResult]


class PLUAgent:
    """Agent RAG contraint à la collection PLU, avec filtrage par commune/zone."""

    def __init__(self, collection_name: str = "plu_documents") -> None:
        settings = get_settings()
        self.store = VectorStore(collection_name)
        self.searcher = HybridSearcher(self.store)
        self.reranker = Reranker()
        self._llm = Mistral(api_key=settings.mistral_api_key) if settings.mistral_api_key else None
        self._llm_model = settings.mistral_llm_model

    def answer(self, question: str, commune: str | None = None, zone: str | None = None,
               top_k: int = 10) -> AgentAnswer:
        where: dict = {}
        if commune:
            where["commune"] = commune
        if zone:
            where["zone"] = zone

        fused = self.searcher.search(question, where=where or None, top_k_final=top_k * 3)
        top_results = self.reranker.rerank(question, fused, top_k=top_k)

        if not top_results:
            msg = "Aucune information PLU trouvée"
            msg += f" pour {commune}" if commune else ""
            msg += f" en zone {zone}" if zone else ""
            msg += " dans la base indexée."
            return AgentAnswer(answer=msg, sources=[])

        context = "\n\n".join(
            f"[{r.metadata.get('commune', '')} | zone {r.metadata.get('zone', 'N/A')} | "
            f"p.{r.metadata.get('page', '?')}] {r.text}"
            for r in top_results
        )

        if self._llm is None:
            logger.warning("MISTRAL_API_KEY manquante — retour des extraits sans synthèse LLM.")
            return AgentAnswer(answer=context, sources=top_results)

        response = self._llm.chat.complete(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": PLU_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extraits :\n{context}\n\nQuestion : {question}"},
            ],
        )
        return AgentAnswer(answer=response.choices[0].message.content, sources=top_results)
