"""Agent 'parcelle' : orchestrateur qui combine résolution d'adresse (API BAN),
règles PLU locales et réglementation nationale applicable, pour qualifier une
parcelle avant projet.

Ce n'est pas un cadastre officiel : c'est une synthèse basée sur ce qui a été
indexé (PLU de la commune + textes nationaux). L'agent le rappelle explicitement
quand des informations précises (numéro de parcelle, servitudes) manquent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mistralai import Mistral

from src.agents.agent_national import NationalAgent
from src.agents.agent_plu import PLUAgent
from src.agents.prompts import PARCELLE_SYSTEM_PROMPT
from src.ban import BANClient
from src.config import get_settings
from src.logger import logger
from src.processing.hybrid_search import FusedResult


@dataclass
class ParcelleQualification:
    address: str
    commune: str | None
    plu_answer: str
    national_answer: str
    synthesis: str
    plu_sources: list[FusedResult] = field(default_factory=list)
    national_sources: list[FusedResult] = field(default_factory=list)


class ParcelleAgent:
    def __init__(self, plu_collection: str = "plu_documents",
                 national_collection: str = "national_regulations") -> None:
        settings = get_settings()
        self.ban = BANClient()
        self.plu_agent = PLUAgent(collection_name=plu_collection)
        self.national_agent = NationalAgent(collection_name=national_collection)
        self._llm = Mistral(api_key=settings.mistral_api_key) if settings.mistral_api_key else None
        self._llm_model = settings.mistral_llm_model

    def qualify(self, address: str, project_question: str | None = None) -> ParcelleQualification:
        """Qualifie une parcelle à partir d'une adresse.

        `project_question` permet d'orienter la recherche (ex: "extension de
        maison individuelle de 40m2"). Si absent, une question générique sur
        les règles applicables est utilisée.
        """
        address_result = self.ban.geocode(address)
        commune = address_result.commune if address_result else None

        if commune is None:
            logger.warning(f"Adresse non résolue: {address}")

        question = project_question or "Quelles sont les règles de constructibilité applicables (hauteur, emprise au sol, recul, stationnement) ?"

        plu_result = self.plu_agent.answer(question, commune=commune)
        national_result = self.national_agent.answer(question)

        synthesis = self._synthesize(address, commune, plu_result.answer, national_result.answer)

        return ParcelleQualification(
            address=address,
            commune=commune,
            plu_answer=plu_result.answer,
            national_answer=national_result.answer,
            synthesis=synthesis,
            plu_sources=plu_result.sources,
            national_sources=national_result.sources,
        )

    def _synthesize(self, address: str, commune: str | None, plu_answer: str, national_answer: str) -> str:
        if self._llm is None:
            return f"[PLU - {commune or 'commune non résolue'}]\n{plu_answer}\n\n[National]\n{national_answer}"

        prompt = (
            f"Adresse: {address}\nCommune résolue: {commune or 'inconnue'}\n\n"
            f"Règles PLU locales identifiées:\n{plu_answer}\n\n"
            f"Règles nationales applicables:\n{national_answer}\n\n"
            "Rédige une synthèse claire et structurée pour un porteur de projet, "
            "en signalant les points qui nécessitent une vérification complémentaire "
            "(ex: consultation du service urbanisme de la commune, cadastre officiel)."
        )
        response = self._llm.chat.complete(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": PARCELLE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
