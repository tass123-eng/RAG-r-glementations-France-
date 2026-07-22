"""Client VLM (Pixtral) — extrait des informations structurées des images
(plans de zonage, tableaux de règles) plutôt que de simples descriptions.

Le prompt demande explicitement les champs réglementaires clés (zonage,
hauteur, recul, implantation, emprise au sol, stationnement, espaces verts,
alignement, matériaux/façade/toiture) afin que chaque image devienne du
texte vectorisable et exploitable en recherche hybride.
"""
from __future__ import annotations

import json
from typing import Any

from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.logger import logger

EXTRACTION_PROMPT = """Tu analyses une page ou une image extraite d'un document \
d'urbanisme français (PLU ou règlement de construction). Décris précisément ce \
que tu vois et extrais, si présents, les champs suivants au format JSON strict \
(uniquement les champs que tu peux identifier avec certitude, ne rien inventer) :

{
  "type_contenu": "plan de zonage | tableau réglementaire | photo | schéma | texte | autre",
  "zone": null,
  "coefficient_emprise_sol": null,
  "hauteur_max_m": null,
  "recul_m": null,
  "implantation": null,
  "stationnement": null,
  "espaces_verts": null,
  "alignement": null,
  "materiaux_facade": null,
  "materiaux_toiture": null,
  "description_libre": ""
}

Réponds UNIQUEMENT avec ce JSON, sans texte avant ou après."""


class VLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = Mistral(api_key=settings.mistral_api_key) if settings.mistral_api_key else None
        self.model = settings.mistral_vlm_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
    def describe_image(self, image_b64: str, mime_type: str = "image/png") -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("MISTRAL_API_KEY manquante.")

        response = self._client.chat.complete(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {"type": "image_url", "image_url": f"data:{mime_type};base64,{image_b64}"},
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Réponse VLM non JSON, retour brut sous 'description_libre'")
            return {"type_contenu": "autre", "description_libre": raw}

    @staticmethod
    def to_text(structured: dict[str, Any]) -> str:
        """Convertit l'extraction structurée en texte plat vectorisable."""
        parts = []
        for key, value in structured.items():
            if value not in (None, "", []):
                label = key.replace("_", " ")
                parts.append(f"{label}: {value}")
        return " | ".join(parts)
