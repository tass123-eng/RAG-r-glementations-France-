"""Client pour l'API Base Adresse Nationale (BAN).

Résout une adresse en coordonnées GPS + commune + code INSEE, utilisé pour
filtrer les recherches Chroma par commune.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.logger import logger


@dataclass
class AddressResult:
    label: str
    commune: str
    postcode: str
    citycode: str  # code INSEE
    lat: float
    lon: float


class BANClient:
    """Client synchrone pour https://api-adresse.data.gouv.fr"""

    def __init__(self) -> None:
        self.base_url = get_settings().ban_api_url

    def _search(self, query: str, limit: int) -> list[dict]:
        params = {"q": query, "limit": limit}
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{self.base_url}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        return data.get("features", [])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def geocode(self, address: str) -> AddressResult | None:
        features = self._search(address, limit=1)
        if not features:
            logger.warning(f"Aucune adresse trouvée pour: {address}")
            return None

        feat = features[0]
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]

        return AddressResult(
            label=props.get("label", address),
            commune=props.get("city", ""),
            postcode=props.get("postcode", ""),
            citycode=props.get("citycode", ""),
            lat=lat,
            lon=lon,
        )

    def commune_from_address(self, address: str) -> str | None:
        result = self.geocode(address)
        return result.commune if result else None

    def suggest_addresses(self, query: str, limit: int = 5) -> list[AddressResult]:
        """Retourne quelques suggestions BAN pour une saisie partielle."""
        suggestions: list[AddressResult] = []
        for feat in self._search(query, limit=limit):
            props = feat.get("properties", {})
            geometry = feat.get("geometry", {})
            coordinates = geometry.get("coordinates") or [None, None]
            lon, lat = coordinates
            if lat is None or lon is None:
                continue
            suggestions.append(
                AddressResult(
                    label=props.get("label", query),
                    commune=props.get("city", ""),
                    postcode=props.get("postcode", ""),
                    citycode=props.get("citycode", ""),
                    lat=lat,
                    lon=lon,
                )
            )
        return suggestions
