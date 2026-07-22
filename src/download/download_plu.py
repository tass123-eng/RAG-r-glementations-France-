"""Téléchargement des documents PLU via l'API du Géoportail de l'Urbanisme (GPU).

L'API Carto GPU (apicarto.ign.fr/api/gpu) expose les zonages PLU et les
documents associés par commune (code INSEE). Ce module :
1. résout la commune -> code INSEE (via BANClient)
2. interroge le GPU pour retrouver le partition/document PLU de la commune
3. télécharge les fichiers PDF associés dans data/raw/
"""
from __future__ import annotations

from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.ban import BANClient
from src.config import get_settings
from src.logger import logger

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


class PLUDownloader:
    def __init__(self) -> None:
        self.gpu_url = get_settings().gpu_api_url
        self.ban = BANClient()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _get_document_partitions(self, insee_code: str) -> list[dict]:
        """Récupère les partitions de documents d'urbanisme pour une commune."""
        params = {"code_insee": insee_code}
        with httpx.Client(timeout=20) as client:
            resp = client.get(f"{self.gpu_url}/document", params=params)
            resp.raise_for_status()
            return resp.json().get("features", [])

    def download_for_commune(self, commune: str, output_dir: Path | None = None) -> list[Path]:
        """Télécharge les documents PLU disponibles pour une commune donnée."""
        output_dir = output_dir or RAW_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        address_result = self.ban.geocode(commune)
        if not address_result or not address_result.citycode:
            logger.error(f"Impossible de résoudre le code INSEE pour: {commune}")
            return []

        insee = address_result.citycode
        logger.info(f"Recherche des documents PLU pour {commune} (INSEE {insee})")

        try:
            partitions = self._get_document_partitions(insee)
        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur API GPU pour {commune}: {e}")
            return []

        if not partitions:
            logger.warning(f"Aucun document PLU trouvé pour {commune} sur le GPU")
            return []

        downloaded: list[Path] = []
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            for feat in partitions:
                props = feat.get("properties", {})
                file_url = props.get("fichier") or props.get("url")
                if not file_url:
                    continue
                filename = f"PLU_{insee}_{Path(file_url).name}"
                dest = output_dir / filename
                try:
                    resp = client.get(file_url)
                    resp.raise_for_status()
                    dest.write_bytes(resp.content)
                    downloaded.append(dest)
                    logger.info(f"Téléchargé: {dest.name}")
                except httpx.HTTPError as e:
                    logger.warning(f"Échec téléchargement {file_url}: {e}")

        return downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Télécharge les documents PLU d'une commune")
    parser.add_argument("commune", help="Nom de la commune, ex: Nantes")
    args = parser.parse_args()

    downloader = PLUDownloader()
    files = downloader.download_for_commune(args.commune)
    print(f"{len(files)} fichier(s) téléchargé(s)")
