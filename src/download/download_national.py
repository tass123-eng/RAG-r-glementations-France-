"""Téléchargeur de textes réglementaires nationaux.

La liste des sources (Code de l'urbanisme, RE2020, ERP, accessibilité,
DTU, Eurocodes...) est définie dans `config.yaml` sous `national_sources`.

Ce module ne prétend pas connaître par cœur les URLs stables de chaque
texte officiel (elles changent, et certaines sources comme les DTU/
Eurocodes sont payantes/AFNOR). Il fournit donc :
- `register_source(name, tag, url)` pour brancher une source réelle une fois
  l'URL identifiée (Légifrance, service-public, etc.)
- `download_all()` qui télécharge tout ce qui a été enregistré

C'est un point d'extension volontairement explicite plutôt qu'une fausse
liste d'URLs qui casserait silencieusement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential

from src.logger import logger

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


@dataclass
class NationalSource:
    name: str
    tag: str
    url: str | None = None


@dataclass
class NationalDownloader:
    sources: list[NationalSource] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sources:
            self.sources = self._load_from_config()

    @staticmethod
    def _load_from_config() -> list[NationalSource]:
        if not CONFIG_PATH.exists():
            return []
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return [
            NationalSource(name=s["name"], tag=s["tag"], url=s.get("url"))
            for s in cfg.get("national_sources", [])
        ]

    def register_source(self, name: str, tag: str, url: str) -> None:
        self.sources.append(NationalSource(name=name, tag=tag, url=url))
        logger.info(f"Source enregistrée: {name} -> {url}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _download_one(self, source: NationalSource, output_dir: Path) -> Path | None:
        if not source.url:
            logger.warning(f"Pas d'URL configurée pour '{source.name}' — utiliser register_source().")
            return None
        dest = output_dir / f"{source.tag}.pdf"
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(source.url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        logger.info(f"Téléchargé: {source.name} -> {dest.name}")
        return dest

    def download_all(self, output_dir: Path | None = None) -> list[Path]:
        output_dir = output_dir or RAW_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[Path] = []
        for source in self.sources:
            try:
                path = self._download_one(source, output_dir)
                if path:
                    results.append(path)
            except httpx.HTTPError as e:
                logger.error(f"Échec téléchargement '{source.name}': {e}")
        return results


if __name__ == "__main__":
    downloader = NationalDownloader()
    files = downloader.download_all()
    print(f"{len(files)} fichier(s) téléchargé(s) sur {len(downloader.sources)} source(s) configurée(s)")
