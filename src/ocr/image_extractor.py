"""Extraction des images embarquées et rendu de pages en image (pour le VLM)."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import fitz

from src.logger import logger

IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "images"


@dataclass
class ExtractedImage:
    page_number: int
    path: Path
    b64: str


def render_page_as_image(pdf_path: Path, page_number: int, dpi: int = 200,
                          output_dir: Path | None = None) -> ExtractedImage:
    """Rend une page complète en PNG (utile pour les plans/cartes de zonage
    qui ne sont pas des 'images' embarquées mais du contenu vectoriel)."""
    output_dir = output_dir or IMAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=dpi)
        dest = output_dir / f"{pdf_path.stem}_p{page_number}.png"
        pix.save(dest)
        b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

    logger.debug(f"Page rendue: {dest.name}")
    return ExtractedImage(page_number=page_number, path=dest, b64=b64)


def extract_embedded_images(pdf_path: Path, output_dir: Path | None = None) -> list[ExtractedImage]:
    """Extrait les images embarquées (photos, logos, petits plans) page par page."""
    output_dir = output_dir or IMAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExtractedImage] = []

    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Échec extraction image xref={xref}: {e}")
                    continue
                ext = base_image.get("ext", "png")
                dest = output_dir / f"{pdf_path.stem}_p{page_index + 1}_img{img_index}.{ext}"
                dest.write_bytes(base_image["image"])
                b64 = base64.b64encode(base_image["image"]).decode("utf-8")
                results.append(ExtractedImage(page_number=page_index + 1, path=dest, b64=b64))

    logger.info(f"{pdf_path.name}: {len(results)} image(s) embarquée(s) extraite(s)")
    return results
