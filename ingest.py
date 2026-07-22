"""CLI d'ingestion complète.

Pipeline par PDF :
1. Extraction texte natif (rapide) ; pages sans texte -> Mistral OCR
2. Extraction des images (embarquées + rendu de page) -> Pixtral -> texte structuré
3. Fusion texte OCR + texte issu des images
4. Nettoyage -> chunking -> métadonnées -> embeddings -> upsert ChromaDB

Usage :
    python ingest.py --input data/raw --source PLU --commune Nantes
    python ingest.py --input data/raw --source national
"""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from src.logger import logger
from src.ocr.image_extractor import extract_embedded_images, render_page_as_image
from src.ocr.mistral_ocr import MistralOCRClient
from src.ocr.pdf_parser import extract_native_text, page_count
from src.ocr.vlm import VLMClient
from src.processing.chunking import chunk_text
from src.processing.clean_text import clean_text, remove_boilerplate
from src.processing.hybrid_search import HybridSearcher
from src.processing.metadata import build_metadata
from src.processing.vector_store import VectorStore
from src.utils import file_hash


def ingest_pdf(pdf_path: Path, store: VectorStore, source: str, commune: str | None,
                use_ocr_fallback: bool = True, use_vlm: bool = True) -> int:
    """Ingère un seul PDF dans le vector store. Retourne le nombre de chunks indexés."""
    logger.info(f"Ingestion: {pdf_path.name}")
    doc_hash = file_hash(pdf_path)

    ocr_client = MistralOCRClient() if use_ocr_fallback else None
    vlm_client = VLMClient() if use_vlm else None

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    # 1. Texte (natif, avec fallback OCR pour les pages scannées)
    pages = extract_native_text(pdf_path)
    for page in pages:
        page_text = page.text
        if not page.has_native_text and ocr_client is not None:
            try:
                image = render_page_as_image(pdf_path, page.page_number)
                page_text = ocr_client.ocr_page_image(image.b64)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"OCR échoué p.{page.page_number} de {pdf_path.name}: {e}")
                continue

        page_text = remove_boilerplate(clean_text(page_text))
        if not page_text:
            continue

        for chunk in chunk_text(page_text):
            meta = build_metadata(chunk.text, commune, source, page.page_number, pdf_path.name, "texte")
            meta.extra["doc_hash"] = doc_hash
            ids.append(f"{doc_hash}_p{page.page_number}_c{chunk.index}_{uuid.uuid4().hex[:6]}")
            texts.append(chunk.text)
            metadatas.append(meta.to_dict())

    # 2. Images -> Pixtral -> texte structuré vectorisable
    if vlm_client is not None:
        try:
            images = extract_embedded_images(pdf_path)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Extraction d'images échouée pour {pdf_path.name}: {e}")
            images = []

        for image in images:
            try:
                structured = vlm_client.describe_image(image.b64)
                image_text = VLMClient.to_text(structured)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"VLM échoué sur {image.path.name}: {e}")
                continue
            if not image_text:
                continue

            meta = build_metadata(image_text, commune, source, image.page_number, pdf_path.name, "image")
            meta.zone = meta.zone or structured.get("zone")
            meta.extra["doc_hash"] = doc_hash
            ids.append(f"{doc_hash}_p{image.page_number}_img_{uuid.uuid4().hex[:6]}")
            texts.append(image_text)
            metadatas.append(meta.to_dict())

    store.upsert(ids, texts, metadatas)
    logger.info(f"{pdf_path.name}: {len(ids)} chunk(s) indexé(s) ({page_count(pdf_path)} pages)")
    return len(ids)


def ingest_directory(input_dir: Path, source: str, commune: str | None,
                      collection_name: str = "plu_documents") -> None:
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"Aucun PDF trouvé dans {input_dir}")
        return

    store = VectorStore(collection_name)
    total = 0
    for pdf_path in pdfs:
        total += ingest_pdf(pdf_path, store, source, commune)

    logger.info(f"Ingestion terminée: {total} chunk(s) au total sur {len(pdfs)} PDF(s)")

    # Reconstruit l'index BM25 pour que la recherche hybride voie les nouveaux documents
    HybridSearcher(store).build_bm25_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion de documents PLU/réglementaires dans ChromaDB")
    parser.add_argument("--input", required=True, type=Path, help="Dossier contenant les PDF")
    parser.add_argument("--source", default="PLU", help="PLU ou national")
    parser.add_argument("--commune", default=None, help="Commune associée (pour les PLU)")
    parser.add_argument("--collection", default="plu_documents")
    parser.add_argument("--no-ocr", action="store_true", help="Désactive le fallback OCR")
    parser.add_argument("--no-vlm", action="store_true", help="Désactive l'extraction des images via VLM")
    args = parser.parse_args()

    store = VectorStore(args.collection)
    pdfs = sorted(args.input.glob("*.pdf"))
    total = 0
    for pdf_path in pdfs:
        total += ingest_pdf(pdf_path, store, args.source, args.commune,
                             use_ocr_fallback=not args.no_ocr, use_vlm=not args.no_vlm)
    logger.info(f"Ingestion terminée: {total} chunk(s) au total sur {len(pdfs)} PDF(s)")
    HybridSearcher(store).build_bm25_index()
