"""API FastAPI : ingestion, recherche brute et question/réponse RAG."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from ingest import ingest_pdf
from src.agents.agent_national import NationalAgent
from src.agents.agent_parcelle import ParcelleAgent
from src.agents.agent_plu import PLUAgent
from src.logger import logger
from src.processing.hybrid_search import HybridSearcher
from src.processing.rag import RAGPipeline
from src.processing.vector_store import VectorStore

app = FastAPI(
    title="PLU RAG AI",
    version="0.1.0",
    description="API de recherche RAG pour les documents PLU et réglementations nationales",
)

UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# Models
# ==========================

class AskRequest(BaseModel):
    question: str
    commune: str | None = None
    top_k: int = 10
    collection: str = "plu_documents"


class SearchRequest(BaseModel):
    query: str
    commune: str | None = None
    top_k: int = 10
    collection: str = "plu_documents"


class PLUAgentRequest(BaseModel):
    question: str
    commune: str | None = None
    zone: str | None = None
    top_k: int = 10
    collection: str = "plu_documents"


class NationalAgentRequest(BaseModel):
    question: str
    source_tag: str | None = None
    top_k: int = 10
    collection: str = "national_regulations"


class ParcelleRequest(BaseModel):
    address: str
    project_question: str | None = None
    plu_collection: str = "plu_documents"
    national_collection: str = "national_regulations"


# ==========================
# Routes
# ==========================

@app.get("/")
def root():
    return {
        "message": "PLU RAG AI API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ==========================
# Ingestion
# ==========================

@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
    source: str = "PLU",
    commune: str | None = None,
    collection: str = "plu_documents",
):
    if file.filename is None:
        raise HTTPException(400, "Nom du fichier manquant.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Seuls les fichiers PDF sont acceptés.")

    dest = UPLOAD_DIR / file.filename

    try:
        content = await file.read()
        dest.write_bytes(content)

        store = VectorStore(collection)

        n_chunks = ingest_pdf(
            pdf_path=dest,
            store=store,
            source=source,
            commune=commune,
        )

        HybridSearcher(store).build_bm25_index()

        return {
            "status": "success",
            "filename": file.filename,
            "collection": collection,
            "chunks_indexed": n_chunks,
        }

    except Exception as e:
        logger.exception("Erreur pendant l'ingestion")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================
# Hybrid Search
# ==========================

@app.post("/search")
def search_endpoint(req: SearchRequest):

    store = VectorStore(req.collection)
    searcher = HybridSearcher(store)

    where = {"commune": req.commune} if req.commune else None

    results = searcher.search(
        query=req.query,
        where=where,
        top_k_final=req.top_k,
    )

    return {
        "results": [
            {
                "id": r.id,
                "text": r.text,
                "metadata": r.metadata,
                "score": r.rrf_score,
            }
            for r in results
        ]
    }


# ==========================
# Generic RAG
# ==========================

@app.post("/ask")
def ask_endpoint(req: AskRequest):

    pipeline = RAGPipeline(collection_name=req.collection)

    result = pipeline.ask(
        question=req.question,
        address_or_commune=req.commune,
        top_k=req.top_k,
    )

    return {
        "answer": result.answer,
        "sources": [
            {
                "id": s.id,
                "text": s.text,
                "metadata": s.metadata,
                "score": s.rrf_score,
            }
            for s in result.sources
        ],
    }


# ==========================
# PLU Agent
# ==========================

@app.post("/agents/plu")
def plu_agent_endpoint(req: PLUAgentRequest):

    agent = PLUAgent(collection_name=req.collection)

    result = agent.answer(
        question=req.question,
        commune=req.commune,
        zone=req.zone,
        top_k=req.top_k,
    )

    return {
        "answer": result.answer,
        "sources": [
            {
                "id": s.id,
                "text": s.text,
                "metadata": s.metadata,
                "score": s.rrf_score,
            }
            for s in result.sources
        ],
    }


# ==========================
# National Agent
# ==========================

@app.post("/agents/national")
def national_agent_endpoint(req: NationalAgentRequest):

    agent = NationalAgent(collection_name=req.collection)

    result = agent.answer(
        question=req.question,
        source_tag=req.source_tag,
        top_k=req.top_k,
    )

    return {
        "answer": result.answer,
        "sources": [
            {
                "id": s.id,
                "text": s.text,
                "metadata": s.metadata,
                "score": s.rrf_score,
            }
            for s in result.sources
        ],
    }


# ==========================
# Parcelle Agent
# ==========================

@app.post("/agents/parcelle")
def parcelle_agent_endpoint(req: ParcelleRequest):

    agent = ParcelleAgent(
        plu_collection=req.plu_collection,
        national_collection=req.national_collection,
    )

    result = agent.qualify(
        address=req.address,
        project_question=req.project_question,
    )

    return {
        "address": result.address,
        "commune": result.commune,
        "synthesis": result.synthesis,
        "plu_answer": result.plu_answer,
        "national_answer": result.national_answer,
    }