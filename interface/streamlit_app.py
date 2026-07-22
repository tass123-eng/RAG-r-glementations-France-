"""Interface de test manuelle (Streamlit).

Lancement :
    streamlit run interface/streamlit_app.py

Permet d'interroger rapidement la base sans passer par curl/Postman :
analyse par localisation, recherche brute et question/réponse RAG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.agent_national import NationalAgent  # noqa: E402
from src.agents.agent_parcelle import ParcelleAgent  # noqa: E402
from src.agents.agent_plu import PLUAgent  # noqa: E402
from src.ban import BANClient  # noqa: E402
from src.download.download_plu import PLUDownloader  # noqa: E402
from ingest import ingest_pdf  # noqa: E402
from src.processing.vector_store import VectorStore  # noqa: E402
from src.processing.hybrid_search import HybridSearcher  # noqa: E402
from src.processing.rag import RAGPipeline  # noqa: E402

st.set_page_config(page_title="PLU RAG AI — Interface de test", layout="wide")
st.title("PLU RAG AI — Interface de test")
st.caption(
    "La localisation est résolue vers la commune via BAN. L'OCR intervient à l'ingestion "
    "des PDF scannés pour alimenter la base vectorielle."
)

ban = BANClient()


@st.cache_data(ttl=300)
def cached_suggestions(query: str) -> list[dict]:
    return [s.__dict__ for s in ban.suggest_addresses(query, limit=5)]


@st.cache_data(ttl=300)
def cached_commune_documents(collection_name: str, commune: str) -> list[dict]:
    store = VectorStore(collection_name)
    docs = store.get_all_documents(where={"commune": commune})
    rows: list[dict] = []
    for doc in docs:
        rows.append(
            {
                "id": doc.id,
                "commune": doc.metadata.get("commune", ""),
                "zone": doc.metadata.get("zone", ""),
                "chapter": doc.metadata.get("chapter", ""),
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", ""),
                "filename": doc.metadata.get("filename", ""),
                "score": doc.score,
                "text": doc.text[:300],
            }
        )
    return rows


def _index_downloaded_plu(files: list[Path], collection_name: str, commune: str) -> int:
    store = VectorStore(collection_name)
    total_chunks = 0
    for pdf_path in files:
        total_chunks += ingest_pdf(pdf_path, store=store, source="PLU", commune=commune)
    HybridSearcher(store).build_bm25_index(where={"commune": commune})
    return total_chunks

mode = st.sidebar.radio(
    "Mode",
    ["Analyse par localisation", "RAG générique", "Agent PLU", "Agent National"],
)

if mode == "Analyse par localisation":
    st.subheader("Analyse d'une localisation")
    st.write(
        "Saisissez une adresse, un lieu ou une commune. L'application résout la commune, "
        "puis interroge la base PLU et la réglementation nationale associée."
    )
    col1, col2 = st.columns(2)
    plu_collection = col1.text_input("Collection PLU", value="plu_documents")
    national_collection = col2.text_input("Collection nationale", value="national_regulations")
    location = st.text_input("Lieu / adresse / commune")
    suggestions = cached_suggestions(location) if location and len(location) >= 3 else []
    selected = None
    if suggestions:
        options = [s["label"] for s in suggestions]
        choice = st.selectbox("Suggestions BAN", options)
        selected = next((s for s in suggestions if s["label"] == choice), None)
    project_question = st.text_area(
        "Question projet (optionnelle)",
        placeholder="Ex: extension de maison, création d'une annexe, changement de destination...",
    )
    if selected:
        st.markdown(f"**Commune détectée :** {selected['commune']} ({selected['postcode']})")
        st.map([{"lat": selected["lat"], "lon": selected["lon"]}])

    if st.button("Analyser la localisation") and (location or selected):
        with st.spinner("Recherche en cours..."):
            agent = ParcelleAgent(plu_collection=plu_collection, national_collection=national_collection)
            query_value = selected["label"] if selected else location
            result = agent.qualify(query_value, project_question=project_question or None)
        st.markdown(f"### Commune résolue : {result.commune or 'non résolue'}")
        st.markdown("### Synthèse")
        st.write(result.synthesis)
        with st.expander("Détail PLU"):
            st.write(result.plu_answer)
        with st.expander("Détail réglementation nationale"):
            st.write(result.national_answer)

        if result.commune:
            st.markdown("### Documents PLU indexés pour cette commune")
            docs = cached_commune_documents(plu_collection, result.commune)
            if docs:
                labels = [
                    f"{doc['filename']} | p.{doc['page']} | zone {doc['zone'] or 'N/A'} | {doc['chapter'] or 'sans chapitre'}"
                    for doc in docs
                ]
                selected_doc_label = st.selectbox("Document à explorer", labels)
                selected_doc = docs[labels.index(selected_doc_label)]
                st.dataframe(docs, use_container_width=True, hide_index=True)
                st.markdown("### Extrait du document sélectionné")
                st.caption(
                    f"{selected_doc['filename']} | commune {selected_doc['commune']} | zone {selected_doc['zone'] or 'N/A'} | page {selected_doc['page']}"
                )
                st.write(selected_doc["text"])
            else:
                st.info("Aucun document indexé trouvé pour cette commune dans la collection PLU.")

            if st.button("Télécharger et indexer les PLU de cette commune"):
                with st.spinner("Téléchargement et indexation en cours..."):
                    downloader = PLUDownloader()
                    downloaded = downloader.download_for_commune(result.commune)
                    if downloaded:
                        chunks = _index_downloaded_plu(downloaded, plu_collection, result.commune)
                        st.cache_data.clear()
                        st.success(f"{len(downloaded)} fichier(s) téléchargé(s), {chunks} chunk(s) indexé(s).")
                    else:
                        st.warning("Aucun document PLU n'a pu être téléchargé pour cette commune.")

elif mode == "RAG générique":
    st.subheader("Pipeline RAG générique")
    collection = st.text_input("Collection", value="plu_documents")
    question = st.text_area("Question")
    commune = st.text_input("Commune / adresse (filtre optionnel)")
    if st.button("Interroger") and question:
        with st.spinner("Recherche en cours..."):
            pipeline = RAGPipeline(collection_name=collection)
            result = pipeline.ask(question, address_or_commune=commune or None)
        st.markdown("### Réponse")
        st.write(result.answer)
        st.markdown("### Sources")
        for src in result.sources:
            meta = src.metadata
            st.caption(
                f"{meta.get('commune', '')} | zone {meta.get('zone', 'N/A')} | "
                f"p.{meta.get('page', '?')} | score={src.rrf_score:.4f}"
            )
            st.text(src.text[:400])

elif mode == "Agent PLU":
    st.subheader("Agent PLU (filtré commune/zone)")
    collection = st.text_input("Collection", value="plu_documents")
    question = st.text_area("Question")
    col1, col2 = st.columns(2)
    commune = col1.text_input("Commune")
    zone = col2.text_input("Zone (ex: UA)")
    if st.button("Interroger l'agent PLU") and question:
        with st.spinner("Recherche en cours..."):
            agent = PLUAgent(collection_name=collection)
            result = agent.answer(question, commune=commune or None, zone=zone or None)
        st.markdown("### Réponse")
        st.write(result.answer)

else:
    st.subheader("Agent réglementation nationale")
    collection = st.text_input("Collection", value="national_regulations")
    question = st.text_area("Question")
    source_tag = st.text_input("Filtre par source (tag, ex: re2020)")
    if st.button("Interroger l'agent National") and question:
        with st.spinner("Recherche en cours..."):
            agent = NationalAgent(collection_name=collection)
            result = agent.answer(question, source_tag=source_tag or None)
        st.markdown("### Réponse")
        st.write(result.answer)
