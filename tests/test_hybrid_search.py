from dataclasses import dataclass

from src.processing.hybrid_search import HybridSearcher, _tokenize
from src.processing.vector_store import SearchResult


class _FakeStore:
    """Store minimal qui simule get_all_documents sans dépendance réseau/modèle."""

    def __init__(self, docs: list[SearchResult]):
        self._docs = docs

    def get_all_documents(self, where=None):
        return self._docs

    def query(self, query_text, top_k=30, where=None):
        # Pas utilisé dans ces tests (pas de recherche vectorielle réelle)
        return []


def test_tokenize_lowercases():
    assert _tokenize("Hauteur MAXIMALE") == ["hauteur", "maximale"]


def test_bm25_search_ranks_relevant_doc_first():
    docs = [
        SearchResult(id="1", text="stationnement obligatoire un emplacement par logement", metadata={}, score=0.0),
        SearchResult(id="2", text="hauteur maximale des constructions neuf metres zone UA", metadata={}, score=0.0),
        SearchResult(id="3", text="espaces verts plantations obligatoires en limite separative", metadata={}, score=0.0),
    ]
    searcher = HybridSearcher(store=_FakeStore(docs))
    searcher.build_bm25_index()

    results = searcher._bm25_search("hauteur maximale zone UA", top_k=3)
    assert results[0].id == "2"


def test_bm25_search_empty_corpus_returns_empty():
    searcher = HybridSearcher(store=_FakeStore([]))
    searcher.build_bm25_index()
    assert searcher._bm25_search("quoi que ce soit", top_k=5) == []


def test_bm25_search_respects_where_filter():
    docs = [
        SearchResult(id="1", text="stationnement obligatoire", metadata={"commune": "Nantes"}, score=0.0),
        SearchResult(id="2", text="hauteur maximale", metadata={"commune": "Rennes"}, score=0.0),
    ]
    store = _FakeStore(docs)
    searcher = HybridSearcher(store=store)
    searcher.build_bm25_index(where={"commune": "Nantes"})

    results = searcher._bm25_search("stationnement", top_k=5, where={"commune": "Nantes"})

    assert [result.id for result in results] == ["1"]
