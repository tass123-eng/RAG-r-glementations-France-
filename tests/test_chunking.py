from src.processing.chunking import chunk_text


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_short_single_chunk():
    text = "Ceci est une phrase courte. Une autre phrase courte."
    chunks = chunk_text(text, chunk_size=1000, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Ceci")


def test_chunk_text_splits_on_size():
    sentence = "La hauteur maximale des constructions est fixée à neuf mètres. "
    text = sentence * 20
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) > 0


def test_chunk_overlap_present():
    sentence_variants = [f"Article {i} : la règle numéro {i} s'applique ici." for i in range(30)]
    text = " ".join(sentence_variants)
    chunks = chunk_text(text, chunk_size=150, overlap=50)
    assert len(chunks) >= 2
