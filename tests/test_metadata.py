from src.processing.metadata import build_metadata, detect_chapter, detect_zone


def test_detect_zone_found():
    assert detect_zone("Les constructions en zone UA doivent respecter...") == "UA"


def test_detect_zone_not_found():
    assert detect_zone("Aucune mention de zonage ici.") is None


def test_detect_chapter_hauteur():
    assert detect_chapter("La hauteur maximale des bâtiments est de 12 mètres") == "hauteur"


def test_detect_chapter_stationnement():
    assert detect_chapter("Le nombre de places de stationnement requis est de 1 par logement") == "stationnement"


def test_build_metadata_full():
    meta = build_metadata(
        text="En zone UB, la hauteur maximale est fixée à 15 mètres.",
        commune="Nantes",
        source="PLU",
        page=32,
        filename="plu_nantes.pdf",
    )
    d = meta.to_dict()
    assert d["commune"] == "Nantes"
    assert d["zone"] == "UB"
    assert d["chapter"] == "hauteur"
    assert d["page"] == 32
    assert d["filename"] == "plu_nantes.pdf"
