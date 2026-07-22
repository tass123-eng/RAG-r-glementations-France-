from src.processing.clean_text import clean_text, remove_boilerplate


def test_clean_text_collapses_whitespace():
    assert clean_text("Bonjour    le   monde") == "Bonjour le monde"


def test_clean_text_removes_hyphen_linebreak():
    text = "Le coeffi-\ncient d'emprise au sol"
    assert "coefficient" in clean_text(text)


def test_clean_text_collapses_blank_lines():
    text = "Ligne 1\n\n\n\nLigne 2"
    result = clean_text(text)
    assert "\n\n\n" not in result


def test_remove_boilerplate_removes_page_numbers():
    text = "Contenu utile\nPage 3/40\n12\nAutre contenu"
    result = remove_boilerplate(text)
    assert "Page 3/40" not in result
    assert "Contenu utile" in result
    assert "Autre contenu" in result
