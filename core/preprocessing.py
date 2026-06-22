import re

def preprocess_text(text: str) -> str:
    """
    Nettoie un texte utilisateur : supprime les espaces multiples,
    met en minuscule, enlève quelques caractères parasites.
    """
    if not text:
        return ""

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text
