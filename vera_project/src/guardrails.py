"""Garde-fous du produit : avertissement, construction de prompt d'explication,
et détection heuristique de contradiction entre le verdict et l'explication
générée par le LLM.

Aucune dépendance à un modèle ici : ces fonctions sont pures et testables
indépendamment du GPU.
"""

from typing import Dict, List

DISCLAIMER = "Cet outil ne fournit pas de conseil médical ni de diagnostic."

VERDICT_LABELS_FR = {
    "yes": "OUI",
    "no": "NON",
    "maybe": "INCERTAIN (peut-être)",
}

CONTRADICTION_MARKERS: Dict[str, List[str]] = {
    "yes": ["ne confirme pas", "infirme", "réponse est non", "réponse : non"],
    "no": ["confirme que", "réponse est oui", "réponse : oui"],
    "maybe": [],
}


def build_explanation_prompt(question: str, citation: str, verdict: str) -> str:
    """Construit le prompt envoyé au LLM génératif pour expliquer un verdict
    déjà décidé par le classifieur. Le LLM ne doit jamais changer ce verdict.
    """
    if verdict == "incertain":
        instruction = (
            "Le système ne peut pas conclure avec suffisamment de confiance. "
            "Explique en 2 phrases maximum pourquoi la phrase source ci-dessous "
            "ne permet pas de répondre clairement à la question, sans avancer "
            "toi-même un verdict."
        )
    else:
        verdict_fr = VERDICT_LABELS_FR.get(verdict, verdict)
        instruction = (
            f"Le verdict déterminé est : {verdict_fr}. "
            "Explique en 2 phrases maximum, en te basant UNIQUEMENT sur la phrase "
            "source ci-dessous, pourquoi ce verdict est cohérent. Ne contredis "
            "jamais ce verdict et n'utilise aucune connaissance extérieure à la "
            "phrase source."
        )

    return (
        f"{instruction}\n\n"
        f"Question : {question}\n"
        f"Phrase source : {citation}\n\n"
        "Explication concise :"
    )


def contains_contradiction(explanation: str, verdict: str) -> bool:
    """Filet de sécurité heuristique : détecte si l'explication semble
    affirmer littéralement le contraire du verdict. Ce n'est pas une garantie
    absolue, seulement un signal pour une revue manuelle.
    """
    markers = CONTRADICTION_MARKERS.get(verdict)
    if not markers:
        return False
    explanation_lower = explanation.lower()
    return any(marker in explanation_lower for marker in markers)


def format_response(verdict: str, confidence: float, citation: str, explanation: str) -> dict:
    """Formate la réponse finale telle qu'elle sera affichée à l'utilisateur."""
    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "citation": citation,
        "explanation": explanation,
        "disclaimer": DISCLAIMER,
    }
