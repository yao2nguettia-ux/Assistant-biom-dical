import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.guardrails import (
    DISCLAIMER,
    build_explanation_prompt,
    contains_contradiction,
    format_response,
)


def test_disclaimer_is_not_empty():
    assert isinstance(DISCLAIMER, str)
    assert "conseil médical" in DISCLAIMER.lower() or "médical" in DISCLAIMER.lower()


def test_build_prompt_for_yes_verdict_includes_verdict_and_citation():
    prompt = build_explanation_prompt("La molécule est-elle efficace ?", "Les résultats montrent une efficacité significative.", "yes")
    assert "OUI" in prompt
    assert "Les résultats montrent une efficacité significative." in prompt
    assert "La molécule est-elle efficace ?" in prompt


def test_build_prompt_for_incertain_does_not_state_a_verdict():
    prompt = build_explanation_prompt("Est-ce risqué ?", "Les données sont insuffisantes.", "incertain")
    assert "OUI" not in prompt
    assert "NON" not in prompt
    assert "ne peut pas conclure" in prompt.lower()


def test_contains_contradiction_detects_opposite_claim_for_yes():
    verdict = "yes"
    explanation = "Cependant, cette étude infirme largement cette hypothèse."
    assert contains_contradiction(explanation, verdict) is True


def test_contains_contradiction_false_when_consistent():
    verdict = "yes"
    explanation = "Les résultats confirment clairement l'effet observé dans l'étude."
    assert contains_contradiction(explanation, verdict) is False


def test_contains_contradiction_for_no_verdict():
    verdict = "no"
    explanation = "L'étude confirme que le traitement n'a aucun effet significatif ici."
    # Contient bien le marqueur "confirme que" -> signalé pour revue manuelle
    assert contains_contradiction(explanation, verdict) is True


def test_contains_contradiction_maybe_never_flagged():
    assert contains_contradiction("Texte quelconque, confirme, infirme, oui, non.", "maybe") is False


def test_format_response_structure():
    response = format_response("yes", 0.873456, "Une phrase source.", "Une explication.")
    assert response["verdict"] == "yes"
    assert response["confidence"] == 0.8735  # arrondi à 4 décimales
    assert response["citation"] == "Une phrase source."
    assert response["disclaimer"] == DISCLAIMER
