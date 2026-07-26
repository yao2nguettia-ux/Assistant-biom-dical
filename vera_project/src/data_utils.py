"""
Fonctions de préparation et de nettoyage des données PubMedQA.

Ces fonctions sont volontairement indépendantes de tout modèle (pas de torch,
pas de transformers) afin de rester rapides à tester et faciles à réutiliser
dans les notebooks Colab comme dans l'application finale.
"""

import random
import re
from collections import Counter
from typing import Dict, List, Sequence


VALID_LABELS = ("yes", "no", "maybe")


def flatten_example(ex: dict) -> dict:
    """Convertit un exemple brut PubMedQA (format HuggingFace `datasets`) en un
    format simple {question, context, label}."""
    contexts = ex.get("context", {})
    if isinstance(contexts, dict) and "contexts" in contexts:
        context_text = " ".join(contexts["contexts"])
    else:
        context_text = str(contexts)

    return {
        "question": ex["question"],
        "context": context_text.strip(),
        "label": ex["final_decision"].strip().lower(),
    }


def clean_dataset(flat_examples: Sequence[dict]) -> List[dict]:
    """Retire les exemples avec un contexte vide ou un label invalide."""
    return [
        ex for ex in flat_examples
        if ex.get("context") and ex.get("label") in VALID_LABELS
    ]


def stratified_sample(data: Sequence[dict], n_total: int, seed: int = 42) -> List[dict]:
    """Échantillonne n_total exemples en conservant approximativement la
    proportion des classes d'origine."""
    if n_total >= len(data):
        return list(data)

    rng = random.Random(seed)
    by_label: Dict[str, List[dict]] = {label: [] for label in VALID_LABELS}
    for ex in data:
        by_label.setdefault(ex["label"], []).append(ex)

    total = len(data)
    sampled: List[dict] = []
    for label, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        n_label = round(n_total * len(items) / total) if total else 0
        sampled.extend(items[:n_label])

    rng.shuffle(sampled)
    return sampled


def stratified_split(data: Sequence[dict], ratios=(0.6, 0.2, 0.2), seed: int = 42):
    """Découpe un jeu de données en 3 parts stratifiées par label.

    Utilisé pour réinjecter une partie des exemples experts (`pqa_labeled`)
    dans le train/val, tout en réservant un test held-out jamais entraîné.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Les ratios doivent sommer à 1.0"

    rng = random.Random(seed)
    by_label: Dict[str, List[dict]] = {label: [] for label in VALID_LABELS}
    for ex in data:
        by_label.setdefault(ex["label"], []).append(ex)

    part1, part2, part3 = [], [], []
    for label, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n1 = round(n * ratios[0])
        n2 = round(n * ratios[1])
        part1.extend(items[:n1])
        part2.extend(items[n1:n1 + n2])
        part3.extend(items[n1 + n2:])

    for part in (part1, part2, part3):
        rng.shuffle(part)
    return part1, part2, part3


def label_distribution(data: Sequence[dict]) -> Counter:
    return Counter(ex["label"] for ex in data)


def split_sentences(text: str) -> List[str]:
    """Découpage simple en phrases, suffisant pour des abstracts PubMed."""
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def compute_ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 15) -> float:
    """Expected Calibration Error, implémentation pure Python/numpy-free.

    confidences : score de confiance (probabilité max) pour chaque prédiction
    correct     : booléen, la prédiction était-elle correcte ?
    """
    assert len(confidences) == len(correct)
    if not confidences:
        return 0.0

    n = len(confidences)
    ece = 0.0
    bin_edges = [i / n_bins for i in range(n_bins + 1)]

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bin_indices = [j for j in range(n) if lo < confidences[j] <= hi]
        if not bin_indices:
            continue
        bin_conf = sum(confidences[j] for j in bin_indices) / len(bin_indices)
        bin_acc = sum(1 for j in bin_indices if correct[j]) / len(bin_indices)
        ece += (len(bin_indices) / n) * abs(bin_conf - bin_acc)

    return ece
