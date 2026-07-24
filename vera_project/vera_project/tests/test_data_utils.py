import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import (
    flatten_example,
    clean_dataset,
    stratified_sample,
    stratified_split,
    label_distribution,
    split_sentences,
    compute_ece,
)


def make_raw_example(question, contexts, label):
    return {
        "question": question,
        "context": {"contexts": contexts},
        "final_decision": label,
    }


def test_flatten_example_joins_contexts():
    raw = make_raw_example("Est-ce efficace ?", ["Phrase A.", "Phrase B."], "Yes")
    flat = flatten_example(raw)
    assert flat["question"] == "Est-ce efficace ?"
    assert flat["context"] == "Phrase A. Phrase B."
    assert flat["label"] == "yes"


def test_clean_dataset_removes_invalid_labels_and_empty_context():
    data = [
        {"question": "q1", "context": "un contexte", "label": "yes"},
        {"question": "q2", "context": "", "label": "no"},
        {"question": "q3", "context": "un contexte", "label": "unknown"},
        {"question": "q4", "context": "un contexte", "label": "maybe"},
    ]
    cleaned = clean_dataset(data)
    assert len(cleaned) == 2
    assert {ex["label"] for ex in cleaned} == {"yes", "maybe"}


def test_stratified_sample_preserves_approximate_proportions():
    data = (
        [{"question": f"q{i}", "context": "c", "label": "yes"} for i in range(80)]
        + [{"question": f"q{i}", "context": "c", "label": "no"} for i in range(20)]
    )
    sample = stratified_sample(data, n_total=50, seed=1)
    dist = label_distribution(sample)
    # Proportions ~80/20 attendues sur un échantillon de 50 -> ~40/10
    assert 35 <= dist["yes"] <= 45
    assert 5 <= dist["no"] <= 15


def test_stratified_sample_returns_all_if_n_total_too_large():
    data = [{"question": "q", "context": "c", "label": "yes"}] * 5
    sample = stratified_sample(data, n_total=100)
    assert len(sample) == 5


def test_stratified_split_ratios_sum_correctly():
    data = [{"question": f"q{i}", "context": "c", "label": "yes"} for i in range(100)]
    part1, part2, part3 = stratified_split(data, ratios=(0.6, 0.2, 0.2), seed=0)
    assert len(part1) + len(part2) + len(part3) == 100
    assert 55 <= len(part1) <= 65
    assert 15 <= len(part2) <= 25
    assert 15 <= len(part3) <= 25


def test_stratified_split_no_overlap_between_parts():
    data = [{"question": f"q{i}", "context": "c", "label": "no"} for i in range(30)]
    part1, part2, part3 = stratified_split(data, seed=7)
    q1 = {ex["question"] for ex in part1}
    q2 = {ex["question"] for ex in part2}
    q3 = {ex["question"] for ex in part3}
    assert q1.isdisjoint(q2)
    assert q1.isdisjoint(q3)
    assert q2.isdisjoint(q3)


def test_split_sentences_basic():
    text = "Ceci est une phrase. Voici une deuxième phrase ! Et une troisième ?"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("Ceci")


def test_split_sentences_empty_text():
    assert split_sentences("") == []
    assert split_sentences(None if False else "") == []


def test_split_sentences_filters_short_fragments():
    text = "Ok. Ceci est une phrase correcte."
    sentences = split_sentences(text)
    # "Ok." fait moins de 5 caractères utiles et doit être filtré
    assert all(len(s) > 5 for s in sentences)


def test_compute_ece_perfect_calibration_is_zero():
    # Confiance = accuracy exacte dans chaque bin -> ECE proche de 0
    confidences = [0.9] * 10
    correct = [True] * 9 + [False]  # accuracy = 0.9, confiance moyenne = 0.9
    ece = compute_ece(confidences, correct, n_bins=10)
    assert ece < 0.05


def test_compute_ece_overconfident_model_has_high_ece():
    confidences = [0.99] * 10
    correct = [True] * 3 + [False] * 7  # accuracy = 0.3, confiance = 0.99
    ece = compute_ece(confidences, correct, n_bins=10)
    assert ece > 0.5


def test_compute_ece_empty_input():
    assert compute_ece([], []) == 0.0
