"""Serveur Flask d'analyse biomédicale Véra.

Expose l'interface web (vera.html) et les endpoints d'API pour le verdict,
la recherche par FAISS et la comparaison d'études biomédicales.
"""

import json
import os

from flask import Flask, jsonify, request, send_from_directory

from src.pipeline import BiomedicalAssistant, PipelineConfig

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "./checkpoints")
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=None)


def load_config() -> PipelineConfig:
    """Charge la configuration du pipeline à partir des fichiers d'artefacts."""
    config = PipelineConfig(
        adapter_dir=f"{CHECKPOINTS_DIR}/pubmedbert_lora_adapter_final",
        faiss_index_path=f"{CHECKPOINTS_DIR}/biomedical_corpus.index",
    )

    calibration_path = f"{RESULTS_DIR}/day3_calibration.json"
    abstention_path = f"{RESULTS_DIR}/day3_abstention.json"
    ood_path = f"{RESULTS_DIR}/day3_ood_detection.json"

    if os.path.exists(calibration_path):
        with open(calibration_path, encoding="utf-8") as f:
            config.temperature = json.load(f)["temperature"]

    if os.path.exists(abstention_path):
        with open(abstention_path, encoding="utf-8") as f:
            config.abstention_threshold = json.load(f)["threshold"]

    if os.path.exists(ood_path):
        with open(ood_path, encoding="utf-8") as f:
            config.oos_threshold = json.load(f)["oos_threshold"]

    return config


# Chargement conditionnel du modèle
assistant = None
try:
    print("Initialisation du modèle biomédical...")
    assistant = BiomedicalAssistant(load_config())
    print("Modèle chargé avec succès. Serveur disponible sur http://127.0.0.1:5000")
except Exception as err:
    print(f"Information : Artefacts non trouvés ({err}). Mode démonstration local actif.")


@app.route("/")
def index():
    """Sert l'interface web principale (index.html ou vera.html)."""
    filename = "index.html" if os.path.exists(os.path.join(STATIC_DIR, "index.html")) else "vera.html"
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Endpoint principal d'analyse d'une question biomédicale."""
    if assistant is None:
        return jsonify({"error": "Modèle non chargé en local (artefacts absents)."}), 503
    payload = request.get_json(force=True, silent=True) or {}
    question = (payload.get("question") or "").strip()
    context = (payload.get("context") or "").strip()

    if not question:
        return jsonify({"error": "La question est requise."}), 400

    result = assistant.answer(question, context)
    return jsonify(result)


@app.route("/api/compare", methods=["POST"])
def compare():
    """Endpoint de comparaison de deux études ou contextes."""
    if assistant is None:
        return jsonify({"error": "Modèle non chargé en local (artefacts absents)."}), 503
    payload = request.get_json(force=True, silent=True) or {}
    question = (payload.get("question") or "").strip()
    context_a = (payload.get("context_a") or "").strip()
    context_b = (payload.get("context_b") or "").strip()

    if not question or not context_a or not context_b:
        return jsonify({"error": "La question et les deux contextes sont requis."}), 400

    result = assistant.compare(question, context_a, context_b)
    return jsonify(result)


@app.route("/api/health")
def health():
    """Vérification de l'état du serveur."""
    return jsonify({"status": "ok", "pipeline_loaded": assistant is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
