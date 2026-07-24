"""API + interface web Véra — assistante de lecture biomédicale calibrée.

Ce serveur fait deux choses :
1. Il sert la page `vera.html` (interface HTML/CSS/JS) à l'adresse `/`.
2. Il expose `/api/analyze` (POST) que le JavaScript de la page appelle pour
   obtenir un vrai verdict du pipeline (classifieur calibré + FAISS + LLM
   explicatif), au lieu de la simulation locale utilisée quand le serveur
   n'est pas joignable.

Usage local (après avoir récupéré les artefacts `checkpoints/` et `results/`
depuis Google Drive vers ce dépôt, à la suite des notebooks Jour 1 à 4) :

    python app.py

Puis ouvrir http://127.0.0.1:5000 dans un navigateur.

Les chemins et hyperparamètres (température, seuils) sont chargés depuis les
fichiers JSON produits au Jour 3, pour éviter toute valeur codée en dur.
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


print("Chargement du pipeline (peut prendre 1-2 minutes la première fois)...")
assistant = BiomedicalAssistant(load_config())
print("Pipeline prêt. Serveur sur http://127.0.0.1:5000")


@app.route("/")
def index():
    """Sert l'interface Véra (vera.html), qui appelle /api/analyze en JS."""
    return send_from_directory(STATIC_DIR, "vera.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(force=True, silent=True) or {}
    question = (payload.get("question") or "").strip()
    context = (payload.get("context") or "").strip()

    if not question:
        return jsonify({"error": "La question est vide."}), 400

    result = assistant.answer(question, context)
    return jsonify(result)


@app.route("/api/compare", methods=["POST"])
def compare():
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
    """Petite route de contrôle : utile pour vérifier que le serveur tourne
    avant une démonstration, sans passer par l'interface."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # host="0.0.0.0" pour rester joignable si le serveur tourne sur Colab
    # avec un tunnel (ngrok) ; debug=False pour éviter un rechargement qui
    # relancerait le chargement du modèle en pleine démonstration.
    app.run(host="0.0.0.0", port=5000, debug=False)
