# Véra — assistante de lecture biomédicale calibrée

Assistant de question-réponse biomédical fondé sur PubMedQA : à partir d'une
question et d'un contexte (résumé PubMed), Véra produit un verdict **oui /
non / peut-être**, calibré (avec possibilité d'abstention), justifié par une
citation du passage source, et accompagné d'une explication en langage
naturel.

> ⚠️ **Cet outil ne fournit pas de conseil médical ni de diagnostic.** Il est
> destiné à l'aide à la lecture de littérature biomédicale par des
> professionnels de la recherche, pas à un usage patient direct.

## Architecture

```
PubMedQA (artificial + labeled)
        │
        ▼
  Tokenizer PubMedBERT
        │
        ▼
Transformer + LoRA (r=8, alpha=16) ──► Verdict brut (yes/no/maybe)
        │
        ▼
  Calibrateur (température scaling) ──► Score de confiance + abstention
        │
        ▼
  FAISS (MiniLM-L6-v2) ──► Passage source cité + détection hors périmètre
        │
        ▼
  LLM génératif (Qwen2.5-1.5B, prompté) ──► Explication en langage naturel
        │
        ▼
  API Flask (app.py) ──► /api/analyze, /api/compare ──► Interface web (vera.html)
```

Si le serveur Flask n'est pas joignable, `vera.html` bascule automatiquement
sur une démonstration locale simulée (mention explicite « mode démonstration
hors-ligne » affichée à l'écran) plutôt que de bloquer l'utilisateur.

## Fonctionnalités de l'interface

- **Analyser** : verdict + confiance + **distribution complète des trois
  probabilités** (oui/non/peut-être) + phrase citée + explication
- **Signaler une incohérence** : marque le résultat courant comme "à revoir"
  dans l'historique
- **Exporter la fiche** : génère une fiche imprimable (question, verdict,
  citation, explication) via l'impression navigateur (Ctrl+P / Enregistrer en PDF)
- **Comparer** : une question, deux contextes (deux études) → verdict de
  cohérence (accord / contradiction / partiel)
- **Lot (CSV)** : importe un CSV (`question`, `context`) et analyse jusqu'à
  50 lignes d'un coup, avec export des résultats en CSV
- **Historique** : les analyses de la session (y compris comparaisons et
  lots), non persistées après fermeture de la page
- **À propos** : garde-fous du système

## Structure du dépôt

```
.
├── notebooks/
│   ├── jour1_chargement_baseline.ipynb       # Données + baseline zero-shot
│   ├── jour2_finetuning_lora.ipynb           # Fine-tuning LoRA PubMedBERT
│   ├── jour3_calibration_faiss.ipynb         # Calibration + FAISS + garde-fous
│   └── jour4_llm_explication_erreurs.ipynb   # LLM explicatif + analyse d'erreurs
├── src/
│   ├── data_utils.py      # Préparation des données (testé, sans dépendance GPU)
│   ├── guardrails.py       # Garde-fous, prompts, détection de contradiction (testé)
│   └── pipeline.py         # Orchestration complète (nécessite GPU + artefacts)
├── tests/
│   ├── test_data_utils.py
│   └── test_guardrails.py
├── app.py                  # Serveur Flask : API /api/analyze + sert vera.html
├── vera.html                # Interface web (HTML/CSS/JS, sidebar de navigation)
├── requirements.txt
├── data_card.md
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Reproduire l'entraînement

Exécuter les notebooks dans l'ordre sur Google Colab (GPU T4 gratuit suffit) :
`jour1` → `jour2` → `jour3` → `jour4`. Chaque notebook sauvegarde ses
artefacts sur Google Drive (`/content/drive/MyDrive/assistant_biomedical/`).

## Lancer l'interface complète

Télécharger les artefacts (`checkpoints/`, `results/`) depuis Drive vers ce
dépôt, puis :

```bash
python app.py
```

Ouvrir ensuite **http://127.0.0.1:5000** dans un navigateur.

## Tests

Les fonctions de logique pure (préparation des données, garde-fous) sont
testées indépendamment du modèle, pour un feedback rapide sans GPU :

```bash
pytest tests/ -v
```

## Limites connues

- Le sous-ensemble `pqa_artificial` de PubMedQA ne contient quasiment que des
  labels *yes/no* ; la classe *maybe* a été réinjectée depuis les exemples
  experts (`pqa_labeled`) — voir `data_card.md` pour le détail du traitement.
- La détection hors périmètre s'appuie sur une distance d'embeddings à un
  corpus biomédical, testée uniquement sur un jeu synthétique de questions
  non biomédicales (le dataset PubMedQA n'en contient pas nativement).
- Le contrôle de cohérence verdict/explication est une heuristique par
  mots-clés, pas une garantie formelle — les cas signalés méritent une revue
  manuelle.
- L'historique de session dans l'interface n'est pas persisté (pas de
  stockage navigateur) : il se réinitialise à chaque rechargement de page.
- Projet réalisé à des fins pédagogiques (capstone bootcamp GenAI/Data) ; non
  validé pour un usage clinique réel.

## Licence

À compléter selon les conditions de licence de PubMedQA et des modèles
Hugging Face utilisés (voir `data_card.md`).
