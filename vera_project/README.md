# Véra — Assistante de lecture et d'analyse biomédicale

Plateforme d'aide à la lecture de la littérature biomédicale basée sur le corpus **PubMedQA**. À partir d'une question clinique et d'un résumé d'étude PubMed, le système évalue la certitude scientifique (`oui` / `non` / `peut-être`), extrait la citation justificative et synthétise une explication structurée.

> **Avertissement :** Outil de recherche académique. Ne constitue pas un dispositif médical et ne fournit aucun diagnostic.

---

## Sommaire

- [Architecture du Pipeline](#architecture-du-pipeline)
- [Fonctionnalités](#fonctionnalités)
- [Structure du Dépôt](#structure-du-dépôt)
- [Installation et Démarrage](#installation-et-démarrage)
- [Entraînement des Modèles](#entraînement-des-modèles)
- [Tests Unitaires](#tests-unitaires)
- [Limites et Garde-fous](#limites-et-garde-fous)

---

## Architecture du Pipeline

```text
    ┌─────────────────────────────────────────┐
    │             Corpus PubMedQA             │
    └────────────────────┬────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │        Tokenisation PubMedBERT          │
    └────────────────────┬────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │        PubMedBERT + LoRA (r=8)          │
    └────────────────────┬────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌───────────────┐                 ┌───────────────┐
│ Calibration   │                 │ Index FAISS   │
│ & Abstention  │                 │  (Citation)   │
└───────┬───────┘                 └───────┬───────┘
        │                                 │
        └────────────────┬────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │         Module Explicatif LLM           │
    └────────────────────┬────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │      API Flask (app.py) ──► vera.html   │
    └─────────────────────────────────────────┘
```

### Description détaillée des composants

1. **Préparation et Tokenisation des textes :**  
   - Normalisation Unicode NFD et nettoyage des paires question-contexte du corpus PubMedQA.
   - Tokenisation spécialisée via le vocabulaire biomédical `PubMedBERT` (28 000 subwords).

2. **Classification et Inférence LoRA :**  
   - Fine-tuning d'un backbone Transformer `PubMedBERT-abs` (110M de paramètres) avec des matrices d'adaptation de bas rang LoRA (`r=8`, `alpha=16`).
   - Prédiction des logits pour les 3 catégories de réponse (`yes`, `no`, `maybe`).

3. **Module de Calibration et d'Abstention :**  
   - Application d'un scaling de température sur les logits pour convertir les scores en probabilités calibrées et interprétables.
   - Mécanisme d'abstention automatique lorsque le score de certitude passe sous le seuil d'incertitude.

4. **Indexation et Recherche Vectorielle (FAISS) :**  
   - Segmentation du texte de l'étude en phrases distinctes et projection dans un espace d'embedding dense (`all-MiniLM-L6-v2`).
   - Recherche du vecteur le plus proche via index FAISS pour isoler et extraire la citation exacte qui étaye le verdict.

5. **Couche d'Exposition et Service API :**  
   - Micro-service Web basé sur Flask (`app.py`) exposant les endpoints REST `/api/analyze` et `/api/compare`.
   - Interface graphique dynamique (`vera.html`) assurant le rendu des verdicts, l'animation du score de confiance et le repli autonome en mode hors-ligne.

---

## Fonctionnalités

- **Analyse individuelle :** Calcul du verdict, du score de confiance et affichage de la citation source.
- **Comparaison d'études :** Analyse comparative de deux résumés d'études sur une même question clinique.
- **Traitement par lot (CSV) :** Inférence groupée jusqu'à 50 lignes avec export des résultats au format CSV.
- **Historique de session :** Suivi des évaluations réalisées au cours de la session utilisateur.

---

## Structure du Dépôt

```
.
├── notebooks/
│   ├── jour1_chargement_baseline.ipynb       # Exploration des données PubMedQA
│   ├── jour2_finetuning_lora.ipynb           # Fine-tuning LoRA de PubMedBERT
│   ├── jour3_calibration_faiss.ipynb         # Calibration en température & FAISS
│   └── jour4_llm_explication_erreurs.ipynb   # Module d'explication & revue d'erreurs
├── src/
│   ├── data_utils.py                         # Prétraitement et nettoyage des textes
│   ├── guardrails.py                         # Contrôle de cohérence et garde-fous
│   └── pipeline.py                           # Pipeline d'inférence principal
├── tests/
│   ├── test_data_utils.py                    # Tests unitaires du prétraitement
│   └── test_guardrails.py                    # Tests unitaires des garde-fous
├── app.py                                    # Serveur Flask et routes API
├── vera.html                                 # Interface utilisateur HTML/CSS/JS
├── requirements.txt                          # Dépendances du projet
└── README.md                                 # Documentation du projet
```

---

## Installation et Démarrage

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. Démarrage du serveur local

```bash
python app.py
```

L'interface est accessible sur **http://127.0.0.1:5000**.

---

## Entraînement des Modèles

Les notebooks du dossier `notebooks/` permettent de reproduire la chaîne de traitement séquentiellement :

1. `jour1_chargement_baseline.ipynb`
2. `jour2_finetuning_lora.ipynb`
3. `jour3_calibration_faiss.ipynb`
4. `jour4_llm_explication_erreurs.ipynb`

---

## Tests Unitaires

Exécution des tests de logique et de prétraitement sans dépendance GPU :

```bash
pytest tests/ -v
```

---

## Limites et Garde-fous

- **Répartition des labels :** Traitement d'équilibrage appliqué pour la classe *maybe* via `pqa_labeled`.
- **Détection de périmètre :** Repose sur la distance d'embedding par rapport au corpus d'apprentissage.
- **Absence de persistance :** L'historique de session est conservé en mémoire locale du navigateur.
