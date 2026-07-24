# Data Card — Véra (assistante de lecture biomédicale calibrée)

## Source des données

**PubMedQA** — https://github.com/pubmedqa/pubmedqa

Trois sous-ensembles existent :
- `pqa_labeled` (1 000 exemples) : annoté par des experts, labels yes/no/maybe fiables
- `pqa_unlabeled` (~61 000 exemples) : sans label final
- `pqa_artificial` (~211 000 exemples) : labels générés automatiquement

**Licence :** vérifier les conditions spécifiques à chaque sous-ensemble avant
toute redistribution ou usage commercial (non vérifié de manière exhaustive
dans le cadre de ce projet pédagogique).

## Transformations appliquées

1. **Aplatissement** : chaque exemple est réduit à `{question, context, label}`,
   le contexte étant la concaténation des sections de l'abstract.
2. **Nettoyage** : suppression des exemples à contexte vide ou label invalide.
3. **Échantillonnage** : 20 000 exemples tirés de `pqa_artificial` (échantillonnage
   stratifié par label) pour l'entraînement, afin de rester compatible avec les
   ressources Colab gratuites.
4. **Correction du déséquilibre "maybe"** (point important) : `pqa_artificial`
   ne contient presque aucun exemple labellisé *maybe*. Les 1 000 exemples de
   `pqa_labeled` ont donc été redécoupés en 3 parts stratifiées :
   - ~60 % réinjectés dans le train (exposition à la classe *maybe*)
   - ~20 % ajoutés à la validation
   - ~20 % réservés comme **test final held-out**, jamais vu à l'entraînement

## Biais et limites connues

- **Déséquilibre de classe résiduel** : même après correction, *maybe* reste
  minoritaire. Une pondération de la fonction de perte (`class_weight`) est
  appliquée pendant le fine-tuning, mais la classe reste la plus difficile
  (voir l'analyse d'erreurs du Jour 4).
- **Origine automatique des labels** : les labels de `pqa_artificial` sont
  eux-mêmes dérivés automatiquement des abstracts, avec un bruit potentiel
  hérité de cette génération.
- **Absence de données hors périmètre** : PubMedQA ne contient que des
  questions biomédicales. Le détecteur hors périmètre a donc été testé sur un
  jeu **synthétique** de questions non biomédicales construit manuellement,
  qui n'est pas représentatif de la diversité réelle des requêtes hors
  périmètre possibles en production.
- **Contexte fourni, pas de recherche à grande échelle** : contrairement à un
  RAG classique sur un corpus ouvert, le contexte pertinent est fourni avec
  chaque exemple PubMedQA. Le FAISS ici sert à citer la phrase exacte dans ce
  contexte donné, pas à rechercher parmi des millions de documents.
- **Aucune donnée personnelle ou sensible** : PubMedQA porte sur des résumés
  scientifiques publiés, pas sur des dossiers patients.

## Usage prévu et interdit

- ✅ Aide à la lecture rapide de littérature biomédicale par des chercheurs
- ✅ Projet pédagogique / démonstration technique
- ❌ Diagnostic médical, conseil de traitement, usage patient direct
- ❌ Décision clinique sans validation humaine
