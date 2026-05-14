# Rapport d'incident — Modèle ML manquant

## Informations générales

- Date de l'incident : 14/05/2026
- Durée : 10 minutes
- Sévérité : Critique
- Statut : Résolu

## Description de l'incident

Au démarrage de l'API FastAPI, le fichier de modèle
xgboost_fruits_legumes.pkl était introuvable. L'API démarrait
correctement mais ne pouvait pas effectuer de prédictions.

## Symptômes observés

- Message dans les logs : ATTENTION modèle non trouvé
- Endpoint GET /health retournait modele_charge: false
- Endpoint POST /predict retournait une erreur 500

## Cause identifiée

Le fichier .pkl avait été renommé accidentellement lors d'une
manipulation dans le dossier NOTEBOOKS/models/. DVC ne peut pas
restaurer automatiquement le fichier sans accès au stockage distant.

## Procédure de résolution

1. Identifier le fichier de sauvegarde présent dans le dossier
2. Restaurer le fichier avec la commande :
   Rename-Item "xgboost_fruits_legumes.pkl.backup"
   "xgboost_fruits_legumes.pkl"
3. Relancer l'API et vérifier que modele_charge: true
4. Vérifier que POST /predict retourne 200 OK

## Mesures correctives

Ajout d'un gestionnaire d'erreur explicite dans FastAPI pour
afficher un message clair quand le modèle est absent, au lieu
d'une erreur 500 silencieuse.

## Feedback loop MLOps

Cet incident montre l'importance de versionner le modèle avec DVC
et de configurer une restauration automatique en cas de fichier
manquant. Un seuil d'alerte MLflow sur modele_charge=false
permettrait de détecter ce type d'incident en production.