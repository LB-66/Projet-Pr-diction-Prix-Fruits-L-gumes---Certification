# Sécurité OWASP Top 10 — API Fruits & Légumes ML

## Contexte

Ce document recense les protections mises en place sur l'API
FastAPI de prédiction des prix des fruits et légumes,
conformément au Top 10 OWASP.

---

## A01 : Contrôle d'accès

Chaque requête sur POST /predict doit présenter un header
X-API-Key valide. Sans clé : 403 Forbidden. Avec clé : 200 OK.

## A03 : Injection

Validation stricte des 12 types de features via Pydantic.
Tout type incorrect retourne automatiquement 422.

## A05 : Mauvaise configuration

Toutes les valeurs sensibles dans le fichier .env.
Fichier .env dans .gitignore. Fichier .env.example sur GitHub.
Chargement via python-dotenv avec chemin absolu.

## A09 : Logging insuffisant

Python logging configuré dans FastAPI. Fichier logs/api.log
créé automatiquement. Chaque prédiction journalisée avec ses
paramètres. 3 seuils d'alerte WARNING automatiques.

---

## Tableau récapitulatif

| Risque OWASP | Protection | Statut |
|---|---|---|
| A01 Contrôle d'accès | Header X-API-Key obligatoire | En place |
| A03 Injection | Validation Pydantic stricte | En place |
| A05 Mauvaise configuration | Variables via .env + .gitignore | En place |
| A09 Logging insuffisant | Python logging + api.log | En place |