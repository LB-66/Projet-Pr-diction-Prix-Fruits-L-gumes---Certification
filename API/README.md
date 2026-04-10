# API FastAPI — Prédiction Prix Fruits & Légumes

API REST pour prédire le prix par cup equivalent des fruits et légumes.

**Modèle** : XGBoost — R²=0.9755 — RMSE=0.0886 $/cup  
**Certification** : RNCP37827 DevIA Simplon 2026 — Compétences C5, C8, C9

---

## Installation

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'API
uvicorn main:app --reload --port 8000
```

L'API est accessible sur : http://localhost:8000  
La documentation Swagger est sur : http://localhost:8000/docs

---

## Endpoints disponibles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | / | Page d'accueil |
| GET | /health | Vérification santé |
| GET | /features | Liste des features |
| POST | /predict | Prédire un prix |
| GET | /docs | Documentation Swagger |

---

## Tester avec Postman

### Étape 1 — Créer une requête POST

- Méthode : **POST**
- URL : `http://localhost:8000/predict`
- Headers : `Content-Type: application/json`

### Étape 2 — Corps de la requête (Body → raw → JSON)

```json
{
    "prix_detail": 1.50,
    "rendement": 0.75,
    "taille_cup": 0.33,
    "forme_encoded": 0,
    "categorie_encoded": 1,
    "annee": 2024,
    "production_lbs": 500000.0,
    "temp_moyenne": 15.0,
    "jours_gel": 10.0,
    "prix_diesel": 3.50,
    "prix_electricite": 12.0,
    "urea": 350.0
}
```

### Étape 3 — Réponse attendue

```json
{
    "prix_predit_cup": 0.8734,
    "unite": "$/cup equivalent",
    "modele": "XGBoost",
    "r2_modele": 0.9755,
    "rmse_modele": 0.0886,
    "statut": "succès"
}
```

---

## Description des features

| Feature | Type | Description |
|---------|------|-------------|
| prix_detail | float | Prix en rayon ($/lb) |
| rendement | float | Part utilisable après préparation (0 à 1) |
| taille_cup | float | Taille de la portion standard (lb) |
| forme_encoded | int | Fresh=0, Canned=1, Frozen=2, Juice=3, Dried=4 |
| categorie_encoded | int | fruit=1, legume=0 |
| annee | int | Année (2013-2026) |
| production_lbs | float | Volume de production par état (lbs) |
| temp_moyenne | float | Température annuelle de la zone (°C) |
| jours_gel | float | Nombre de jours sous 0°C |
| prix_diesel | float | Prix du diesel ($/gallon) |
| prix_electricite | float | Prix de l'électricité (¢/kWh) |
| urea | float | Prix de l'urée ($/tonne) |