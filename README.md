# Prédiction des prix des Fruits & Légumes

**Certification RNCP37827 — Développeur en Intelligence Artificielle — Simplon 2026**

Système de prédiction du prix par portion ($/cup) des fruits et légumes américains, basé sur un modèle XGBoost entraîné sur des données USDA ERS enrichies de sources économiques et climatiques.

---

## Résultats du modèle

| Métrique | Valeur |
|---|---|
| R² (test) | 0.9782 |
| RMSE (test) | 0.0835 $/cup |
| MAE (test) | 0.0451 $/cup |
| Overfitting gap | 0.019 (< seuil 0.05) |
| Cross-validation (5 folds) | 0.956 ± 0.019 |

---

## Stack technique

| Composant | Technologie |
|---|---|
| Langage | Python 3.11 |
| Machine Learning | XGBoost 2.0.3, scikit-learn |
| Explicabilité | SHAP |
| Suivi ML | MLflow |
| API REST | FastAPI 0.115.0, Uvicorn 0.30.0 |
| Validation données | Pydantic 2.7.0 |
| Interface | Streamlit |
| Base de données | PostgreSQL 16 (Docker) |
| Connexion BDD | psycopg2-binary |
| Tests | Pytest 8.1.1, pytest-cov, httpx |
| CI/CD | GitHub Actions |
| Versionnement modèle | DVC |
| Conteneurisation | Docker, Docker Compose |
| Monitoring | Prometheus, Grafana |
| Métriques API | prometheus-fastapi-instrumentator 6.1.0 |
| Secrets | python-dotenv |

---

## Structure du projet

```
.
├── API/
│   ├── main.py                    # API FastAPI — 7 endpoints REST
│   ├── requirements.txt           # Dépendances production
│   ├── Dockerfile                 # Image Docker API
│   └── logs/
│       └── api.log                # Journalisation des prédictions
├── NOTEBOOKS/
│   ├── 01_ETL_FruitsLegumes.ipynb      # Collecte et chargement USDA ERS
│   ├── 02_Enrichissement.ipynb         # Jointures multi-sources
│   ├── 03_Visualisation_EDA.ipynb      # Analyse exploratoire
│   ├── 04_Modele_ML.ipynb              # Benchmark + Grid Search XGBoost
│   ├── 05_Export_SHAP.ipynb            # Explicabilité SHAP
│   ├── 06_MLflow_Monitoring.ipynb      # Suivi MLflow
│   ├── 07_Import_PostgreSQL.py         # Import CSV → PostgreSQL
│   └── models/
│       ├── xgboost_fruits_legumes.pkl  # Modèle entraîné (DVC)
│       ├── features.json               # Liste des 12 features
│       ├── shap_summary_plot.png
│       ├── shap_bar_plot.png
│       └── shap_waterfall.png
├── DATA/
│   ├── ers_toutes_annees.csv           # Dataset brut USDA ERS
│   ├── CLEAN/
│   │   └── fruits_legumes_enrichi.csv  # Dataset final (DVC)
├── DOCS/
│   ├── CDC_Prédiction_Prix_Fruits_Légumes.pdf
│   └── Architecture N-tiers.drawio.png
├── tests/
│   ├── test_modele.py             # 12 tests sur le modèle .pkl
│   └── test_api.py                # 22 tests sur les endpoints FastAPI
├── monitoring/
│   ├── docker-compose.yml         # Prometheus + Grafana
│   └── prometheus.yml             # Configuration scraping FastAPI
├── streamlit_app.py               # Dashboard 4 pages
├── docker-compose.yml             # PostgreSQL
├── docker-compose.staging.yml     # Staging complet
├── Dockerfile.api.staging         # Image FastAPI staging
├── Dockerfile.streamlit.staging   # Image Streamlit staging
├── requirements.txt               # Dépendances projet complet
├── env.example                    # Template variables d'environnement
├── SECURITE.md                    # Documentation OWASP
├── ACCESSIBILITE.md               # Documentation WCAG 2.1 AA
└── INCIDENT_REPORT.md             # Rapport d'incident résolu
```

---

## Installation et lancement

### Prérequis

- Python 3.11
- Docker Desktop
- Git

### 1. Cloner le projet

```bash
git clone https://github.com/LB-66/Projet-Pr-diction-Prix-Fruits-L-gumes---Certification.git
cd Projet-Pr-diction-Prix-Fruits-L-gumes---Certification
```

### 2. Configurer les variables d'environnement

```bash
cp env.example .env
# Editer .env avec vos identifiants PostgreSQL et clé API
```

Contenu du fichier `.env` à renseigner :

```
POSTGRES_USER=votre_utilisateur
POSTGRES_PASSWORD=votre_mot_de_passe
POSTGRES_DB=fruits_legumes_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
API_KEY=votre_cle_api
APP_ENV=staging
APP_PORT=8000
STREAMLIT_PORT=8501
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_PORT=5000
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer PostgreSQL via Docker

```bash
docker compose up -d postgres
```

Vérifier que le conteneur tourne :

```bash
docker ps
# → fruits_legumes_postgres doit apparaître
```

### 5. Importer les données dans PostgreSQL

```bash
python NOTEBOOKS/07_Import_PostgreSQL.py
# → 710 lignes importées dans 5 tables
```

### 6. Lancer l'API FastAPI

```bash
uvicorn API.main:app --reload
# → http://localhost:8000
# → Documentation Swagger : http://localhost:8000/docs
```

### 7. Lancer le dashboard Streamlit

Dans un second terminal :

```bash
streamlit run streamlit_app.py
# → http://localhost:8501
```

---

## Endpoints API

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| GET | / | Page d'accueil, statut et liste des endpoints | Non |
| GET | /health | Vérification santé — modèle chargé en mémoire | Non |
| GET | /features | Liste des 12 features attendues par le modèle | Non |
| GET | /metrics | Métriques Prometheus (HTTP count, latence) | Non |
| GET | /docs | Documentation Swagger interactive | Non |
| GET | /produits | 20 premiers produits depuis PostgreSQL | Non |
| GET | /prix/stats | Prix moyen par catégorie depuis PostgreSQL | Non |
| POST | /predict | Prédiction du prix $/cup à partir de 12 features | Oui (X-API-Key) |

### Exemple d'appel POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: votre_cle_api" \
  -d '{
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
  }'
```

Réponse :

```json
{
  "prix_predit_cup": 0.7035,
  "unite": "$/cup equivalent",
  "modele": "XGBoost",
  "r2_modele": 0.9782,
  "rmse_modele": 0.0835,
  "statut": "succes"
}
```

---

## Features du modèle

Les 12 features utilisées par le modèle XGBoost :

| Feature | Description | Unité |
|---|---|---|
| prix_detail | Prix en rayon | $/lb |
| rendement | Part comestible après préparation | 0 à 1 |
| taille_cup | Poids d'une portion standard | lb/cup |
| forme_encoded | Forme de commercialisation (Fresh=0, Canned=1, Frozen=2, Juice=3, Dried=4) | entier |
| categorie_encoded | Catégorie (fruit=1, légume=0) | entier |
| annee | Année de la donnée | 2013-2026 |
| production_lbs | Volume de production par état | livres |
| temp_moyenne | Température annuelle de la zone productrice | °C |
| jours_gel | Nombre de jours sous 0°C | jours |
| prix_diesel | Prix du diesel | $/gallon |
| prix_electricite | Prix de l'électricité | ¢/kWh |
| urea | Prix de l'urée (engrais azoté) | $/tonne |

Variable cible : `prix_cup` — prix d'une portion de 240 ml en dollars.

---

## Tests

```bash
# Lancer tous les tests avec rapport de couverture
pytest tests/ -v --cov=API --cov-report=term-missing
```

Résultats attendus :

```
34 tests PASSED (100%) — 4.72 secondes
API/main.py couverture : 90%
```

Les tests couvrent :
- `test_modele.py` (12 tests) : chargement du .pkl, prédictions positives, features.json
- `test_api.py` (22 tests) : tous les endpoints, authentification X-API-Key, codes d'erreur 403/422

---

## Base de données PostgreSQL

5 tables relationnelles (modèle Merise) :

| Table | Lignes | Description |
|---|---|---|
| PRODUIT | 210 | Catalogue des 132 produits × 5 formes |
| PRIX | 4 779 | Historique des prix par produit et année |
| ETAT_PRODUCTEUR | 6 | États agricoles américains avec coordonnées GPS |
| PRODUIT_ETAT | variable | Volumes de production par état et par année |
| CONTEXTE_ANNUEL | 34 | Météo, diesel, électricité et engrais par année |

---

## Monitoring

### MLflow

```bash
mlflow ui --port 5000
# → http://localhost:5000
```

13 runs enregistrés. Seuils d'alerte configurés :
- RMSE > 0.15 $/cup : avertissement
- RMSE > 0.25 $/cup : alerte critique — ré-entraînement
- R² < 0.85 : dégradation significative

### Prometheus + Grafana

```bash
cd monitoring
docker compose up -d
# Prometheus : http://localhost:9090
# Grafana    : http://localhost:3000 (admin/admin)
```

Prometheus scrape l'endpoint `/metrics` de FastAPI toutes les 15 secondes.

---

## CI/CD — GitHub Actions

Le pipeline se déclenche à chaque `push` sur `main` :

1. **Job CI (tests)** : installation Python 3.11, dépendances, modèle factice léger, 34 tests Pytest
2. **Job CD (staging)** : se déclenche uniquement si les tests passent, build des images Docker `fruits_api_staging` et `fruits_streamlit_staging`

---

## Déploiement staging

```bash
docker compose -f docker-compose.staging.yml up --build
# FastAPI   : http://localhost:8000
# Streamlit : http://localhost:8501
```

---

## Sécurité

Conformité OWASP Top 10 — voir [SECURITE.md](SECURITE.md) :

- **A01** : Header X-API-Key obligatoire sur POST /predict — 403 sans clé valide
- **A03** : Validation stricte des 12 types via Pydantic — 422 immédiat si type incorrect
- **A05** : Credentials dans `.env` — fichier dans `.gitignore`, `.env.example` sur GitHub
- **A09** : Logging Python structuré — fichier `logs/api.log` — 3 seuils WARNING automatiques

---

## Accessibilité

Conformité WCAG 2.1 niveau AA — voir [ACCESSIBILITE.md](ACCESSIBILITE.md) :

- Contraste texte principal : 18.1:1 (minimum requis : 4.5:1)
- Navigation clavier : Tab + Entrée sur toutes les pages
- Labels explicites sur tous les sélecteurs
- Messages d'erreur explicites quand l'API est indisponible
- Score Lighthouse : 94/100

---

## Sources des données

| Source | Type | Données |
|---|---|---|
| USDA ERS | CSV | Prix de 132 produits sur 5 années |
| USDA NASS | API REST | Volume de production agricole par état |
| Open-Meteo | API REST | Météo par zone productrice |
| EIA | API REST | Prix diesel et électricité par région |
| World Bank | CSV | Prix des engrais mondiaux (urée, DAP, MOP) |
| BLS | Scraping | Prix de référence retail |

---

## Rapport d'incident

Un incident a été documenté et résolu — voir [INCIDENT_REPORT.md](INCIDENT_REPORT.md) :

- **Date** : 14/05/2026
- **Sévérité** : Critique
- **Cause** : Fichier .pkl renommé accidentellement
- **Résolution** : Restauration du fichier + ajout d'un gestionnaire d'erreur explicite dans FastAPI
- **Durée de résolution** : 10 minutes

---

## Auteur

LB — Candidate RNCP37827 Développeur en Intelligence Artificielle  
Simplon 2026
