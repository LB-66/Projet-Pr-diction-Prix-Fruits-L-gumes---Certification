"""
API FastAPI — Prédiction des prix des fruits & légumes
"""

from fastapi import FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from fastapi import Security, status
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
import os
import logging
import psycopg2
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator

# ── Chargement des variables d'environnement ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# ── Configuration du logging structuré ──
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fruits_legumes_api")

# ── Clé API chargée depuis le fichier .env ──
API_KEY = os.getenv("API_KEY", "fruits-legumes-api-key-2026")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Fonction qui vérifie la clé API sur chaque requête protégée
async def verifier_cle_api(cle: str = Security(api_key_header)):
    if cle != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé API invalide ou absente"
        )
    return cle

# ── Création de l'application FastAPI ──
app = FastAPI(
    title="API Prédiction Prix Fruits & Légumes",
    description="""
    API REST pour prédire le prix par cup equivalent des fruits et légumes.
    
    **Modèle** : XGBoost Grid Search — R²=0.9782 — RMSE=0.0835 $/cup
    
    **Source données** : USDA ERS enrichi (météo, énergie, engrais)
    
    **Certification** : RNCP37827 DevIA Simplon 2026
    """,
    version="1.0.0"
)

# ── Prometheus — expose les métriques sur /metrics ──
Instrumentator().instrument(app).expose(app)

# ── Chargement du modèle au démarrage ──
CHEMIN_MODELE   = os.path.join(BASE_DIR, "NOTEBOOKS", "models", "xgboost_fruits_legumes.pkl")
CHEMIN_FEATURES = os.path.join(BASE_DIR, "NOTEBOOKS", "models", "features.json")

# Chargement du modèle
if os.path.exists(CHEMIN_MODELE):
    modele = joblib.load(CHEMIN_MODELE)
    print(f"Modele charge : {CHEMIN_MODELE}")
else:
    modele = None
    print(f"ATTENTION : modele non trouve — {CHEMIN_MODELE}")

# Chargement de la liste des features
if os.path.exists(CHEMIN_FEATURES):
    with open(CHEMIN_FEATURES, 'r') as f:
        config = json.load(f)
    FEATURES = config['features']
    print(f"Features chargees : {FEATURES}")
else:
    FEATURES = [
        'prix_detail', 'rendement', 'taille_cup',
        'forme_encoded', 'categorie_encoded', 'annee',
        'production_lbs', 'temp_moyenne', 'jours_gel',
        'prix_diesel', 'prix_electricite', 'urea'
    ]
    print("Features par defaut utilisees")


# ── Schéma des données d'entrée ──
class PredictionInput(BaseModel):
    prix_detail      : float = Field(..., description="Prix en rayon ($/lb)", example=1.50)
    rendement        : float = Field(..., description="Part utilisable (0 a 1)", example=0.75)
    taille_cup       : float = Field(..., description="Taille de la portion (lb)", example=0.33)
    forme_encoded    : int   = Field(..., description="Fresh=0, Canned=1, Frozen=2, Juice=3, Dried=4", example=0)
    categorie_encoded: int   = Field(..., description="fruit=1, legume=0", example=1)
    annee            : int   = Field(..., description="Annee (2013-2026)", example=2024)
    production_lbs   : float = Field(..., description="Volume production (lbs)", example=500000.0)
    temp_moyenne     : float = Field(..., description="Temperature moyenne (C)", example=15.0)
    jours_gel        : float = Field(..., description="Nombre de jours de gel", example=10.0)
    prix_diesel      : float = Field(..., description="Prix diesel ($/gallon)", example=3.50)
    prix_electricite : float = Field(..., description="Prix electricite (c/kWh)", example=12.0)
    urea             : float = Field(..., description="Prix uree ($/tonne)", example=350.0)

    class Config:
        json_schema_extra = {
            "example": {
                "prix_detail"      : 1.50,
                "rendement"        : 0.75,
                "taille_cup"       : 0.33,
                "forme_encoded"    : 0,
                "categorie_encoded": 1,
                "annee"            : 2024,
                "production_lbs"   : 500000.0,
                "temp_moyenne"     : 15.0,
                "jours_gel"        : 10.0,
                "prix_diesel"      : 3.50,
                "prix_electricite" : 12.0,
                "urea"             : 350.0
            }
        }


# ── Schéma des données de sortie ──
class PredictionOutput(BaseModel):
    prix_predit_cup : float
    unite           : str
    modele          : str
    r2_modele       : float
    rmse_modele     : float
    statut          : str


# ══════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════

@app.get("/", summary="Page d'accueil")
def accueil():
    return {
        "message" : "API Prediction Prix Fruits & Legumes",
        "version" : "1.0.0",
        "modele"  : "XGBoost",
        "statut"  : "operationnel" if modele is not None else "modele non charge",
        "endpoints": {
            "GET  /"           : "Cette page",
            "GET  /health"     : "Verification sante de l'API",
            "POST /predict"    : "Predire le prix d'un fruit ou legume",
            "GET  /features"   : "Liste des features attendues",
            "GET  /produits"   : "Liste des produits depuis PostgreSQL",
            "GET  /prix/stats" : "Prix moyen par categorie depuis PostgreSQL",
            "GET  /metrics"    : "Metriques Prometheus",
            "GET  /docs"       : "Documentation Swagger interactive"
        }
    }


@app.get("/health", summary="Verification sante")
def health_check():
    return {
        "statut"        : "ok",
        "modele_charge" : modele is not None,
        "nb_features"   : len(FEATURES),
        "version_api"   : "1.0.0"
    }


@app.get("/features", summary="Liste des features")
def get_features():
    return {
        "features"    : FEATURES,
        "nb_features" : len(FEATURES),
        "description" : {
            "prix_detail"      : "Prix en rayon ($/lb)",
            "rendement"        : "Part utilisable apres preparation (0 a 1)",
            "taille_cup"       : "Taille de la portion standard (lb)",
            "forme_encoded"    : "Fresh=0, Canned=1, Frozen=2, Juice=3, Dried=4",
            "categorie_encoded": "fruit=1, legume=0",
            "annee"            : "Annee de la donnee (2013-2026)",
            "production_lbs"   : "Volume de production par etat (lbs)",
            "temp_moyenne"     : "Temperature annuelle de la zone (C)",
            "jours_gel"        : "Nombre de jours sous 0C",
            "prix_diesel"      : "Prix du diesel ($/gallon)",
            "prix_electricite" : "Prix de l'electricite (c/kWh)",
            "urea"             : "Prix de l'uree ($/tonne)"
        }
    }


@app.post("/predict",
          response_model=PredictionOutput,
          summary="Predire le prix d'un fruit ou legume")
def predict(data: PredictionInput, cle: str = Security(verifier_cle_api)):
    if modele is None:
        raise HTTPException(
            status_code=503,
            detail="Modele non disponible"
        )

    valeurs = np.array([[
        data.prix_detail,
        data.rendement,
        data.taille_cup,
        data.forme_encoded,
        data.categorie_encoded,
        data.annee,
        data.production_lbs,
        data.temp_moyenne,
        data.jours_gel,
        data.prix_diesel,
        data.prix_electricite,
        data.urea
    ]])

    prix = float(modele.predict(valeurs)[0])

    logger.info(
        f"PREDICTION | prix_detail={data.prix_detail} | "
        f"forme={data.forme_encoded} | annee={data.annee} | "
        f"prix_predit={round(prix, 4)}"
    )

    if prix > 5.0:
        logger.warning(f"ALERTE SEUIL HAUT | prix_predit={round(prix, 4)}")
    elif prix < 0.01:
        logger.warning(f"ALERTE SEUIL BAS | prix_predit={round(prix, 4)}")

    if data.prix_diesel > 6.0:
        logger.warning(f"ALERTE DIESEL ELEVE | prix_diesel={data.prix_diesel}")

    return PredictionOutput(
        prix_predit_cup = round(prix, 4),
        unite           = "$/cup equivalent",
        modele          = "XGBoost",
        r2_modele       = 0.9782,
        rmse_modele     = 0.0835,
        statut          = "succès"
    )


# ══════════════════════════════════════
# ENDPOINTS DONNÉES — C5
# ══════════════════════════════════════

def get_conn():
    """Connexion a PostgreSQL avec encodage UTF8"""
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "fruits_legumes_db"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "admin123")
    )
    conn.set_client_encoding("UTF8")
    return conn


@app.get("/produits", summary="Liste des produits de la base")
def get_produits():
    """Retourne les 20 premiers produits depuis PostgreSQL."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT nom, categorie, forme FROM produit ORDER BY categorie, nom LIMIT 20"
        )
        resultats = cur.fetchall()
        cur.close()
        conn.close()
        return {
            "produits": [
                {"nom": r[0], "categorie": r[1], "forme": r[2]}
                for r in resultats
            ],
            "total"  : len(resultats),
            "source" : "PostgreSQL - table produit"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Base indisponible : {e}")


@app.get("/prix/stats", summary="Prix moyen par categorie")
def get_prix_stats():
    """Prix moyen par categorie depuis PostgreSQL."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT p.categorie, "
            "ROUND(AVG(pr.prix_cup)::numeric, 4) AS prix_moyen, "
            "COUNT(*) AS nb "
            "FROM prix pr "
            "JOIN produit p ON pr.id_produit = p.id_produit "
            "GROUP BY p.categorie "
            "ORDER BY prix_moyen DESC"
        )
        resultats = cur.fetchall()
        cur.close()
        conn.close()
        return {
            "stats": [
                {
                    "categorie"      : r[0],
                    "prix_moyen_cup" : float(r[1]),
                    "nb_observations": r[2]
                }
                for r in resultats
            ],
            "source": "PostgreSQL - tables prix + produit"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Base indisponible : {e}")
    