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
from dotenv import load_dotenv

# ── Chargement des variables d'environnement ──
# On cherche le .env à la racine du projet (un niveau au-dessus de API/)
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

# ── Chargement du modèle au démarrage ──
CHEMIN_MODELE   = os.path.join(BASE_DIR, "NOTEBOOKS", "models", "xgboost_fruits_legumes.pkl")
CHEMIN_FEATURES = os.path.join(BASE_DIR, "NOTEBOOKS", "models", "features.json")

# Chargement du modèle
if os.path.exists(CHEMIN_MODELE):
    modele = joblib.load(CHEMIN_MODELE)
    print(f"Modèle chargé : {CHEMIN_MODELE}")
else:
    modele = None
    print(f"ATTENTION : modèle non trouvé — {CHEMIN_MODELE}")

# Chargement de la liste des features
if os.path.exists(CHEMIN_FEATURES):
    with open(CHEMIN_FEATURES, 'r') as f:
        config = json.load(f)
    FEATURES = config['features']
    print(f"Features chargées : {FEATURES}")
else:
    FEATURES = [
        'prix_detail', 'rendement', 'taille_cup',
        'forme_encoded', 'categorie_encoded', 'annee',
        'production_lbs', 'temp_moyenne', 'jours_gel',
        'prix_diesel', 'prix_electricite', 'urea'
    ]
    print("Features par défaut utilisées")


# ── Schéma des données d'entrée ──
class PredictionInput(BaseModel):
    """Données nécessaires pour prédire le prix d'un fruit ou légume"""

    prix_detail      : float = Field(..., description="Prix en rayon ($/lb)", example=1.50)
    rendement        : float = Field(..., description="Part utilisable (0 à 1)", example=0.75)
    taille_cup       : float = Field(..., description="Taille de la portion (lb)", example=0.33)
    forme_encoded    : int   = Field(..., description="Fresh=0, Canned=1, Frozen=2, Juice=3, Dried=4", example=0)
    categorie_encoded: int   = Field(..., description="fruit=1, legume=0", example=1)
    annee            : int   = Field(..., description="Année (2013-2026)", example=2024)
    production_lbs   : float = Field(..., description="Volume production (lbs)", example=500000.0)
    temp_moyenne     : float = Field(..., description="Température moyenne (°C)", example=15.0)
    jours_gel        : float = Field(..., description="Nombre de jours de gel", example=10.0)
    prix_diesel      : float = Field(..., description="Prix diesel ($/gallon)", example=3.50)
    prix_electricite : float = Field(..., description="Prix électricité (¢/kWh)", example=12.0)
    urea             : float = Field(..., description="Prix urée ($/tonne)", example=350.0)

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
    """Résultat de la prédiction"""
    prix_predit_cup : float
    unite           : str
    modele          : str
    r2_modele       : float
    rmse_modele     : float
    statut          : str


# ══════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════

# ── GET / — Page d'accueil ──
@app.get("/", summary="Page d'accueil")
def accueil():
    """Point d'entrée de l'API."""
    return {
        "message" : "API Prédiction Prix Fruits & Légumes",
        "version" : "1.0.0",
        "modele"  : "XGBoost",
        "statut"  : "opérationnel" if modele is not None else "modèle non chargé",
        "endpoints": {
            "GET  /"        : "Cette page",
            "GET  /health"  : "Vérification santé de l'API",
            "POST /predict" : "Prédire le prix d'un fruit ou légume",
            "GET  /features": "Liste des features attendues",
            "GET  /docs"    : "Documentation Swagger interactive"
        }
    }


# ── GET /health — Vérification santé ──
@app.get("/health", summary="Vérification santé")
def health_check():
    """Vérifie que l'API et le modèle sont opérationnels."""
    return {
        "statut"        : "ok",
        "modele_charge" : modele is not None,
        "nb_features"   : len(FEATURES),
        "version_api"   : "1.0.0"
    }


# ── GET /features — Liste des features ──
@app.get("/features", summary="Liste des features")
def get_features():
    """Retourne la liste des features attendues par le modèle."""
    return {
        "features"    : FEATURES,
        "nb_features" : len(FEATURES),
        "description" : {
            "prix_detail"      : "Prix en rayon ($/lb)",
            "rendement"        : "Part utilisable après préparation (0 à 1)",
            "taille_cup"       : "Taille de la portion standard (lb)",
            "forme_encoded"    : "Fresh=0, Canned=1, Frozen=2, Juice=3, Dried=4",
            "categorie_encoded": "fruit=1, legume=0",
            "annee"            : "Année de la donnée (2013-2026)",
            "production_lbs"   : "Volume de production par état (lbs)",
            "temp_moyenne"     : "Température annuelle de la zone (°C)",
            "jours_gel"        : "Nombre de jours sous 0°C",
            "prix_diesel"      : "Prix du diesel ($/gallon)",
            "prix_electricite" : "Prix de l'électricité (¢/kWh)",
            "urea"             : "Prix de l'urée ($/tonne)"
        }
    }


# ── POST /predict — Prédiction principale ──
@app.post("/predict",
          response_model=PredictionOutput,
          summary="Prédire le prix d'un fruit ou légume")
def predict(data: PredictionInput, cle: str = Security(verifier_cle_api)):
    """
    Prédit le prix par cup equivalent d'un fruit ou légume.
    
    **Entrée** : les 12 features du modèle XGBoost
    
    **Sortie** : le prix prédit en $/cup avec les métriques du modèle
    """

    # Vérification que le modèle est chargé
    if modele is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible — vérifiez que le fichier .pkl existe"
        )

    # Construction du tableau de features dans l'ordre exact
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

    # Prédiction avec le modèle XGBoost
    prix = float(modele.predict(valeurs)[0])

    # Logging de chaque prédiction
    logger.info(
        f"PREDICTION | prix_detail={data.prix_detail} | "
        f"forme={data.forme_encoded} | annee={data.annee} | "
        f"prix_predit={round(prix, 4)}"
    )

    # Seuils d'alerte
    if prix > 5.0:
        logger.warning(
            f"ALERTE SEUIL HAUT | prix_predit={round(prix, 4)} "
            f"superieur a 5.00$/cup | verifier les donnees d entree"
        )
    elif prix < 0.01:
        logger.warning(
            f"ALERTE SEUIL BAS | prix_predit={round(prix, 4)} "
            f"inferieur a 0.01$/cup | verifier les donnees d entree"
        )

    if data.prix_diesel > 6.0:
        logger.warning(
            f"ALERTE DIESEL ELEVE | prix_diesel={data.prix_diesel} "
            f"superieur a 6.00$/gallon | impact sur les couts de transport"
        )

    return PredictionOutput(
        prix_predit_cup = round(prix, 4),
        unite           = "$/cup equivalent",
        modele          = "XGBoost",
        r2_modele       = 0.9782,
        rmse_modele     = 0.0835,
        statut          = "succès"
    )