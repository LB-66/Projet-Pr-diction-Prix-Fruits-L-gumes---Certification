"""
API FastAPI — Prédiction des prix des fruits & légumes
Compétences : C5, C8, C9
"""

from fastapi import FastAPI, HTTPException  # Framework API
from pydantic import BaseModel, Field       # Validation des données
import joblib                               # Pour charger le modèle .pkl
import numpy as np                          # Calculs numériques
import json                                 # Pour lire le fichier features.json
import os                                   # Pour les chemins de fichiers

# ── Création de l'application FastAPI ──
# title et description apparaissent dans la doc Swagger
app = FastAPI(
    title="API Prédiction Prix Fruits & Légumes",
    description="""
    API REST pour prédire le prix par cup equivalent des fruits et légumes.
    
    **Modèle** : XGBoost — R²=0.9755 — RMSE=0.0886 $/cup
    
    **Source données** : USDA ERS enrichi (météo, énergie, engrais)
    
    **Certification** : RNCP37827 DevIA Simplon 2026 — Compétences C5, C8, C9
    """,
    version="1.0.0"
)

# ── Chargement du modèle au démarrage ──
# On charge le .pkl une seule fois au lancement — pas à chaque requête
#CHEMIN_MODELE   = "NOTEBOOKS/models/xgboost_fruits_legumes.pkl"
#CHEMIN_FEATURES = "NOTEBOOKS/models/features.json"

import os

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    # Features par défaut si le fichier n'existe pas
    FEATURES = [
        'prix_detail', 'rendement', 'taille_cup',
        'forme_encoded', 'categorie_encoded', 'annee',
        'production_lbs', 'temp_moyenne', 'jours_gel',
        'prix_diesel', 'prix_electricite', 'urea'
    ]
    print("Features par défaut utilisées")


# ── Schéma des données d'entrée ──
# Pydantic valide automatiquement que toutes les valeurs sont du bon type
class PredictionInput(BaseModel):
    """Données nécessaires pour prédire le prix d'un fruit ou légume"""

    prix_detail     : float = Field(..., description="Prix en rayon ($/lb)", example=1.50)
    rendement       : float = Field(..., description="Part utilisable (0 à 1)", example=0.75)
    taille_cup      : float = Field(..., description="Taille de la portion (lb)", example=0.33)
    forme_encoded   : int   = Field(..., description="Fresh=0, Canned=1, Frozen=2, Juice=3, Dried=4", example=0)
    categorie_encoded: int  = Field(..., description="fruit=1, legume=0", example=1)
    annee           : int   = Field(..., description="Année (2013-2026)", example=2024)
    production_lbs  : float = Field(..., description="Volume production (lbs)", example=500000.0)
    temp_moyenne    : float = Field(..., description="Température moyenne (°C)", example=15.0)
    jours_gel       : float = Field(..., description="Nombre de jours de gel", example=10.0)
    prix_diesel     : float = Field(..., description="Prix diesel ($/gallon)", example=3.50)
    prix_electricite: float = Field(..., description="Prix électricité (¢/kWh)", example=12.0)
    urea            : float = Field(..., description="Prix urée ($/tonne)", example=350.0)

    class Config:
        # Exemple affiché dans la documentation Swagger
        json_schema_extra = {
            "example": {
                "prix_detail"     : 1.50,
                "rendement"       : 0.75,
                "taille_cup"      : 0.33,
                "forme_encoded"   : 0,
                "categorie_encoded": 1,
                "annee"           : 2024,
                "production_lbs"  : 500000.0,
                "temp_moyenne"    : 15.0,
                "jours_gel"       : 10.0,
                "prix_diesel"     : 3.50,
                "prix_electricite": 12.0,
                "urea"            : 350.0
            }
        }


# ── Schéma des données de sortie ──
class PredictionOutput(BaseModel):
    """Résultat de la prédiction"""
    prix_predit_cup : float  # Prix prédit par cup equivalent
    unite           : str    # Unité de la prédiction
    modele          : str    # Modèle utilisé
    r2_modele       : float  # Performance du modèle
    rmse_modele     : float  # Erreur moyenne du modèle
    statut          : str    # Statut de la prédiction


# ══════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════

# ── GET / — Page d'accueil ──
@app.get("/", summary="Page d'accueil")
def accueil():
    """
    Point d'entrée de l'API.
    Retourne les informations générales et les endpoints disponibles.
    """
    return {
        "message"    : "API Prédiction Prix Fruits & Légumes",
        "version"    : "1.0.0",
        "modele"     : "XGBoost",
        "statut"     : "opérationnel" if modele is not None else "modèle non chargé",
        "endpoints"  : {
            "GET  /"          : "Cette page",
            "GET  /health"    : "Vérification santé de l'API",
            "POST /predict"   : "Prédire le prix d'un fruit ou légume",
            "GET  /features"  : "Liste des features attendues",
            "GET  /docs"      : "Documentation Swagger interactive"
        }
    }


# ── GET /health — Vérification santé ──
@app.get("/health", summary="Vérification santé")
def health_check():
    """
    Vérifie que l'API et le modèle sont opérationnels.
    Utilisé par Docker et les systèmes de monitoring.
    """
    return {
        "statut"         : "ok",
        "modele_charge"  : modele is not None,
        "nb_features"    : len(FEATURES),
        "version_api"    : "1.0.0"
    }


# ── GET /features — Liste des features ──
@app.get("/features", summary="Liste des features")
def get_features():
    """
    Retourne la liste des features attendues par le modèle
    dans l'ordre exact à respecter.
    """
    return {
        "features"   : FEATURES,
        "nb_features": len(FEATURES),
        "description": {
            "prix_detail"     : "Prix en rayon ($/lb)",
            "rendement"       : "Part utilisable après préparation (0 à 1)",
            "taille_cup"      : "Taille de la portion standard (lb)",
            "forme_encoded"   : "Fresh=0, Canned=1, Frozen=2, Juice=3, Dried=4",
            "categorie_encoded": "fruit=1, legume=0",
            "annee"           : "Année de la donnée (2013-2026)",
            "production_lbs"  : "Volume de production par état (lbs)",
            "temp_moyenne"    : "Température annuelle de la zone (°C)",
            "jours_gel"       : "Nombre de jours sous 0°C",
            "prix_diesel"     : "Prix du diesel ($/gallon)",
            "prix_electricite": "Prix de l'électricité (¢/kWh)",
            "urea"            : "Prix de l'urée ($/tonne)"
        }
    }


# ── POST /predict — Prédiction principale ──
@app.post("/predict",
          response_model=PredictionOutput,
          summary="Prédire le prix d'un fruit ou légume")
def predict(data: PredictionInput):
    """
    Prédit le prix par cup equivalent d'un fruit ou légume.
    
    **Entrée** : les 12 features du modèle XGBoost
    
    **Sortie** : le prix prédit en $/cup avec les métriques du modèle
    
    **Exemple** : une pomme fraîche avec prix_detail=1.50$ → ~0.75$/cup
    """

    # Vérification que le modèle est chargé
    if modele is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible — vérifiez que le fichier .pkl existe"
        )

    # Construction du tableau de features dans l'ordre exact
    # L'ordre doit correspondre exactement à celui utilisé lors de l'entraînement
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
    prix_predit = float(modele.predict(valeurs)[0])

    # On arrondit à 4 décimales pour la lisibilité
    prix_predit = round(prix_predit, 4)

    # Retour de la prédiction avec les métadonnées
    return PredictionOutput(
        prix_predit_cup = prix_predit,
        unite           = "$/cup equivalent",
        modele          = "XGBoost",
        r2_modele       = 0.9755,
        rmse_modele     = 0.0886,
        statut          = "succès"
    )