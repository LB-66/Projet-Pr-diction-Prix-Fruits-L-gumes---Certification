"""
Tests automatisés — API FastAPI Fruits & Légumes
Pytest avec couverture de code
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# On ajoute le dossier API au chemin Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'API'))

from main import app

# ── Client de test FastAPI ──
CLIENT = TestClient(app)

# ── Clé API pour les tests ──
# Même valeur par défaut que dans main.py
CLE_API = os.getenv("API_KEY", "fruits-legumes-api-key-2026")
HEADERS_VALIDES = {"X-API-Key": CLE_API}

# ── Données de test valides ──
DONNEES_VALIDES = {
    "prix_detail"      : 1.50,
    "rendement"        : 0.90,
    "taille_cup"       : 0.24,
    "forme_encoded"    : 0,
    "categorie_encoded": 1,
    "annee"            : 2024,
    "production_lbs"   : 7400000000,
    "temp_moyenne"     : 10.2,
    "jours_gel"        : 45.0,
    "prix_diesel"      : 3.50,
    "prix_electricite" : 12.0,
    "urea"             : 350.0
}


# ══════════════════════════════════════
# TESTS ENDPOINTS GET (sans clé API)
# ══════════════════════════════════════

def test_accueil_statut_200():
    """GET / doit retourner 200 OK"""
    rep = CLIENT.get("/")
    assert rep.status_code == 200, f"GET / devrait retourner 200, reçu : {rep.status_code}"

def test_accueil_contient_message():
    """GET / doit contenir un champ message"""
    rep = CLIENT.get("/")
    assert "message" in rep.json(), "La page d'accueil doit contenir un champ 'message'"

def test_accueil_contient_endpoints():
    """GET / doit lister les endpoints disponibles"""
    rep = CLIENT.get("/")
    assert "endpoints" in rep.json(), "La page d'accueil doit lister les endpoints"

def test_health_statut_200():
    """GET /health doit retourner 200 OK"""
    rep = CLIENT.get("/health")
    assert rep.status_code == 200, f"GET /health devrait retourner 200, reçu : {rep.status_code}"

def test_health_contient_statut():
    """GET /health doit contenir un champ statut"""
    rep = CLIENT.get("/health")
    assert "statut" in rep.json(), "GET /health doit contenir un champ 'statut'"

def test_health_contient_modele_charge():
    """GET /health doit indiquer si le modèle est chargé"""
    rep = CLIENT.get("/health")
    assert "modele_charge" in rep.json(), "GET /health doit contenir 'modele_charge'"

def test_features_statut_200():
    """GET /features doit retourner 200 OK"""
    rep = CLIENT.get("/features")
    assert rep.status_code == 200, f"GET /features devrait retourner 200, reçu : {rep.status_code}"

def test_features_contient_liste():
    """GET /features doit retourner une liste de features"""
    rep = CLIENT.get("/features")
    assert "features" in rep.json(), "GET /features doit contenir la liste des features"

def test_features_contient_12_features():
    """GET /features doit retourner exactement 12 features"""
    rep = CLIENT.get("/features")
    features = rep.json()["features"]
    assert len(features) == 12, f"Le modèle doit avoir 12 features, reçu : {len(features)}"

def test_features_contient_prix_detail():
    """GET /features doit contenir prix_detail"""
    rep = CLIENT.get("/features")
    features = rep.json()["features"]
    assert "prix_detail" in features, "prix_detail doit être dans la liste des features"


# ══════════════════════════════════════
# TESTS ENDPOINT POST /predict
# ══════════════════════════════════════

def test_predict_statut_200():
    """POST /predict avec données valides et clé API doit retourner 200 OK"""
    rep = CLIENT.post("/predict", json=DONNEES_VALIDES, headers=HEADERS_VALIDES)
    assert rep.status_code == 200, \
        f"POST /predict devrait retourner 200, reçu : {rep.status_code}"

def test_predict_contient_prix():
    """POST /predict doit retourner un champ prix_predit_cup"""
    rep = CLIENT.post("/predict", json=DONNEES_VALIDES, headers=HEADERS_VALIDES)
    assert "prix_predit_cup" in rep.json(), \
        "La réponse doit contenir un champ 'prix_predit_cup'"

def test_predict_prix_est_un_nombre():
    """POST /predict doit retourner un nombre pour prix_predit_cup"""
    rep = CLIENT.post("/predict", json=DONNEES_VALIDES, headers=HEADERS_VALIDES)
    prix = rep.json().get("prix_predit_cup")
    assert isinstance(prix, (int, float)), \
        f"Le prix prédit doit être un nombre, reçu : {type(prix)}"

def test_predict_prix_positif():
    """POST /predict doit retourner un prix positif"""
    rep = CLIENT.post("/predict", json=DONNEES_VALIDES, headers=HEADERS_VALIDES)
    prix = rep.json().get("prix_predit_cup", -1)
    assert prix > 0, f"Le prix prédit doit être positif, reçu : {prix}"

def test_predict_prix_dans_plage_realiste():
    """POST /predict doit retourner un prix entre 0.01 et 10 $/cup"""
    rep = CLIENT.post("/predict", json=DONNEES_VALIDES, headers=HEADERS_VALIDES)
    prix = rep.json().get("prix_predit_cup", -1.0)
    assert 0.01 <= prix <= 10.0, \
        f"Prix prédit hors plage réaliste : {prix:.4f}$/cup"

def test_predict_contient_statut_succes():
    """POST /predict doit retourner statut succès"""
    rep = CLIENT.post("/predict", json=DONNEES_VALIDES, headers=HEADERS_VALIDES)
    statut = rep.json().get("statut")
    assert statut == "succès", f"Le statut devrait être 'succès', reçu : {statut}"

def test_predict_contient_nom_modele():
    """POST /predict doit indiquer le nom du modèle"""
    rep = CLIENT.post("/predict", json=DONNEES_VALIDES, headers=HEADERS_VALIDES)
    assert "modele" in rep.json(), \
        "La réponse doit indiquer le nom du modèle utilisé"

def test_predict_sans_cle_retourne_403():
    """POST /predict sans clé API doit retourner 403 Forbidden"""
    rep = CLIENT.post("/predict", json=DONNEES_VALIDES)
    assert rep.status_code == 403, \
        f"Sans clé API, /predict devrait retourner 403, reçu : {rep.status_code}"

def test_predict_mauvaise_cle_retourne_403():
    """POST /predict avec mauvaise clé API doit retourner 403"""
    rep = CLIENT.post("/predict", json=DONNEES_VALIDES,
                      headers={"X-API-Key": "mauvaise_cle"})
    assert rep.status_code == 403, \
        f"Mauvaise clé API devrait retourner 403, reçu : {rep.status_code}"

def test_predict_donnees_manquantes_retourne_422():
    """POST /predict avec données manquantes doit retourner 422"""
    donnees_incompletes = {"prix_detail": 1.50}
    rep = CLIENT.post("/predict", json=donnees_incompletes,
                      headers=HEADERS_VALIDES)
    assert rep.status_code == 422, \
        f"Des données manquantes devraient retourner 422, reçu : {rep.status_code}"

def test_predict_mauvais_type_retourne_422():
    """POST /predict avec mauvais type de données doit retourner 422"""
    donnees_invalides = {**DONNEES_VALIDES, "prix_detail": "pas_un_nombre"}
    rep = CLIENT.post("/predict", json=donnees_invalides,
                      headers=HEADERS_VALIDES)
    assert rep.status_code == 422, \
        f"Un mauvais type devrait retourner 422, reçu : {rep.status_code}"

def test_predict_reproductible():
    """POST /predict doit retourner le même prix pour les mêmes données"""
    rep1 = CLIENT.post("/predict", json=DONNEES_VALIDES, headers=HEADERS_VALIDES)
    rep2 = CLIENT.post("/predict", json=DONNEES_VALIDES, headers=HEADERS_VALIDES)
    prix1 = rep1.json()["prix_predit_cup"]
    prix2 = rep2.json()["prix_predit_cup"]
    assert prix1 == prix2, \
        f"La prédiction doit être reproductible : {prix1} != {prix2}"