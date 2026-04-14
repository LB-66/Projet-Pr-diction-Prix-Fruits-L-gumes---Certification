"""
Tests automatisés — API FastAPI
Compétence C12 — Certification RNCP37827 DevIA Simplon 2026

Ce fichier teste que l'API fonctionne correctement :
- Les endpoints répondent avec le bon statut HTTP
- Les données retournées ont le bon format
- Les erreurs sont bien gérées
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# ── On ajoute le dossier API au chemin Python ──
# Pour que Python trouve le fichier main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "API"))

# ── Import de l'application FastAPI ──
try:
    from main import app
    CLIENT = TestClient(app)
    API_DISPONIBLE = True
except ImportError:
    API_DISPONIBLE = False

# ── Décorateur pour ignorer les tests si l'API n'est pas disponible ──
skip_si_api_absente = pytest.mark.skipif(
    not API_DISPONIBLE,
    reason="API FastAPI non disponible — vérifier le dossier API/"
)

# ── Données d'exemple pour les tests ──
DONNEES_VALIDES = {
    "prix_detail"      : 1.50,
    "rendement"        : 0.75,
    "taille_cup"       : 0.33,
    "forme_encoded"    : 0,
    "categorie_encoded": 1,
    "annee"            : 2024,
    "production_lbs"   : 500000000.0,
    "temp_moyenne"     : 15.0,
    "jours_gel"        : 10.0,
    "prix_diesel"      : 3.50,
    "prix_electricite" : 12.0,
    "urea"             : 350.0
}


# ══════════════════════════════════════
# TESTS — ENDPOINT GET /
# ══════════════════════════════════════

@skip_si_api_absente
def test_accueil_statut_200():
    """Vérifie que la page d'accueil répond avec le statut 200 OK."""
    reponse = CLIENT.get("/")
    assert reponse.status_code == 200, \
        f"L'accueil devrait retourner 200, reçu : {reponse.status_code}"


@skip_si_api_absente
def test_accueil_contient_message():
    """Vérifie que la réponse contient un champ message."""
    reponse = CLIENT.get("/")
    data    = reponse.json()
    assert "message" in data, \
        "La réponse de l'accueil doit contenir un champ 'message'"


@skip_si_api_absente
def test_accueil_contient_statut():
    """Vérifie que la réponse contient un champ statut."""
    reponse = CLIENT.get("/")
    data    = reponse.json()
    assert "statut" in data, \
        "La réponse de l'accueil doit contenir un champ 'statut'"


# ══════════════════════════════════════
# TESTS — ENDPOINT GET /health
# ══════════════════════════════════════

@skip_si_api_absente
def test_health_statut_200():
    """Vérifie que le health check répond avec le statut 200 OK."""
    reponse = CLIENT.get("/health")
    assert reponse.status_code == 200, \
        f"Le health check devrait retourner 200, reçu : {reponse.status_code}"


@skip_si_api_absente
def test_health_contient_statut_ok():
    """Vérifie que le health check retourne statut ok."""
    reponse = CLIENT.get("/health")
    data    = reponse.json()
    assert data.get("statut") == "ok", \
        f"Le health check devrait retourner 'ok', reçu : {data.get('statut')}"


@skip_si_api_absente
def test_health_modele_charge():
    """Vérifie que le health check confirme que le modèle est chargé."""
    reponse = CLIENT.get("/health")
    data    = reponse.json()
    assert "modele_charge" in data, \
        "Le health check doit indiquer si le modèle est chargé"


# ══════════════════════════════════════
# TESTS — ENDPOINT GET /features
# ══════════════════════════════════════

@skip_si_api_absente
def test_features_statut_200():
    """Vérifie que l'endpoint features répond avec le statut 200 OK."""
    reponse = CLIENT.get("/features")
    assert reponse.status_code == 200


@skip_si_api_absente
def test_features_contient_liste():
    """Vérifie que la réponse contient une liste de features."""
    reponse = CLIENT.get("/features")
    data    = reponse.json()
    assert "features" in data, \
        "La réponse doit contenir une clé 'features'"
    assert isinstance(data["features"], list), \
        "La clé 'features' doit être une liste"


@skip_si_api_absente
def test_features_contient_12_elements():
    """Vérifie que la liste contient bien 12 features."""
    reponse = CLIENT.get("/features")
    data    = reponse.json()
    nb      = len(data.get("features", []))
    assert nb == 12, \
        f"On attend 12 features, on en trouve {nb}"


@skip_si_api_absente
def test_features_contient_prix_detail():
    """Vérifie que prix_detail est dans la liste des features."""
    reponse = CLIENT.get("/features")
    data    = reponse.json()
    assert "prix_detail" in data.get("features", []), \
        "prix_detail doit être dans la liste des features"


# ══════════════════════════════════════
# TESTS — ENDPOINT POST /predict
# ══════════════════════════════════════

@skip_si_api_absente
def test_predict_statut_200():
    """Vérifie que POST /predict répond avec le statut 200 OK."""
    reponse = CLIENT.post("/predict", json=DONNEES_VALIDES)
    assert reponse.status_code == 200, \
        f"POST /predict devrait retourner 200, reçu : {reponse.status_code}"


@skip_si_api_absente
def test_predict_contient_prix():
    """Vérifie que la réponse contient un champ prix_predit_cup."""
    reponse = CLIENT.post("/predict", json=DONNEES_VALIDES)
    data    = reponse.json()
    assert "prix_predit_cup" in data, \
        "La réponse doit contenir un champ 'prix_predit_cup'"


@skip_si_api_absente
def test_predict_prix_est_un_nombre():
    """Vérifie que le prix prédit est bien un nombre."""
    reponse = CLIENT.post("/predict", json=DONNEES_VALIDES)
    data    = reponse.json()
    prix    = data.get("prix_predit_cup")
    assert isinstance(prix, (int, float)), \
        f"Le prix prédit doit être un nombre, reçu : {type(prix)}"


@skip_si_api_absente
def test_predict_prix_positif():
    """Vérifie que le prix prédit est toujours positif."""
    reponse = CLIENT.post("/predict", json=DONNEES_VALIDES)
    prix    = reponse.json().get("prix_predit_cup", -1)
    assert prix > 0, \
        f"Le prix prédit doit être positif, reçu : {prix}"


@skip_si_api_absente
def test_predict_prix_dans_plage_realiste():
    """Vérifie que le prix prédit est dans une plage réaliste."""
    reponse = CLIENT.post("/predict", json=DONNEES_VALIDES)
    prix    = reponse.json().get("prix_predit_cup", -1)
    assert 0.05 <= prix <= 6.0, \
        f"Prix prédit hors plage réaliste : {prix:.4f}$/cup"


@skip_si_api_absente
def test_predict_contient_statut_succes():
    """Vérifie que la réponse contient un statut succès."""
    reponse = CLIENT.post("/predict", json=DONNEES_VALIDES)
    data    = reponse.json()
    assert data.get("statut") == "succès", \
        f"Le statut devrait être 'succès', reçu : {data.get('statut')}"


@skip_si_api_absente
def test_predict_contient_nom_modele():
    """Vérifie que la réponse contient le nom du modèle."""
    reponse = CLIENT.post("/predict", json=DONNEES_VALIDES)
    data    = reponse.json()
    assert "modele" in data, \
        "La réponse doit indiquer le nom du modèle utilisé"


@skip_si_api_absente
def test_predict_donnees_manquantes_retourne_422():
    """Vérifie que l'API retourne 422 si une feature est manquante."""
    donnees_incompletes = {"prix_detail": 1.50}  # 11 features manquantes
    reponse = CLIENT.post("/predict", json=donnees_incompletes)
    assert reponse.status_code == 422, \
        f"Des données manquantes devraient retourner 422, reçu : {reponse.status_code}"


@skip_si_api_absente
def test_predict_mauvais_type_retourne_422():
    """Vérifie que l'API retourne 422 si le type est incorrect."""
    donnees_mauvais_type = DONNEES_VALIDES.copy()
    donnees_mauvais_type["prix_detail"] = "pas_un_nombre"
    reponse = CLIENT.post("/predict", json=donnees_mauvais_type)
    assert reponse.status_code == 422, \
        f"Un mauvais type devrait retourner 422, reçu : {reponse.status_code}"


@skip_si_api_absente
def test_predict_reproductible():
    """Vérifie que deux appels identiques donnent le même résultat."""
    pred1 = CLIENT.post("/predict", json=DONNEES_VALIDES).json()["prix_predit_cup"]
    pred2 = CLIENT.post("/predict", json=DONNEES_VALIDES).json()["prix_predit_cup"]
    assert pred1 == pred2, \
        "Deux appels identiques doivent retourner le même prix prédit"
