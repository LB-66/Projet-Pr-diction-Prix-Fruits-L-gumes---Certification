"""
Tests automatisés — Modèle XGBoost
Compétence C12 — Certification RNCP37827 DevIA Simplon 2026

Ce fichier teste que le modèle fonctionne correctement :
- Le fichier .pkl existe et charge sans erreur
- Les prédictions sont dans une plage réaliste
- Les features sont correctes
"""

import pytest
import joblib
import numpy as np
import json
import os

# ── Chemins vers les fichiers du modèle ──
CHEMIN_PKL      = "NOTEBOOKS/models/xgboost_fruits_legumes.pkl"
CHEMIN_FEATURES = "NOTEBOOKS/models/features.json"

# ── Données d'exemple pour les tests ──
# Ces valeurs correspondent à une pomme fraîche en 2024
EXEMPLE_VALIDE = np.array([[
    1.50,        # prix_detail
    0.75,        # rendement
    0.33,        # taille_cup
    0,           # forme_encoded (Fresh)
    1,           # categorie_encoded (fruit)
    2024,        # annee
    500000000.0, # production_lbs
    15.0,        # temp_moyenne
    10.0,        # jours_gel
    3.50,        # prix_diesel
    12.0,        # prix_electricite
    350.0        # urea
]])


# ══════════════════════════════════════
# TESTS — FICHIERS
# ══════════════════════════════════════

def test_fichier_pkl_existe():
    """Vérifie que le fichier .pkl existe bien sur le disque."""
    assert os.path.exists(CHEMIN_PKL), \
        f"Le fichier .pkl est introuvable : {CHEMIN_PKL}"


def test_fichier_features_existe():
    """Vérifie que le fichier features.json existe bien sur le disque."""
    assert os.path.exists(CHEMIN_FEATURES), \
        f"Le fichier features.json est introuvable : {CHEMIN_FEATURES}"


def test_taille_fichier_pkl():
    """Vérifie que le fichier .pkl a une taille raisonnable (> 10 Ko)."""
    taille = os.path.getsize(CHEMIN_PKL)
    assert taille > 10_000, \
        f"Le fichier .pkl semble trop petit : {taille} octets"


# ══════════════════════════════════════
# TESTS — CHARGEMENT DU MODÈLE
# ══════════════════════════════════════

@pytest.fixture
def modele():
    """Charge le modèle une seule fois pour tous les tests."""
    return joblib.load(CHEMIN_PKL)


@pytest.fixture
def features():
    """Charge la configuration des features."""
    with open(CHEMIN_FEATURES, "r", encoding="utf-8") as f:
        return json.load(f)


def test_modele_charge_sans_erreur(modele):
    """Vérifie que le modèle se charge sans lever d'exception."""
    assert modele is not None


def test_modele_a_methode_predict(modele):
    """Vérifie que le modèle possède bien une méthode predict."""
    assert hasattr(modele, "predict"), \
        "Le modèle ne possède pas de méthode predict"


def test_features_json_contient_liste(features):
    """Vérifie que features.json contient bien une liste de features."""
    assert "features" in features, \
        "La clé 'features' est absente de features.json"
    assert isinstance(features["features"], list), \
        "La clé 'features' doit être une liste"


def test_features_json_12_features(features):
    """Vérifie qu'il y a bien 12 features dans la configuration."""
    nb = len(features["features"])
    assert nb == 12, \
        f"On attend 12 features, on en trouve {nb}"


# ══════════════════════════════════════
# TESTS — PRÉDICTIONS
# ══════════════════════════════════════

def test_prediction_retourne_un_nombre(modele):
    """Vérifie que la prédiction retourne bien un nombre."""
    prediction = modele.predict(EXEMPLE_VALIDE)
    assert isinstance(prediction[0], (float, np.floating)), \
        "La prédiction doit être un nombre décimal"


def test_prediction_dans_plage_realiste(modele):
    """Vérifie que le prix prédit est dans une plage réaliste.
    Le dataset contient des prix entre 0.17$ et 4.28$/cup.
    On autorise une marge de 50% de chaque côté."""
    prediction = float(modele.predict(EXEMPLE_VALIDE)[0])
    assert 0.05 <= prediction <= 6.0, \
        f"Prix prédit hors plage réaliste : {prediction:.4f}$/cup"


def test_prediction_positive(modele):
    """Vérifie que le prix prédit est toujours positif."""
    prediction = float(modele.predict(EXEMPLE_VALIDE)[0])
    assert prediction > 0, \
        f"Le prix prédit ne peut pas être négatif : {prediction}"


def test_prediction_reproductible(modele):
    """Vérifie que deux prédictions identiques donnent le même résultat."""
    pred1 = float(modele.predict(EXEMPLE_VALIDE)[0])
    pred2 = float(modele.predict(EXEMPLE_VALIDE)[0])
    assert pred1 == pred2, \
        "La prédiction n'est pas reproductible"






def test_prediction_plusieurs_observations(modele):
    """Vérifie que le modèle prédit correctement sur plusieurs observations."""
    plusieurs = np.repeat(EXEMPLE_VALIDE, 5, axis=0)
    predictions = modele.predict(plusieurs)
    assert len(predictions) == 5, \
        "Le modèle doit retourner autant de prédictions que d'observations"
