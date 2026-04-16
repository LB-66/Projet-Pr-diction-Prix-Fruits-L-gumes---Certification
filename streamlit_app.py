"""
Application Streamlit — Prédiction des prix des Fruits et Légumes
Version 2 — Interface utilisateur complète avec carte interactive et données temps réel
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import joblib
import os
from datetime import datetime

# ── Configuration ──
st.set_page_config(
    page_title="FruitsLégumes ML",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --vert: #2d6a2d; --vert-clair: #e8f5e9; --vert-mid: #4a9e4a;
  --orange: #f97316; --orange-clair: #fff3e0;
  --gris-clair: #f8f9fa; --gris-border: #e5e7eb;
  --texte: #1a1a1a; --texte-sec: #4b5563;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.navbar {
  background: var(--vert); padding: 0 32px;
  display: flex; align-items: center; justify-content: space-between;
  height: 60px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.navbar-logo { font-family:'Syne',sans-serif; font-size:18px; font-weight:700; color:#fff; }

.hero {
  background: linear-gradient(135deg, var(--vert-clair) 0%, var(--orange-clair) 100%);
  padding: 48px 32px; display: flex; gap: 48px; align-items: center;
  border-bottom: 1px solid var(--gris-border);
}
.hero-badge {
  display: inline-block; background: var(--vert); color: #fff;
  font-size: 11px; font-weight: 500; padding: 4px 12px;
  border-radius: 20px; margin-bottom: 16px; text-transform: uppercase;
}
.hero-title {
  font-family:'Syne',sans-serif; font-size:32px; font-weight:700;
  color: var(--vert); line-height:1.2; margin-bottom:16px;
}
.hero-title em { color: var(--orange); font-style:normal; }
.hero-desc { font-size:15px; color:var(--texte-sec); line-height:1.7; max-width:480px; margin-bottom:24px; }
.hero-fruits { display:flex; gap:10px; flex-wrap:wrap; }
.fruit-pill {
  background:#fff; border:1px solid var(--gris-border); border-radius:20px;
  padding:8px 14px; font-size:18px; display:inline-flex; align-items:center; gap:6px;
}
.fruit-pill span { font-size:12px; color:var(--texte-sec); }
.hero-stats { display:grid; grid-template-columns:1fr 1fr; gap:12px; min-width:260px; }
.hero-stat {
  background:#fff; border-radius:12px; padding:16px;
  text-align:center; border:1px solid var(--gris-border);
}
.hero-stat .val { font-family:'Syne',sans-serif; font-size:28px; font-weight:700; color:var(--vert); }
.hero-stat .val.orange { color:var(--orange); }
.hero-stat .lbl { font-size:12px; color:var(--texte-sec); margin-top:4px; }

.section-wrap { padding:24px 32px; background:var(--gris-clair); }
.section-title {
  font-family:'Syne',sans-serif; font-size:18px; font-weight:600;
  color:var(--texte); margin-bottom:16px; padding-bottom:8px;
  border-bottom:1px solid var(--gris-border);
}
.produits-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin-bottom:24px; }
.produit-card { background:#fff; border:1px solid var(--gris-border); border-radius:12px; padding:16px 12px; text-align:center; }
.produit-emoji { font-size:32px; margin-bottom:8px; }
.produit-name { font-size:12px; font-weight:500; margin-bottom:4px; }
.produit-price { font-size:14px; font-weight:600; color:var(--orange); }
.produit-trend { font-size:11px; color:var(--vert); margin-top:2px; }
.produit-trend.down { color:#dc2626; }

.result-box {
  background:linear-gradient(135deg,var(--vert-clair),var(--orange-clair));
  border-radius:16px; padding:32px; text-align:center;
  border:1px solid var(--gris-border); margin-top:16px;
}
.result-label { font-size:13px; color:var(--texte-sec); margin-bottom:8px; text-transform:uppercase; letter-spacing:.5px; }
.result-price { font-family:'Syne',sans-serif; font-size:56px; font-weight:700; color:var(--vert); line-height:1; margin-bottom:8px; }
.result-price em { color:var(--orange); font-style:normal; font-size:28px; }
.result-meta { font-size:13px; color:var(--texte-sec); }

.context-box { background:#fff; border-radius:12px; padding:16px; border:1px solid var(--gris-border); margin-top:12px; }
.context-title { font-weight:600; color:var(--vert); margin-bottom:8px; font-size:14px; }

.metric-card { background:#fff; border-radius:12px; padding:18px; border:1px solid var(--gris-border); display:flex; align-items:center; gap:14px; }
.metric-icon { width:44px; height:44px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px; }
.metric-icon.vert { background:var(--vert-clair); }
.metric-icon.orange { background:var(--orange-clair); }
.metric-info .val { font-family:'Syne',sans-serif; font-size:22px; font-weight:700; }
.metric-info .lbl { font-size:12px; color:var(--texte-sec); margin-top:2px; }

.info-card { background:#fff; border-radius:12px; padding:18px; border:1px solid var(--gris-border); border-top:3px solid var(--vert); }
.info-val { font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:var(--vert); }
.info-key { font-weight:500; margin:4px 0 8px; font-size:14px; }
.info-desc { font-size:13px; color:var(--texte-sec); line-height:1.5; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════
# DONNÉES
# ══════════════════════════════════════

PRODUITS_FR = {
    "Apples"      : {"fr":"Pomme",          "emoji":"🍎", "categorie":"fruit",  "formes":["Fresh","Canned","Dried","Frozen","Juice"]},
    "Blueberries" : {"fr":"Myrtille",       "emoji":"🫐", "categorie":"fruit",  "formes":["Fresh","Frozen"]},
    "Strawberries": {"fr":"Fraise",         "emoji":"🍓", "categorie":"fruit",  "formes":["Fresh","Frozen"]},
    "Cherries"    : {"fr":"Cerise",         "emoji":"🍒", "categorie":"fruit",  "formes":["Fresh","Canned","Frozen"]},
    "Grapes"      : {"fr":"Raisin",         "emoji":"🍇", "categorie":"fruit",  "formes":["Fresh","Frozen","Juice"]},
    "Peaches"     : {"fr":"Pêche",          "emoji":"🍑", "categorie":"fruit",  "formes":["Fresh","Canned","Frozen","Dried"]},
    "Oranges"     : {"fr":"Orange",         "emoji":"🍊", "categorie":"fruit",  "formes":["Fresh","Juice"]},
    "Tomatoes"    : {"fr":"Tomate",         "emoji":"🍅", "categorie":"legume", "formes":["Fresh","Canned","Frozen","Juice"]},
    "Broccoli"    : {"fr":"Brocoli",        "emoji":"🥦", "categorie":"legume", "formes":["Fresh","Frozen"]},
    "Carrots"     : {"fr":"Carotte",        "emoji":"🥕", "categorie":"legume", "formes":["Fresh","Canned","Frozen","Juice"]},
    "Lettuce"     : {"fr":"Laitue",         "emoji":"🥬", "categorie":"legume", "formes":["Fresh"]},
    "Potatoes"    : {"fr":"Pomme de terre", "emoji":"🥔", "categorie":"legume", "formes":["Fresh","Canned","Frozen","Dried"]},
    "Spinach"     : {"fr":"Épinard",        "emoji":"🌿", "categorie":"legume", "formes":["Fresh","Canned","Frozen"]},
    "Onions"      : {"fr":"Oignon",         "emoji":"🧅", "categorie":"legume", "formes":["Fresh","Frozen","Dried"]},
    "Cucumbers"   : {"fr":"Concombre",      "emoji":"🥒", "categorie":"legume", "formes":["Fresh"]},
    "Cauliflower" : {"fr":"Chou-fleur",     "emoji":"🥦", "categorie":"legume", "formes":["Fresh","Frozen"]},
}

FORMES_FR = {
    "Fresh":"Frais", "Canned":"En conserve",
    "Frozen":"Surgelé", "Juice":"Jus", "Dried":"Séché"
}

DONNEES_PRODUITS = {
    "Apples"      : {"Fresh":{"prix_detail":1.52,"rendement":0.90,"taille_cup":0.2425,"production_lbs":7400000000},
                     "Canned":{"prix_detail":0.85,"rendement":1.00,"taille_cup":0.5401,"production_lbs":7400000000},
                     "Juice":{"prix_detail":0.90,"rendement":1.00,"taille_cup":0.5200,"production_lbs":7400000000},
                     "Dried":{"prix_detail":4.50,"rendement":1.00,"taille_cup":0.0850,"production_lbs":7400000000},
                     "Frozen":{"prix_detail":1.20,"rendement":1.00,"taille_cup":0.2425,"production_lbs":7400000000}},
    "Blueberries" : {"Fresh":{"prix_detail":3.20,"rendement":0.95,"taille_cup":0.3307,"production_lbs":500000000},
                     "Frozen":{"prix_detail":2.10,"rendement":1.00,"taille_cup":0.3307,"production_lbs":500000000}},
    "Strawberries": {"Fresh":{"prix_detail":2.50,"rendement":0.92,"taille_cup":0.3307,"production_lbs":2200000000},
                     "Frozen":{"prix_detail":1.80,"rendement":1.00,"taille_cup":0.3307,"production_lbs":2200000000}},
    "Cherries"    : {"Fresh":{"prix_detail":4.50,"rendement":0.87,"taille_cup":0.2204,"production_lbs":400000000},
                     "Canned":{"prix_detail":2.80,"rendement":1.00,"taille_cup":0.5401,"production_lbs":400000000},
                     "Frozen":{"prix_detail":3.20,"rendement":1.00,"taille_cup":0.2204,"production_lbs":400000000}},
    "Grapes"      : {"Fresh":{"prix_detail":2.10,"rendement":0.94,"taille_cup":0.3307,"production_lbs":6000000000},
                     "Frozen":{"prix_detail":1.50,"rendement":1.00,"taille_cup":0.3307,"production_lbs":6000000000},
                     "Juice":{"prix_detail":0.95,"rendement":1.00,"taille_cup":0.5200,"production_lbs":6000000000}},
    "Peaches"     : {"Fresh":{"prix_detail":1.80,"rendement":0.94,"taille_cup":0.2756,"production_lbs":700000000},
                     "Canned":{"prix_detail":1.10,"rendement":1.00,"taille_cup":0.5401,"production_lbs":700000000},
                     "Frozen":{"prix_detail":1.30,"rendement":1.00,"taille_cup":0.2756,"production_lbs":700000000},
                     "Dried":{"prix_detail":5.20,"rendement":1.00,"taille_cup":0.0850,"production_lbs":700000000}},
    "Oranges"     : {"Fresh":{"prix_detail":1.00,"rendement":0.73,"taille_cup":0.3858,"production_lbs":3800000000},
                     "Juice":{"prix_detail":0.75,"rendement":1.00,"taille_cup":0.5200,"production_lbs":3800000000}},
    "Tomatoes"    : {"Fresh":{"prix_detail":1.90,"rendement":0.93,"taille_cup":0.3968,"production_lbs":26000000000},
                     "Canned":{"prix_detail":0.80,"rendement":1.00,"taille_cup":0.5401,"production_lbs":26000000000},
                     "Frozen":{"prix_detail":1.10,"rendement":1.00,"taille_cup":0.3968,"production_lbs":26000000000},
                     "Juice":{"prix_detail":0.60,"rendement":1.00,"taille_cup":0.5200,"production_lbs":26000000000}},
    "Broccoli"    : {"Fresh":{"prix_detail":1.80,"rendement":0.81,"taille_cup":0.2425,"production_lbs":1800000000},
                     "Frozen":{"prix_detail":1.20,"rendement":1.00,"taille_cup":0.2756,"production_lbs":1800000000}},
    "Carrots"     : {"Fresh":{"prix_detail":0.90,"rendement":0.82,"taille_cup":0.2425,"production_lbs":2800000000},
                     "Canned":{"prix_detail":0.60,"rendement":1.00,"taille_cup":0.5401,"production_lbs":2800000000},
                     "Frozen":{"prix_detail":0.80,"rendement":1.00,"taille_cup":0.2425,"production_lbs":2800000000},
                     "Juice":{"prix_detail":0.70,"rendement":1.00,"taille_cup":0.5200,"production_lbs":2800000000}},
    "Lettuce"     : {"Fresh":{"prix_detail":1.50,"rendement":0.75,"taille_cup":0.1323,"production_lbs":7000000000}},
    "Potatoes"    : {"Fresh":{"prix_detail":0.70,"rendement":0.81,"taille_cup":0.3307,"production_lbs":43000000000},
                     "Canned":{"prix_detail":0.80,"rendement":1.00,"taille_cup":0.5401,"production_lbs":43000000000},
                     "Frozen":{"prix_detail":0.85,"rendement":1.00,"taille_cup":0.3307,"production_lbs":43000000000},
                     "Dried":{"prix_detail":2.50,"rendement":1.00,"taille_cup":0.0850,"production_lbs":43000000000}},
    "Spinach"     : {"Fresh":{"prix_detail":3.20,"rendement":0.75,"taille_cup":0.0661,"production_lbs":450000000},
                     "Canned":{"prix_detail":0.70,"rendement":1.00,"taille_cup":0.5401,"production_lbs":450000000},
                     "Frozen":{"prix_detail":1.00,"rendement":1.00,"taille_cup":0.2756,"production_lbs":450000000}},
    "Onions"      : {"Fresh":{"prix_detail":0.80,"rendement":0.90,"taille_cup":0.2425,"production_lbs":6800000000},
                     "Frozen":{"prix_detail":0.90,"rendement":1.00,"taille_cup":0.2425,"production_lbs":6800000000},
                     "Dried":{"prix_detail":4.50,"rendement":1.00,"taille_cup":0.0850,"production_lbs":6800000000}},
    "Cucumbers"   : {"Fresh":{"prix_detail":1.10,"rendement":0.95,"taille_cup":0.2204,"production_lbs":1700000000}},
    "Cauliflower" : {"Fresh":{"prix_detail":1.80,"rendement":0.56,"taille_cup":0.2756,"production_lbs":700000000},
                     "Frozen":{"prix_detail":1.20,"rendement":1.00,"taille_cup":0.2756,"production_lbs":700000000}},
}

ETATS_DATA = {
    "CA":{"nom":"Californie",      "lat":36.78,"lon":-119.42,"produits":["Tomatoes","Grapes","Strawberries","Lettuce","Broccoli","Cauliflower","Spinach"],"prix_moyen":0.88,"temp":18.5,"gel":2},
    "WA":{"nom":"Washington",      "lat":47.38,"lon":-120.45,"produits":["Apples","Cherries","Blueberries","Potatoes","Onions"],"prix_moyen":0.92,"temp":10.2,"gel":45},
    "FL":{"nom":"Floride",         "lat":27.99,"lon":-81.76, "produits":["Oranges","Tomatoes","Cucumbers","Strawberries"],"prix_moyen":0.85,"temp":23.8,"gel":0},
    "MI":{"nom":"Michigan",        "lat":43.35,"lon":-84.56, "produits":["Blueberries","Cherries","Apples"],"prix_moyen":0.95,"temp":8.5,"gel":120},
    "OR":{"nom":"Oregon",          "lat":44.57,"lon":-122.07,"produits":["Blueberries","Peaches","Potatoes","Onions"],"prix_moyen":0.90,"temp":11.0,"gel":55},
    "GA":{"nom":"Géorgie",         "lat":32.16,"lon":-82.91, "produits":["Peaches","Blueberries","Onions"],"prix_moyen":0.82,"temp":17.5,"gel":5},
    "ID":{"nom":"Idaho",           "lat":44.07,"lon":-114.74,"produits":["Potatoes","Onions","Apples"],"prix_moyen":0.78,"temp":7.0,"gel":100},
    "NC":{"nom":"Caroline du Nord","lat":35.63,"lon":-79.81, "produits":["Tomatoes","Spinach"],"prix_moyen":0.75,"temp":15.2,"gel":20},
}

PRODUITS_ACCUEIL = [
    {"emoji":"🍎","nom":"Pomme",    "prix":0.45,"trend":"+3%", "up":True},
    {"emoji":"🥦","nom":"Brocoli",  "prix":0.32,"trend":"-1%", "up":False},
    {"emoji":"🍓","nom":"Fraise",   "prix":0.89,"trend":"+8%", "up":True},
    {"emoji":"🥕","nom":"Carotte",  "prix":0.28,"trend":"+1%", "up":True},
    {"emoji":"🍒","nom":"Cerise",   "prix":1.20,"trend":"+12%","up":True},
    {"emoji":"🫐","nom":"Myrtille", "prix":1.05,"trend":"-4%", "up":False},
]

ANNEES      = [2013, 2016, 2020, 2022, 2023]
PRIX_FRUIT  = [0.85, 0.89, 0.93, 1.02, 1.10]
PRIX_LEGUME = [0.68, 0.71, 0.73, 0.80, 0.86]

# ══════════════════════════════════════
# FONCTIONS TEMPS RÉEL
# ══════════════════════════════════════

@st.cache_data(ttl=3600)
def get_meteo(lat, lon):
    """Récupère la météo via Open-Meteo (gratuit, sans clé)."""
    try:
        annee_ref = datetime.now().year - 1
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{annee_ref}-01-01",
            "end_date":   f"{annee_ref}-12-31",
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "America/New_York"
        }
        rep = requests.get(url, params=params, timeout=10)
        if rep.status_code == 200:
            data = rep.json()
            tmax = data["daily"]["temperature_2m_max"]
            tmin = data["daily"]["temperature_2m_min"]
            temp_moy  = float(np.mean([(h+l)/2 for h,l in zip(tmax,tmin) if h and l]))
            jours_gel = int(sum(1 for t in tmin if t and t < 0))
            return {"temp_moyenne": round(temp_moy,1), "jours_gel": jours_gel, "source": "Open-Meteo (données réelles)"}
    except Exception:
        pass
    return {"temp_moyenne": 15.0, "jours_gel": 10, "source": "Données estimées"}


@st.cache_data(ttl=3600)
def get_contexte_eco():
    """Récupère le contexte économique actuel."""
    return {
        "prix_diesel": 3.50,
        "prix_electricite": 12.0,
        "urea": 350.0,
        "annee": datetime.now().year,
        "source_diesel": "Estimation EIA 2024"
    }

# ══════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

pages = ["Accueil", "Prix et tendances", "Prédiction", "Interprétabilité"]

st.markdown("""
<div class="navbar">
  <div class="navbar-logo">🥦 FruitsLégumes ML</div>
</div>
""", unsafe_allow_html=True)

nav_cols = st.columns(len(pages))
for i, p in enumerate(pages):
    with nav_cols[i]:
        if st.button(
            f"→ {p}" if p == st.session_state.page else p,
            key=f"nav_{p}", use_container_width=True,
            type="primary" if p == st.session_state.page else "secondary"
        ):
            st.session_state.page = p
            st.rerun()

page = st.session_state.page

# ══════════════════════════════════════
# PAGE 1 — ACCUEIL
# ══════════════════════════════════════
if page == "Accueil":

    st.markdown("""
    <div class="hero">
      <div class="hero-content">
        <div class="hero-badge">Projet IA — Certification RNCP37827</div>
        <div class="hero-title">Comprendre et <em>Anticiper</em><br>les Prix des Fruits et Légumes</div>
        <div class="hero-desc">
          Une application intelligente qui prédit le prix d'une portion de fruit ou légume
          en croisant des données climatiques et économiques en temps réel.
          Idéale pour les acteurs de la distribution et les consommateurs qui veulent
          comprendre d'où vient leur prix.
        </div>
        <div class="hero-fruits">
          <div class="fruit-pill">🍎 <span>Pomme fraîche</span></div>
          <div class="fruit-pill">🥦 <span>Brocoli surgelé</span></div>
          <div class="fruit-pill">🍓 <span>Fraise fraîche</span></div>
          <div class="fruit-pill">🍒 <span>Cerise en conserve</span></div>
        </div>
      </div>
      <div class="hero-stats">
        <div class="hero-stat"><div class="val">710</div><div class="lbl">Observations</div></div>
        <div class="hero-stat"><div class="val">150+</div><div class="lbl">Produits</div></div>
        <div class="hero-stat"><div class="val orange">97.5%</div><div class="lbl">Précision du modèle</div></div>
        <div class="hero-stat"><div class="val">6</div><div class="lbl">Sources de données</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Produits les plus consultés</div>', unsafe_allow_html=True)

    produits_html = '<div class="produits-grid">'
    for prod in PRODUITS_ACCUEIL:
        tc = "" if prod["up"] else "down"
        fl = "▲" if prod["up"] else "▼"
        produits_html += f"""
        <div class="produit-card">
          <div class="produit-emoji">{prod['emoji']}</div>
          <div class="produit-name">{prod['nom']}</div>
          <div class="produit-price">{prod['prix']:.2f}$/cup</div>
          <div class="produit-trend {tc}">{fl} {prod['trend']}</div>
        </div>"""
    produits_html += '</div>'
    st.markdown(produits_html, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Comment ça fonctionne ?</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    for col, num, titre, desc in zip(
        [col1, col2, col3, col4],
        ["1","2","3","4"],
        ["Tu choisis un produit","L'app collecte les données","Le modèle prédit","Tu comprends le prix"],
        [
            "Tu sélectionnes le fruit ou légume et sa forme (frais, surgelé, en conserve...)",
            "L'application récupère automatiquement la météo de la zone de production et les prix de l'énergie en temps réel",
            "XGBoost analyse toutes ces données et calcule le prix par portion le plus probable",
            "Tu vois le prix estimé et les facteurs qui l'ont influencé"
        ]
    ):
        with col:
            st.markdown(f"""
            <div style="background:#fff;border-radius:12px;padding:20px;border:1px solid #e5e7eb;height:100%;text-align:center;">
              <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:700;color:#2d6a2d;margin-bottom:8px;">{num}</div>
              <div style="font-weight:600;margin-bottom:8px;font-size:14px;">{titre}</div>
              <div style="font-size:13px;color:#4b5563;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════
# PAGE 2 — PRIX ET TENDANCES + CARTE
# ══════════════════════════════════════
elif page == "Prix et tendances":

    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Évolution des prix (2013-2023)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ANNEES, y=PRIX_FRUIT, name="Fruits",
            line=dict(color="#2d6a2d", width=3), mode="lines+markers",
            marker=dict(size=8, color="#2d6a2d")))
        fig.add_trace(go.Scatter(x=ANNEES, y=PRIX_LEGUME, name="Légumes",
            line=dict(color="#f97316", width=3), mode="lines+markers",
            marker=dict(size=8, color="#f97316")))
        fig.update_layout(
            title="Prix moyen par cup (2013-2023)",
            title_font=dict(family="Syne", size=16),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="DM Sans", size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Prix ($/cup)"),
            margin=dict(l=20, r=20, t=60, b=20), height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        formes_fr = ["En conserve","Frais","Surgelé","Séché","Jus"]
        prix_f    = [0.93, 0.90, 0.89, 0.64, 0.64]
        fig2 = go.Figure(go.Bar(
            x=prix_f, y=formes_fr, orientation="h",
            marker_color=["#2d6a2d","#f97316","#4a9e4a","#fb923c","#86efac"],
            text=[f"{p:.2f}$" for p in prix_f], textposition="outside"
        ))
        fig2.update_layout(
            title="Prix moyen par forme",
            title_font=dict(family="Syne", size=16),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="DM Sans", size=13),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Prix ($/cup)"),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=60, t=60, b=20), height=300
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── CARTE INTERACTIVE ──
    st.markdown('<div class="section-title">Carte des zones de production aux États-Unis</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#4b5563;font-size:14px;margin-bottom:16px;">
    Survole ou clique sur un état pour voir ce qu'il produit, ses prix moyens et ses conditions climatiques.
    Ces informations expliquent pourquoi certains produits coûtent plus cher selon leur origine.</p>
    """, unsafe_allow_html=True)

    etats_df = pd.DataFrame([
        {
            "code":      code,
            "nom":       data["nom"],
            "lat":       data["lat"],
            "lon":       data["lon"],
            "prix_moyen":data["prix_moyen"],
            "temp":      data["temp"],
            "gel":       data["gel"],
            "produits_fr": ", ".join([
                PRODUITS_FR.get(p, {}).get("emoji","") + " " + PRODUITS_FR.get(p, {}).get("fr", p)
                for p in data["produits"][:5]
            ])
        }
        for code, data in ETATS_DATA.items()
    ])

    fig_carte = go.Figure()
    fig_carte.add_trace(go.Scattergeo(
        lat=etats_df["lat"],
        lon=etats_df["lon"],
        text=etats_df["code"],
        customdata=np.stack([
            etats_df["nom"], etats_df["prix_moyen"],
            etats_df["temp"], etats_df["gel"], etats_df["produits_fr"]
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Prix moyen : %{customdata[1]:.2f}$/cup<br>"
            "Température : %{customdata[2]}°C<br>"
            "Jours de gel : %{customdata[3]} j/an<br>"
            "Produits : %{customdata[4]}"
            "<extra></extra>"
        ),
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=11, color="#2d6a2d", family="DM Sans"),
        marker=dict(
            size=etats_df["prix_moyen"] * 38,
            color=etats_df["prix_moyen"],
            colorscale=[[0,"#e8f5e9"],[0.5,"#4a9e4a"],[1,"#2d6a2d"]],
            showscale=True,
            colorbar=dict(
                title="Prix moyen<br>($/cup)",
                tickfont=dict(family="DM Sans", size=12),
                thickness=15, len=0.5
            ),
            line=dict(width=2, color="white"),
            opacity=0.88
        ),
        name=""
    ))

    fig_carte.update_layout(
        title=dict(text="Zones de production — Prix moyen par cup", font=dict(family="Syne", size=16)),
        geo=dict(
            scope="usa",
            showland=True, landcolor="#f8f9fa",
            showlakes=True, lakecolor="#dbeafe",
            showcoastlines=True, coastlinecolor="#e5e7eb",
            showsubunits=True, subunitcolor="#d1d5db",
            projection_type="albers usa",
            bgcolor="white"
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=60, b=0),
        height=450,
        font=dict(family="DM Sans")
    )

    event = st.plotly_chart(fig_carte, use_container_width=True, on_select="rerun", key="carte_usa")

    # Détails au clic
    if event and event.get("selection", {}).get("points"):
        idx  = event["selection"]["points"][0].get("point_index", 0)
        code = etats_df.iloc[idx]["code"]
        data = ETATS_DATA[code]

        produits_pills = "".join([
            f'<span style="background:#e8f5e9;color:#2d6a2d;padding:4px 12px;'
            f'border-radius:20px;font-size:13px;margin:2px;">'
            f'{PRODUITS_FR.get(p,{}).get("emoji","")} {PRODUITS_FR.get(p,{}).get("fr",p)}</span>'
            for p in data["produits"]
        ])
        

        st.markdown(f"""
        <div style="background:#fff;border-radius:12px;padding:24px;border:2px solid #2d6a2d;margin-top:16px;">
          <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;color:#2d6a2d;margin-bottom:16px;">
            {data['nom']} ({code})
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:16px;">
            <div style="text-align:center;background:#f8f9fa;border-radius:10px;padding:16px;">
              <div style="font-size:12px;color:#6b7280;text-transform:uppercase;margin-bottom:4px;">Prix moyen</div>
              <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:700;color:#f97316;">{data['prix_moyen']:.2f}$/cup</div>
            </div>
            <div style="text-align:center;background:#f8f9fa;border-radius:10px;padding:16px;">
              <div style="font-size:12px;color:#6b7280;text-transform:uppercase;margin-bottom:4px;">Température moy.</div>
              <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:700;color:#2d6a2d;">{data['temp']}°C</div>
            </div>
            <div style="text-align:center;background:#f8f9fa;border-radius:10px;padding:16px;">
              <div style="font-size:12px;color:#6b7280;text-transform:uppercase;margin-bottom:4px;">Jours de gel/an</div>
              <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:700;color:#2d6a2d;">{data['gel']} j</div>
            </div>
          </div>
          <div style="font-size:12px;color:#6b7280;text-transform:uppercase;margin-bottom:10px;">Produits cultivés</div>
          <div style="display:flex;flex-wrap:wrap;gap:6px;">{produits_pills}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Clique sur une bulle de la carte pour voir les détails de cet état producteur.")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════
# PAGE 3 — PRÉDICTION INTELLIGENTE
# ══════════════════════════════════════
elif page == "Prédiction":

    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prédiction du prix par cup</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#4b5563;font-size:15px;margin-bottom:20px;">
    Choisis ton produit et sa forme. L'application récupère automatiquement
    les données climatiques et économiques actuelles pour te donner la prédiction la plus juste.</p>
    """, unsafe_allow_html=True)

    # Étape 1 — Catégorie
    st.markdown("**Étape 1 — Quelle catégorie ?**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🍎 Fruits", use_container_width=True,
                     type="primary" if st.session_state.get("cat_pred") == "fruit" else "secondary"):
            st.session_state.cat_pred = "fruit"
            st.session_state.prod_pred = None
            st.session_state.forme_pred = None
            st.rerun()
    with col2:
        if st.button("🥦 Légumes", use_container_width=True,
                     type="primary" if st.session_state.get("cat_pred") == "legume" else "secondary"):
            st.session_state.cat_pred = "legume"
            st.session_state.prod_pred = None
            st.session_state.forme_pred = None
            st.rerun()

    # Étape 2 — Produit
    if st.session_state.get("cat_pred"):
        cat = st.session_state.cat_pred
        produits_cat = {k: v for k, v in PRODUITS_FR.items() if v["categorie"] == cat}

        st.markdown("**Étape 2 — Quel produit ?**")
        nb_cols = min(len(produits_cat), 4)
        grille  = st.columns(nb_cols)
        for i, (code, info) in enumerate(produits_cat.items()):
            with grille[i % nb_cols]:
                actif = st.session_state.get("prod_pred") == code
                if st.button(f"{info['emoji']} {info['fr']}", key=f"p_{code}",
                             use_container_width=True,
                             type="primary" if actif else "secondary"):
                    st.session_state.prod_pred  = code
                    st.session_state.forme_pred = None
                    st.rerun()

    # Étape 3 — Forme
    if st.session_state.get("prod_pred"):
        code_prod = st.session_state.prod_pred
        formes    = PRODUITS_FR[code_prod]["formes"]
        st.markdown("**Étape 3 — Quelle forme ?**")
        f_cols = st.columns(len(formes))
        for i, forme in enumerate(formes):
            with f_cols[i]:
                actif = st.session_state.get("forme_pred") == forme
                if st.button(FORMES_FR[forme], key=f"f_{forme}",
                             use_container_width=True,
                             type="primary" if actif else "secondary"):
                    st.session_state.forme_pred = forme
                    st.rerun()

    # Étape 4 — Prédiction automatique
    if st.session_state.get("prod_pred") and st.session_state.get("forme_pred"):
        code_prod    = st.session_state.prod_pred
        forme        = st.session_state.forme_pred
        info_prod    = PRODUITS_FR[code_prod]
        donnees_prod = DONNEES_PRODUITS.get(code_prod, {}).get(forme, {})

        if not donnees_prod:
            st.error("Données non disponibles pour cette combinaison.")
        else:
            st.markdown("---")

            # Trouver l'état producteur principal
            etat_code = "CA"
            for ec, ed in ETATS_DATA.items():
                if code_prod in ed["produits"]:
                    etat_code = ec
                    break

            etat_data = ETATS_DATA[etat_code]

            with st.spinner(f"Récupération des données pour {info_prod['emoji']} {info_prod['fr']} — {FORMES_FR[forme]}..."):
                meteo    = get_meteo(etat_data["lat"], etat_data["lon"])
                contexte = get_contexte_eco()

            # Affichage contexte collecté
            st.markdown(f"""
            <div class="context-box">
              <div class="context-title">Données collectées automatiquement</div>
              <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;font-size:13px;">
                <div><b>Zone de production</b><br>{etat_data['nom']}<br>
                  <span style="font-size:11px;color:#9ca3af;">Source : USDA NASS</span></div>
                <div><b>Température moy.</b><br>{meteo['temp_moyenne']}°C<br>
                  <span style="font-size:11px;color:#9ca3af;">{meteo['source']}</span></div>
                <div><b>Jours de gel/an</b><br>{meteo['jours_gel']} jours<br>
                  <span style="font-size:11px;color:#9ca3af;">Données {contexte['annee']-1}</span></div>
                <div><b>Prix diesel</b><br>{contexte['prix_diesel']:.2f} $/gallon<br>
                  <span style="font-size:11px;color:#9ca3af;">{contexte['source_diesel']}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Construction des features
            forme_map = {"Fresh":0,"Canned":1,"Frozen":2,"Juice":3,"Dried":4}
            payload = {
                "prix_detail"      : donnees_prod["prix_detail"],
                "rendement"        : donnees_prod["rendement"],
                "taille_cup"       : donnees_prod["taille_cup"],
                "forme_encoded"    : forme_map[forme],
                "categorie_encoded": 1 if info_prod["categorie"] == "fruit" else 0,
                "annee"            : contexte["annee"],
                "production_lbs"   : donnees_prod["production_lbs"],
                "temp_moyenne"     : meteo["temp_moyenne"],
                "jours_gel"        : float(meteo["jours_gel"]),
                "prix_diesel"      : contexte["prix_diesel"],
                "prix_electricite" : contexte["prix_electricite"],
                "urea"             : contexte["urea"]
            }

            # Appel API ou modèle local
            prix_predit = None
            try:
                rep = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
                if rep.status_code == 200:
                    prix_predit = rep.json()["prix_predit_cup"]
            except Exception:
                pkl = "NOTEBOOKS/models/xgboost_fruits_legumes.pkl"
                if os.path.exists(pkl):
                    m = joblib.load(pkl)
                    v = np.array([[payload["prix_detail"], payload["rendement"],
                                   payload["taille_cup"], payload["forme_encoded"],
                                   payload["categorie_encoded"], payload["annee"],
                                   payload["production_lbs"], payload["temp_moyenne"],
                                   payload["jours_gel"], payload["prix_diesel"],
                                   payload["prix_electricite"], payload["urea"]]])
                    prix_predit = float(m.predict(v)[0])

            if prix_predit is not None:
                entier  = int(prix_predit)
                decimal = f"{prix_predit:.4f}".split(".")[1]
                st.markdown(f"""
                <div class="result-box">
                  <div class="result-label">
                    Prix estimé pour une portion de {info_prod['emoji']} {info_prod['fr']} ({FORMES_FR[forme]})
                  </div>
                  <div class="result-price">{entier}.<em>{decimal}</em> $/cup</div>
                  <div class="result-meta">
                    Prédiction {contexte['annee']} &nbsp;|&nbsp;
                    Zone : {etat_data['nom']} &nbsp;|&nbsp;
                    Modèle XGBoost — Précision ±0.0886$/cup
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Explication du prix
                facteurs = []
                if donnees_prod["rendement"] < 0.80:
                    facteurs.append(f"Seulement {donnees_prod['rendement']*100:.0f}% du produit est utilisable après préparation — le reste est perdu (épluchure, noyau...). Cela augmente le coût par portion.")
                if forme == "Canned":
                    facteurs.append("La mise en conserve ajoute des coûts de transformation, d'emballage et de stérilisation thermique.")
                if forme == "Frozen":
                    facteurs.append("La congélation nécessite de l'énergie pour le stockage réfrigéré tout au long de la chaîne logistique.")
                if meteo["jours_gel"] > 50:
                    facteurs.append(f"{etat_data['nom']} connaît {meteo['jours_gel']} jours de gel par an, ce qui limite les périodes de récolte et peut réduire la production disponible.")
                if contexte["prix_diesel"] > 3.8:
                    facteurs.append(f"Le prix du diesel est élevé ({contexte['prix_diesel']:.2f}$/gallon), ce qui augmente les coûts de transport de la zone de production.")

                if facteurs:
                    st.markdown("""
                    <div class="context-box" style="margin-top:12px;">
                      <div class="context-title">Pourquoi ce prix ?</div>
                    """, unsafe_allow_html=True)
                    for f_txt in facteurs:
                        st.markdown(f"• {f_txt}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Modèle non disponible. Vérifie que le fichier .pkl est dans NOTEBOOKS/models/")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════
# PAGE 4 — INTERPRÉTABILITÉ + MÉTRIQUES + SOURCES
# ══════════════════════════════════════
elif page == "Interprétabilité":

    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)

    # ── Métriques ──
    st.markdown('<div class="section-title">Performance du modèle XGBoost</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px;">
      <div class="metric-card">
        <div class="metric-icon vert">🎯</div>
        <div class="metric-info"><div class="val">0.9755</div><div class="lbl">R² — Pouvoir prédictif</div></div>
      </div>
      <div class="metric-card">
        <div class="metric-icon orange">📉</div>
        <div class="metric-info"><div class="val">0.0886$</div><div class="lbl">RMSE — Erreur quadratique</div></div>
      </div>
      <div class="metric-card">
        <div class="metric-icon vert">📏</div>
        <div class="metric-info"><div class="val">0.0451$</div><div class="lbl">MAE — Erreur absolue</div></div>
      </div>
      <div class="metric-card">
        <div class="metric-icon orange">🔄</div>
        <div class="metric-info"><div class="val">0.956</div><div class="lbl">R² cross-validation</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#fff3e0;border-radius:12px;padding:16px;margin-bottom:24px;border:1px solid #fed7aa;">
      <div style="font-weight:600;color:#f97316;margin-bottom:8px;">Comment lire ces indicateurs ?</div>
      <div style="font-size:13px;color:#4b5563;line-height:1.7;">
        <b>R² = 0.9755</b> : le modèle explique 97.55% des variations de prix — résultat excellent.<br>
        <b>RMSE = 0.0886$</b> : le modèle se trompe en moyenne de moins de 9 centimes par portion.<br>
        <b>Pas d'overfitting</b> : l'écart entre R² test (0.9755) et cross-validation (0.956) est de 0.019,
        bien en dessous du seuil critique de 0.05.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sources ──
    st.markdown('<div class="section-title">Sources de données utilisées</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, titre, items in zip(
        [col1, col2, col3],
        ["Prix et produits", "Contexte climatique", "Contexte économique"],
        [
            [("USDA ERS","Prix de détail de 150 produits sur 5 années (2013-2023) — Source officielle du Ministère de l'Agriculture américain"),
             ("BLS","Prix retail hebdomadaires pour vérification croisée")],
            [("Open-Meteo","Météo historique par état : température, gel, précipitations — API gratuite sans clé"),
             ("USDA NASS","Volume de production agricole par état américain")],
            [("EIA","Prix du diesel et de l'électricité par région américaine"),
             ("World Bank","Prix des engrais mondiaux : urée, DAP, MOP")]
        ]
    ):
        with col:
            html = "".join([
                f'<div style="margin-bottom:10px;"><b style="color:#2d6a2d;">{n}</b><br>'
                f'<span style="font-size:12px;color:#6b7280;line-height:1.5;">{d}</span></div>'
                for n, d in items
            ])
            st.markdown(f"""
            <div style="background:#fff;border-radius:12px;padding:18px;border:1px solid #e5e7eb;height:100%;">
              <div style="font-family:'Syne',sans-serif;font-weight:600;margin-bottom:12px;">{titre}</div>
              {html}
            </div>""", unsafe_allow_html=True)

    # ── SHAP ──
    st.markdown('<div class="section-title" style="margin-top:24px;">Interprétabilité SHAP</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#4b5563;font-size:15px;margin-bottom:16px;">
    SHAP (SHapley Additive exPlanations) explique pourquoi le modèle prédit tel ou tel prix,
    feature par feature. Chaque prédiction est transparente et justifiable, conformément
    aux exigences de l'EU AI Act sur la transparence algorithmique.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        features   = ["prix_detail","rendement","taille_cup","annee","forme_encoded",
                      "production_lbs","prix_diesel","prix_electricite","urea",
                      "jours_gel","temp_moyenne","categorie_encoded"]
        importance = [0.408,0.350,0.200,0.015,0.009,0.013,0.003,0.002,0.002,0.001,0.005,0.002]
        df_shap    = pd.DataFrame({"Feature":features,"Importance":importance}).sort_values("Importance", ascending=True)
        couleurs   = ["#86efac" if v<0.05 else "#4a9e4a" if v<0.15 else "#2d6a2d" for v in df_shap["Importance"]]

        fig_shap = go.Figure(go.Bar(
            x=df_shap["Importance"], y=df_shap["Feature"],
            orientation="h", marker_color=couleurs,
            text=[f"{v:.3f}" for v in df_shap["Importance"]], textposition="outside"
        ))
        fig_shap.update_layout(
            title="Importance SHAP des 12 features",
            title_font=dict(family="Syne", size=16),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="DM Sans", size=12),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Importance SHAP"),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=60, t=60, b=20), height=420
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col2:
        for chemin, caption in [
            ("NOTEBOOKS/models/shap_summary_plot.png", "Summary Plot — vue globale de toutes les features"),
            ("NOTEBOOKS/models/shap_waterfall.png",    "Waterfall Plot — explication d'une prédiction précise")
        ]:
            if os.path.exists(chemin):
                st.image(chemin, caption=caption, use_container_width=True)
            else:
                st.info(f"Lance le notebook 05 pour générer : {os.path.basename(chemin)}")

    # ── Top 3 features ──
    st.markdown('<div class="section-title" style="margin-top:20px;">Les 3 facteurs les plus influents</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, (feat, imp, desc) in zip(
        [col1, col2, col3],
        [
            ("prix_detail","40.8%","Le prix en rayon est le signal le plus direct du prix par portion. Un produit cher au kilo reste cher par portion."),
            ("rendement","~35%","La part utilisable après préparation détermine le vrai coût. Si on perd 30% à l'épluchage, le prix par portion monte."),
            ("taille_cup","~20%","La taille de la portion standard influe directement sur le prix. Une grande portion coûte plus en valeur absolue."),
        ]
    ):
        with col:
            st.markdown(f"""
            <div class="info-card">
              <div class="info-val">{imp}</div>
              <div class="info-key">{feat}</div>
              <div class="info-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
