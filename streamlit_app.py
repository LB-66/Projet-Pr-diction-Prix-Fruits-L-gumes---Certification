"""
Application Streamlit — Prédiction des prix des Fruits et Légumes
Fidèle à la maquette HTML dashboard_fruits_legumes.html
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import joblib
import os

# ── Configuration de la page ──
st.set_page_config(
    page_title="FruitsLégumes ML — Dashboard",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Style CSS fidèle à la maquette ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --vert: #2d6a2d;
  --vert-clair: #e8f5e9;
  --vert-mid: #4a9e4a;
  --orange: #f97316;
  --orange-clair: #fff3e0;
  --blanc: #ffffff;
  --gris-clair: #f8f9fa;
  --gris: #6b7280;
  --gris-border: #e5e7eb;
  --texte: #1a1a1a;
  --texte-secondaire: #4b5563;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  color: var(--texte);
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ─── Navbar ─── */
.navbar {
  background: var(--vert);
  padding: 0 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 60px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.navbar-logo {
  font-family: 'Syne', sans-serif;
  font-size: 18px;
  font-weight: 700;
  color: #fff;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ─── Hero ─── */
.hero {
  background: linear-gradient(135deg, var(--vert-clair) 0%, var(--orange-clair) 100%);
  padding: 48px 32px;
  display: flex;
  gap: 48px;
  align-items: center;
  border-bottom: 1px solid var(--gris-border);
}
.hero-content { flex: 1; }
.hero-badge {
  display: inline-block;
  background: var(--vert);
  color: #fff;
  font-size: 11px;
  font-weight: 500;
  padding: 4px 12px;
  border-radius: 20px;
  margin-bottom: 16px;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 32px;
  font-weight: 700;
  color: var(--vert);
  line-height: 1.2;
  margin-bottom: 16px;
}
.hero-title em { color: var(--orange); font-style: normal; }
.hero-desc {
  font-size: 15px;
  color: var(--texte-secondaire);
  line-height: 1.7;
  max-width: 480px;
  margin-bottom: 24px;
}
.hero-fruits { display: flex; gap: 10px; flex-wrap: wrap; }
.fruit-pill {
  background: #fff;
  border: 1px solid var(--gris-border);
  border-radius: 20px;
  padding: 8px 14px;
  font-size: 18px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.fruit-pill span { font-size: 12px; color: var(--texte-secondaire); }

/* ─── Hero stats ─── */
.hero-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  min-width: 260px;
}
.hero-stat {
  background: #fff;
  border-radius: 12px;
  padding: 16px;
  text-align: center;
  border: 1px solid var(--gris-border);
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.hero-stat .val {
  font-family: 'Syne', sans-serif;
  font-size: 28px;
  font-weight: 700;
  color: var(--vert);
}
.hero-stat .val.orange { color: var(--orange); }
.hero-stat .lbl { font-size: 12px; color: var(--texte-secondaire); margin-top: 4px; }

/* ─── Metrics row ─── */
.metrics-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  padding: 24px 32px;
  background: var(--gris-clair);
}
.metric-card {
  background: #fff;
  border-radius: 12px;
  padding: 18px;
  border: 1px solid var(--gris-border);
  display: flex;
  align-items: center;
  gap: 14px;
}
.metric-icon {
  width: 44px; height: 44px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  flex-shrink: 0;
}
.metric-icon.vert   { background: var(--vert-clair); }
.metric-icon.orange { background: var(--orange-clair); }
.metric-info .val {
  font-family: 'Syne', sans-serif;
  font-size: 22px;
  font-weight: 700;
  color: var(--texte);
}
.metric-info .lbl { font-size: 12px; color: var(--texte-secondaire); margin-top: 2px; }

/* ─── Section ─── */
.section-wrap { padding: 24px 32px; background: var(--gris-clair); }
.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 18px;
  font-weight: 600;
  color: var(--texte);
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--gris-border);
}

/* ─── Produit cards ─── */
.produits-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 12px;
  margin-bottom: 24px;
}
.produit-card {
  background: #fff;
  border: 1px solid var(--gris-border);
  border-radius: 12px;
  padding: 16px 12px;
  text-align: center;
}
.produit-emoji  { font-size: 32px; margin-bottom: 8px; }
.produit-name   { font-size: 12px; font-weight: 500; margin-bottom: 4px; }
.produit-price  { font-size: 14px; font-weight: 600; color: var(--orange); }
.produit-trend  { font-size: 11px; color: var(--vert); margin-top: 2px; }
.produit-trend.down { color: #dc2626; }

/* ─── Résultat prédiction ─── */
.result-box {
  background: linear-gradient(135deg, var(--vert-clair), var(--orange-clair));
  border-radius: 16px;
  padding: 32px;
  text-align: center;
  border: 1px solid var(--gris-border);
  margin-top: 16px;
}
.result-label {
  font-size: 13px;
  color: var(--texte-secondaire);
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.result-price {
  font-family: 'Syne', sans-serif;
  font-size: 56px;
  font-weight: 700;
  color: var(--vert);
  line-height: 1;
  margin-bottom: 8px;
}
.result-price em { color: var(--orange); font-style: normal; font-size: 28px; }
.result-meta { font-size: 13px; color: var(--texte-secondaire); }

/* ─── Carte info ─── */
.info-card {
  background: #fff;
  border-radius: 12px;
  padding: 18px;
  border: 1px solid var(--gris-border);
  border-top: 3px solid var(--vert);
}
.info-val {
  font-family: 'Syne', sans-serif;
  font-size: 22px;
  font-weight: 700;
  color: var(--vert);
}
.info-key { font-weight: 500; margin: 4px 0 8px; font-size: 14px; }
.info-desc { font-size: 13px; color: var(--texte-secondaire); line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════
# DONNÉES
# ══════════════════════════════════════
PRODUITS = [
    {"emoji": "🍎", "nom": "Pomme",    "prix": 0.45, "trend": "+3%",  "up": True},
    {"emoji": "🥦", "nom": "Brocoli",  "prix": 0.32, "trend": "-1%",  "up": False},
    {"emoji": "🍓", "nom": "Fraise",   "prix": 0.89, "trend": "+8%",  "up": True},
    {"emoji": "🥕", "nom": "Carotte",  "prix": 0.28, "trend": "+1%",  "up": True},
    {"emoji": "🍒", "nom": "Cerise",   "prix": 1.20, "trend": "+12%", "up": True},
    {"emoji": "🫐", "nom": "Myrtille", "prix": 1.05, "trend": "-4%",  "up": False},
]
ANNEES      = [2013, 2016, 2020, 2022, 2023]
PRIX_FRUIT  = [0.85, 0.89, 0.93, 1.02, 1.10]
PRIX_LEGUME = [0.68, 0.71, 0.73, 0.80, 0.86]

# ══════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

pages = ["Accueil", "Prix et tendances", "Prédiction", "Interprétabilité"]

# Navbar statique
st.markdown("""
<div class="navbar">
  <div class="navbar-logo">🥦 FruitsLégumes ML</div>
</div>
""", unsafe_allow_html=True)

# Boutons de navigation
nav_cols = st.columns(len(pages))
for i, p in enumerate(pages):
    with nav_cols[i]:
        label = f"{'→ ' if p == st.session_state.page else ''}{p}"
        if st.button(label, key=f"nav_{p}", use_container_width=True,
                     type="primary" if p == st.session_state.page else "secondary"):
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
          Une application de Machine Learning qui prédit le prix par cup d'un fruit
          ou légume américain à partir de données économiques et climatiques.
          Modèle XGBoost entraîné sur 5 années de données USDA ERS.
        </div>
        <div class="hero-fruits">
          <div class="fruit-pill">🍎 <span>0.45$/cup</span></div>
          <div class="fruit-pill">🥦 <span>0.32$/cup</span></div>
          <div class="fruit-pill">🍓 <span>0.89$/cup</span></div>
          <div class="fruit-pill">🥕 <span>0.28$/cup</span></div>
          <div class="fruit-pill">🍒 <span>1.20$/cup</span></div>
        </div>
      </div>
      <div class="hero-stats">
        <div class="hero-stat"><div class="val">710</div><div class="lbl">Observations</div></div>
        <div class="hero-stat"><div class="val">150+</div><div class="lbl">Produits</div></div>
        <div class="hero-stat"><div class="val orange">0.9755</div><div class="lbl">R² du modèle</div></div>
        <div class="hero-stat"><div class="val">6</div><div class="lbl">Sources</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metrics-row">
      <div class="metric-card">
        <div class="metric-icon vert">🎯</div>
        <div class="metric-info"><div class="val">0.9755</div><div class="lbl">R² — Pouvoir prédictif</div></div>
      </div>
      <div class="metric-card">
        <div class="metric-icon orange">📉</div>
        <div class="metric-info"><div class="val">0.0886$</div><div class="lbl">RMSE — Erreur moyenne</div></div>
      </div>
      <div class="metric-card">
        <div class="metric-icon vert">⚡</div>
        <div class="metric-info"><div class="val">107ms</div><div class="lbl">Temps de réponse API</div></div>
      </div>
      <div class="metric-card">
        <div class="metric-icon orange">📦</div>
        <div class="metric-info"><div class="val">278 Ko</div><div class="lbl">Taille du modèle .pkl</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Produits les plus consultés</div>',
                unsafe_allow_html=True)

    produits_html = '<div class="produits-grid">'
    for prod in PRODUITS:
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

    st.markdown('<div class="section-title">Sources de données</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, titre, contenu in zip(
        [col1, col2, col3],
        ["Prix des produits", "Contexte climatique", "Contexte économique"],
        [
            "USDA ERS — Prix de détail 150 produits<br>BLS — Prix retail hebdomadaires",
            "Open-Meteo — Météo historique<br>USDA NASS — Production par état",
            "EIA — Prix diesel et électricité<br>World Bank — Prix des engrais"
        ]
    ):
        with col:
            st.markdown(f"""
            <div style="background:#fff;border-radius:12px;padding:18px;
                        border:1px solid #e5e7eb;height:100%;">
              <div style="font-family:'Syne',sans-serif;font-weight:600;
                          margin-bottom:8px;">{titre}</div>
              <div style="font-size:13px;color:#4b5563;line-height:1.7;">
                {contenu}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════
# PAGE 2 — PRIX ET TENDANCES
# ══════════════════════════════════════
elif page == "Prix et tendances":

    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prix et tendances (2013-2023)</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ANNEES, y=PRIX_FRUIT, name="Fruits",
            line=dict(color="#2d6a2d", width=3),
            mode="lines+markers", marker=dict(size=8, color="#2d6a2d")
        ))
        fig.add_trace(go.Scatter(
            x=ANNEES, y=PRIX_LEGUME, name="Légumes",
            line=dict(color="#f97316", width=3),
            mode="lines+markers", marker=dict(size=8, color="#f97316")
        ))
        fig.update_layout(
            title="Évolution du prix moyen par cup",
            title_font=dict(family="Syne", size=16),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="DM Sans", size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Prix ($/cup)"),
            margin=dict(l=20, r=20, t=60, b=20), height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        formes = ["Fresh", "Canned", "Frozen", "Dried", "Juice"]
        prix_f = [0.90, 0.93, 0.89, 0.64, 0.64]
        fig2 = go.Figure(go.Bar(
            x=prix_f, y=formes, orientation="h",
            marker_color=["#2d6a2d", "#f97316", "#4a9e4a", "#fb923c", "#86efac"],
            text=[f"{p:.2f}$" for p in prix_f], textposition="outside"
        ))
        fig2.update_layout(
            title="Prix moyen par forme",
            title_font=dict(family="Syne", size=16),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="DM Sans", size=13),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Prix ($/cup)"),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=60, t=60, b=20), height=320
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Impact du contexte économique sur les prix</div>',
                unsafe_allow_html=True)
    df_ctx = pd.DataFrame({
        "Année"              : ANNEES,
        "Prix moyen ($/cup)" : [0.77, 0.80, 0.82, 0.90, 0.97],
        "Prix diesel ($/gal)": [4.05, 2.93, 3.20, 4.99, 4.60],
        "Urée ($/tonne)"     : [304,  212,  233,  726,  358]
    })
    st.dataframe(
        df_ctx.style
        .highlight_max(
            subset=["Prix moyen ($/cup)", "Prix diesel ($/gal)", "Urée ($/tonne)"],
            color="#fff3e0"
        )
        .format({
            "Prix moyen ($/cup)": "{:.4f}",
            "Prix diesel ($/gal)": "{:.2f}",
            "Urée ($/tonne)": "{:.0f}"
        }),
        use_container_width=True, hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════
# PAGE 3 — PRÉDICTION
# ══════════════════════════════════════
elif page == "Prédiction":

    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prédiction du prix par cup</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#4b5563;font-size:15px;margin-bottom:20px;">
    Renseigne les caractéristiques d'un produit pour obtenir une estimation
    de son prix par cup equivalent.</p>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Produit**")
            forme     = st.selectbox("Forme", ["Fresh", "Canned", "Frozen", "Juice", "Dried"])
            categorie = st.selectbox("Catégorie", ["fruit", "légume"])
            annee     = st.selectbox("Année", [2013,2016,2020,2022,2023,2024,2025,2026], index=5)

        with col2:
            st.markdown("**Prix et caractéristiques**")
            prix_detail = st.number_input("Prix en rayon ($/lb)", 0.10, 10.0, 1.50, 0.10)
            rendement   = st.slider("Rendement", 0.10, 1.0, 0.75, 0.05)
            taille_cup  = st.number_input("Taille portion (lb)", 0.10, 1.0, 0.33, 0.05)

        with col3:
            st.markdown("**Contexte économique**")
            prix_diesel      = st.number_input("Prix diesel ($/gal)", 1.0, 8.0, 3.50, 0.10)
            prix_electricite = st.number_input("Électricité (¢/kWh)", 5.0, 30.0, 12.0, 0.5)
            urea             = st.number_input("Urée ($/tonne)", 100.0, 1000.0, 350.0, 10.0)

        soumettre = st.form_submit_button(
            "Prédire le prix", type="primary", use_container_width=True
        )

    if soumettre:
        forme_map = {"Fresh": 0, "Canned": 1, "Frozen": 2, "Juice": 3, "Dried": 4}
        donnees = {
            "prix_detail": prix_detail, "rendement": rendement,
            "taille_cup": taille_cup,
            "forme_encoded": forme_map[forme],
            "categorie_encoded": 1 if categorie == "fruit" else 0,
            "annee": annee, "production_lbs": 500000000.0,
            "temp_moyenne": 15.0, "jours_gel": 10.0,
            "prix_diesel": prix_diesel,
            "prix_electricite": prix_electricite, "urea": urea
        }

        prix_predit = None
        try:
            rep = requests.post("http://localhost:8000/predict", json=donnees, timeout=5)
            if rep.status_code == 200:
                prix_predit = rep.json()["prix_predit_cup"]
        except Exception:
            pkl = "../models/xgboost_fruits_legumes.pkl"
            if os.path.exists(pkl):
                m = joblib.load(pkl)
                v = np.array([[donnees["prix_detail"], donnees["rendement"],
                               donnees["taille_cup"], donnees["forme_encoded"],
                               donnees["categorie_encoded"], donnees["annee"],
                               donnees["production_lbs"], donnees["temp_moyenne"],
                               donnees["jours_gel"], donnees["prix_diesel"],
                               donnees["prix_electricite"], donnees["urea"]]])
                prix_predit = float(m.predict(v)[0])

        if prix_predit is not None:
            entier  = int(prix_predit)
            decimal = f"{prix_predit:.4f}".split(".")[1]
            st.markdown(f"""
            <div class="result-box">
              <div class="result-label">Prix estimé par cup equivalent</div>
              <div class="result-price">{entier}.<em>{decimal}</em> $</div>
              <div class="result-meta">
                XGBoost &nbsp;|&nbsp; R² = 0.9755 &nbsp;|&nbsp;
                Erreur moyenne : ±0.0886$/cup
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.error("Modèle non disponible. Lance le notebook 05 pour générer le .pkl.")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════
# PAGE 4 — INTERPRÉTABILITÉ
# ══════════════════════════════════════
elif page == "Interprétabilité":

    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Interprétabilité SHAP</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#4b5563;font-size:15px;margin-bottom:20px;">
    SHAP explique pourquoi le modèle prédit tel ou tel prix, feature par feature.
    Chaque prédiction est transparente et justifiable.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        features   = ["prix_detail", "rendement", "taille_cup", "annee",
                      "forme_encoded", "production_lbs", "prix_diesel",
                      "prix_electricite", "urea", "jours_gel",
                      "temp_moyenne", "categorie_encoded"]
        importance = [0.408, 0.350, 0.200, 0.015, 0.009, 0.013,
                      0.003, 0.002, 0.002, 0.001, 0.005, 0.002]

        df_shap = pd.DataFrame({"Feature": features, "Importance SHAP": importance})
        df_shap = df_shap.sort_values("Importance SHAP", ascending=True)

        couleurs = ["#86efac" if v < 0.05 else
                    "#4a9e4a" if v < 0.15 else
                    "#2d6a2d" for v in df_shap["Importance SHAP"]]

        fig_shap = go.Figure(go.Bar(
            x=df_shap["Importance SHAP"], y=df_shap["Feature"],
            orientation="h", marker_color=couleurs,
            text=[f"{v:.3f}" for v in df_shap["Importance SHAP"]],
            textposition="outside"
        ))
        fig_shap.update_layout(
            title="Importance SHAP des 12 features",
            title_font=dict(family="Syne", size=16),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="DM Sans", size=12),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                       title="Importance SHAP moyenne"),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=60, t=60, b=20), height=420
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col2:
        st.markdown("**Graphiques SHAP générés par le notebook 05**")
        for chemin, caption in [
            ("../models/shap_summary_plot.png",  "Summary Plot — vue globale"),
            ("../models/shap_waterfall.png",      "Waterfall Plot — explication d'une prédiction")
        ]:
            if os.path.exists(chemin):
                st.image(chemin, caption=caption, use_container_width=True)
            else:
                st.info(f"Lance le notebook 05 pour générer : {os.path.basename(chemin)}")

    st.markdown('<div class="section-title" style="margin-top:20px;">Top 3 features influentes</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, (feat, imp, desc) in zip(
        [col1, col2, col3],
        [
            ("prix_detail", "40.8%",
             "Le prix en rayon est le signal le plus direct du prix par portion."),
            ("rendement", "~35%",
             "La part utilisable après préparation détermine le coût réel."),
            ("taille_cup", "~20%",
             "La taille de la portion standard influe directement sur le prix cup."),
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
