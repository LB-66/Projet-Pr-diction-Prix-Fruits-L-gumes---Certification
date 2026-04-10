"""
Script d'import des données dans PostgreSQL
Projet : Prédiction prix fruits & légumes
Compétences : C2, C4
"""

import pandas as pd          # Pour lire le CSV
import psycopg2              # Pour se connecter à PostgreSQL
from psycopg2.extras import execute_values  # Pour insérer rapidement
import os
#from dotenv import load_dotenv  # Pour lire le fichier .env

# ── Chargement des variables d'environnement ──
#load_dotenv(encoding='utf-8')

# ── Paramètres de connexion depuis le .env ──
CONN_PARAMS = {
    'host'    : 'localhost',
    'port'    : '5432',
    'dbname'  : 'fruits_legumes_db',
    'user'    : 'admin',
    'password': 'admin123'
}

# ── Chemin vers le fichier CSV enrichi ──
CSV_PATH = '../DATA/CLEAN/fruits_legumes_enrichi.csv'

print('=' * 55)
print('IMPORT DONNÉES POSTGRESQL')
print('=' * 55)

# ── Étape 1 : Chargement du CSV ──
print('\n1. Chargement du CSV...')
df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8')
print(f'   Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes')
print(f'   Colonnes : {list(df.columns)}')

# ── Étape 2 : Connexion à PostgreSQL ──
print('\n2. Connexion à PostgreSQL...')
conn = psycopg2.connect(**CONN_PARAMS)
cur  = conn.cursor()
print('   Connexion réussie !')

# ── Étape 3 : Import table PRODUIT ──
print('\n3. Import table produit...')

# On extrait les produits uniques du CSV
produits_uniques = df[['produit', 'categorie', 'forme']].drop_duplicates()

# On insère chaque produit unique
produit_map = {}  # Dictionnaire pour retrouver l'id_produit par nom+forme
for _, row in produits_uniques.iterrows():
    cur.execute("""
        INSERT INTO produit (nom, categorie, forme)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING id_produit
    """, (row['produit'], row['categorie'], row['forme']))

    result = cur.fetchone()
    if result:
        produit_map[(row['produit'], row['forme'])] = result[0]

# Récupérer tous les ids produits
cur.execute("SELECT id_produit, nom, forme FROM produit")
for id_p, nom, forme in cur.fetchall():
    produit_map[(nom, forme)] = id_p

conn.commit()
print(f'   {len(produit_map)} produits importés')

# ── Étape 4 : Import table ETAT_PRODUCTEUR ──
print('\n4. Import table etat_producteur...')

# On extrait les états uniques
if 'etat_production' in df.columns and 'code_etat' in df.columns:
    etats_uniques = df[['code_etat', 'etat_production']].drop_duplicates().dropna()
    for _, row in etats_uniques.iterrows():
        cur.execute("""
            INSERT INTO etat_producteur (code_etat, nom_etat)
            VALUES (%s, %s)
            ON CONFLICT (code_etat) DO NOTHING
        """, (str(row['code_etat']), str(row['etat_production'])))
    conn.commit()
    print(f'   {len(etats_uniques)} états importés')
else:
    # Si pas de colonne état, on crée un état générique
    cur.execute("""
        INSERT INTO etat_producteur (code_etat, nom_etat)
        VALUES ('USA', 'United States')
        ON CONFLICT (code_etat) DO NOTHING
    """)
    conn.commit()
    print('   État générique USA créé')

# ── Étape 5 : Import table CONTEXTE_ANNUEL ──
print('\n5. Import table contexte_annuel...')

# On groupe par année pour créer un contexte annuel unique
cols_contexte = ['annee']
for col in ['temp_moyenne', 'jours_gel', 'jours_chaleur',
            'precip_totale', 'prix_diesel', 'urea']:
    if col in df.columns:
        cols_contexte.append(col)

contextes_uniques = df[cols_contexte].drop_duplicates(subset=['annee']).dropna(subset=['annee'])

contexte_map = {}  # Dictionnaire annee → id_contexte

for _, row in contextes_uniques.iterrows():
    annee        = int(row['annee'])
    temp_moyenne = float(row.get('temp_moyenne', 0) or 0)
    jours_gel    = float(row.get('jours_gel', 0) or 0)
    jours_chaleur= float(row.get('jours_chaleur', 0) or 0)
    precip_totale= float(row.get('precip_totale', 0) or 0)
    prix_diesel  = float(row.get('prix_diesel', 0) or 0)
    urea         = float(row.get('urea', 0) or 0)

    cur.execute("""
        INSERT INTO contexte_annuel
            (annee, temp_moyenne, jours_gel, jours_chaleur,
             precip_totale, prix_diesel, urea)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id_contexte
    """, (annee, temp_moyenne, jours_gel, jours_chaleur,
          precip_totale, prix_diesel, urea))

    result = cur.fetchone()
    if result:
        contexte_map[annee] = result[0]

conn.commit()
print(f'   {len(contexte_map)} contextes annuels importés')

# ── Étape 6 : Import table PRIX ──
print('\n6. Import table prix...')

nb_inseres = 0
nb_erreurs = 0

for _, row in df.iterrows():
    try:
        # Récupération des clés étrangères
        id_produit = produit_map.get((row['produit'], row['forme']))
        annee      = int(row['annee'])
        id_contexte= contexte_map.get(annee)

        if id_produit is None or id_contexte is None:
            nb_erreurs += 1
            continue

        cur.execute("""
            INSERT INTO prix
                (annee, prix_detail, prix_cup, rendement,
                 taille_cup, id_produit, id_contexte)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            annee,
            float(row.get('prix_detail', 0) or 0),
            float(row.get('prix_cup', 0) or 0),
            float(row.get('rendement', 0) or 0),
            float(row.get('taille_cup', 0) or 0),
            id_produit,
            id_contexte
        ))
        nb_inseres += 1

    except Exception as e:
        nb_erreurs += 1

conn.commit()
print(f'   {nb_inseres} prix importés')
if nb_erreurs > 0:
    print(f'   {nb_erreurs} lignes ignorées (données manquantes)')

# ── Étape 7 : Vérification ──
print('\n7. Vérification des imports...')
print('-' * 45)

for table in ['produit', 'etat_producteur', 'contexte_annuel', 'prix']:
    cur.execute(f'SELECT COUNT(*) FROM {table}')
    count = cur.fetchone()[0]
    print(f'   {table:<20} : {count:>5} lignes')

print()
print('=' * 55)
print('IMPORT TERMINÉ AVEC SUCCÈS !')
print('=' * 55)

# ── Fermeture de la connexion ──
cur.close()
conn.close()
