import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import numpy as np
import folium
from folium import Circle
from streamlit_folium import st_folium
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# remonter de 3 dossiers
DATA_RAW_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "data", "raw"))
DATA_PROCESSED_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "data", "processed"))

usagers_path = os.path.join(DATA_RAW_DIR, "usagers-2019.csv")
vehicules_path = os.path.join(DATA_RAW_DIR, "vehicules-2019.csv")
lieux_path = os.path.join(DATA_RAW_DIR, "lieux-2019.csv")
caracteristiques_path = os.path.join(DATA_RAW_DIR, "caracteristiques-2019.csv")
accidents_path = os.path.join(DATA_PROCESSED_DIR, "accidents_clean.csv")


def run():
    st.markdown("""
        <style>
        .title {
            font-size: 3em;
            font-weight: bold;
            color: #003262;
            text-align: center;
            margin-bottom: 0.3em;
        }
        .subtitle {
            font-size: 1.2em;
            color: #4d4d4d;
            text-align: center;
            margin-bottom: 1em;
        }
        .metric-card {
            background-color: #f9f9f9;
            padding: 1em;
            border-radius: 1rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title"> Accidents de la route en France</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyse géographique interactive par gravité et année</div>', unsafe_allow_html=True)

    # Chargement des données
  
    accidents = pd.read_csv(accidents_path)

    # Nettoyage des lat/lon
    accidents['lat'] = accidents['lat'].astype(str).str.replace(',', '.')
    accidents['long'] = accidents['long'].astype(str).str.replace(',', '.')
    accidents['lat'] = accidents['lat'].astype(float)
    accidents['long'] = accidents['long'].astype(float)

    # Extraction de l’année
    accidents['Num_Acc'] = accidents['Num_Acc'].astype(str)
    accidents['an'] = accidents['Num_Acc'].str[:4].astype(int)

    # Barre latérale : sélection de l'année
    year_selected = st.sidebar.selectbox("📅 Sélectionnez une année", sorted(accidents['an'].unique()), index=len(accidents['an'].unique())-1)
    df_year = accidents[accidents['an'] == year_selected].copy()

    # Département : on zfill au besoin
    if 'dep' in df_year.columns:
        df_year['dep'] = df_year['dep'].astype(str).str.zfill(2)

    # Analyse variation par département
    acc_by_year_dep = accidents.groupby(['an', 'dep']).size().unstack(fill_value=0)
    if year_selected > min(acc_by_year_dep.index):
        variation = acc_by_year_dep.diff().loc[year_selected]
        dep_max_increase = variation.idxmax()
        dep_max_decrease = variation.idxmin()
        col1, col2 = st.columns(2)
        col1.metric("📈 Département en hausse", f"{dep_max_increase}", f"+{variation[dep_max_increase]}")
        col2.metric("📉 Département en baisse", f"{dep_max_decrease}", f"{variation[dep_max_decrease]}")
    else:
        st.info("Pas de données pour l'année précédente pour comparer les départements.")

    # Création des cellules géographiques
    taille_cellule = 0.05  # ≈ 5 km
    df_year = df_year.dropna(subset=['lat', 'long']).copy()
    df_year['cell_x'] = (df_year['long'] // taille_cellule).astype(int)
    df_year['cell_y'] = (df_year['lat'] // taille_cellule).astype(int)

    grille = (
        df_year
        .groupby(['cell_x', 'cell_y'])
        .agg(
            nb_accidents=('grav', 'count'),
            grav_mediane=('grav', 'median'),
            lat_centre=('lat', 'mean'),
            long_centre=('long', 'mean')
        )
        .reset_index()
    )

    couleurs = {
        1: 'green',
        2: 'orange',
        3: 'red',
        4: 'darkred'
    }

    # Création de la carte Folium
    m = folium.Map(location=[46.6, 2.2], zoom_start=6)

    for _, row in grille.iterrows():
        grav = int(row['grav_mediane'])
        couleur = couleurs.get(grav, 'gray')
        Circle(
            location=[row['lat_centre'], row['long_centre']],
            radius=(row['nb_accidents']) * (np.exp(((grav - 1) / 2) ** (grav + 1))),
            color=couleur,
            fill=True,
            fill_color=couleur,
            fill_opacity=0.6,
            popup=folium.Popup(f"{row['nb_accidents']} accidents<br>Gravité médiane : {grav}", max_width=300)
        ).add_to(m)

    # Affichage Streamlit
    st.markdown("### 🗺️ Carte interactive des accidents")
    st_data = st_folium(m, width=1000, height=600)

    st.markdown("---")
    st.caption("© Données accidents – projet DataScientest")

