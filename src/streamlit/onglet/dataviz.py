import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import folium
from folium import Circle
from streamlit_folium import st_folium

# Définition des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "data", "raw"))
DATA_PROCESSED_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "data", "processed"))

accidents_path = os.path.join(DATA_RAW_DIR, "accidents_2019_2023.csv")


def run():
    # Style et titre
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

    st.markdown('<div class="title">Accidents de la route en France</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyse géographique interactive par gravité et année</div>', unsafe_allow_html=True)

    # Chargement des données
    accidents = pd.read_csv(accidents_path)
    #Clean des departement pour etre sur
    accidents['dep'] = accidents['dep'].astype(str).str.zfill(2)


    # Extraction de l’année
    accidents['Num_Acc'] = accidents['Num_Acc'].astype(str)
    accidents['an'] = accidents['Num_Acc'].str[:4].astype(int)

    # Sélection de l'année via la barre latérale
    year_selected = st.sidebar.selectbox("Sélectionnez une année", sorted(accidents['an'].unique()), index=len(accidents['an'].unique()) - 1)
    df_year = accidents[accidents['an'] == year_selected].copy()


    # METRIQUES DE VARIATION
    acc_by_year_dep = accidents.groupby(['an', 'dep']).size().unstack(fill_value=0)
    if year_selected > min(acc_by_year_dep.index):
        variation = acc_by_year_dep.diff().loc[year_selected]
        dep_max_increase = variation.idxmax()
        dep_max_decrease = variation.idxmin()
        col1, col2 = st.columns(2)
        col1.metric(" Département en hausse", f"{dep_max_increase}", f"+{variation[dep_max_increase]}")
        col2.metric("Département en baisse", f"{dep_max_decrease}", f"{variation[dep_max_decrease]}")
    else:
        st.info("Pas de données pour l'année précédente pour comparer les départements.")

    # CRÉATION DE LA CARTE
    taille_cellule = 0.05
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

    couleurs = {1: 'green', 2: 'orange', 3: 'red', 4: 'darkred'}

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

    # AFFICHAGE INTERFACE
    col_map, col_top = st.columns([3, 1])

    with col_map:
        st.markdown("### Carte interactive des accidents")
        st_data = st_folium(m, width=900, height=600)

    with col_top:
        st.markdown("### Départements les plus accidentogènes")
        top_dep = df_year['dep'].value_counts().head(10).reset_index()
        top_dep.columns = ['Département', 'Nombre d\'accidents']
        st.dataframe(top_dep, use_container_width=True)

        # 📊 RÉPARTITION DES CLASSES DE GRAVITÉ
    
    st.markdown("### 📊 Répartition des classes de gravité")

    fig1, ax1 = plt.subplots()
    grav_counts = df_year['grav'].value_counts().sort_index()
    grav_counts.plot(kind='bar', color='salmon', ax=ax1)

    # Corriger les ticks pour coller aux indices réels
    ax1.set_xticks(grav_counts.index)
    ax1.set_xticklabels(['Indemne', 'Blessé léger', 'Hospitalisé', 'Décès'], rotation=0)
    ax1.set_ylabel("Nombre d'accidents")
    ax1.set_title(f"Gravité des accidents en {year_selected}")
    st.pyplot(fig1)


    # ⏰ GRAVITÉ MOYENNE SELON L'HEURE DE LA JOURNÉE
    if 'hrmn' in df_year.columns:
        st.markdown("### ⏰ Gravité moyenne selon l'heure de la journée")

        # Extraction de l'heure si elle est au format HHMM
        df_year['heure'] = df_year['hrmn'].astype(str).str.zfill(4).str[:2].astype(int)
        gravite_par_heure = df_year.groupby('heure')['grav'].mean()

        fig2, ax2 = plt.subplots()
        sns.lineplot(x=gravite_par_heure.index, y=gravite_par_heure.values, marker='o', ax=ax2, color='darkblue')
        ax2.set_xlabel("Heure")
        ax2.set_ylabel("Gravité moyenne")
        ax2.set_title(f"Gravité moyenne par heure de la journée ({year_selected})")
        ax2.set_xticks(range(0, 24))
        st.pyplot(fig2)
    else:
        st.info("La colonne 'hrmn' n'est pas disponible pour analyser la gravité par heure.")


    st.markdown("---")

    st.caption("© Données accidents – projet DataScientest")

