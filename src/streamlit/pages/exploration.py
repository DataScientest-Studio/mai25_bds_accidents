import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    st.title("Exploration des données")
    st.markdown("### Chargement des données")

    # Chargement
    try:
        usagers = pd.read_csv(usagers_path, sep=";")
        vehicules = pd.read_csv(vehicules_path, sep=";")
        lieux = pd.read_csv(lieux_path, sep=";")
        caract = pd.read_csv(caracteristiques_path, sep=";")
        df = pd.read_csv(accidents_path)
    except FileNotFoundError:
        st.error(usagers_path)
        st.error("Fichiers manquants dans 'data/raw' ou 'data/processed'.")
        return

    # Menu de visualisation des fichiers bruts
    st.markdown("### Aperçu des fichiers bruts")
    fichier = st.selectbox("Sélectionnez un fichier brut à explorer", ["usagers", "vehicules", "lieux", "caracteristiques"])
    if fichier == "usagers":
        st.dataframe(usagers.head(10))
    elif fichier == "vehicules":
        st.dataframe(vehicules.head(10))
    elif fichier == "lieux":
        st.dataframe(lieux.head(10))
    elif fichier == "caracteristiques":
        st.dataframe(caract.head(10))

    # Aperçu du jeu de données fusionné
    st.markdown("### Jeu de données fusionné et nettoyé")
    st.dataframe(df.head(10))

    # Infos générales
    st.markdown("### Informations générales")
    st.write("Nombre de lignes :", df.shape[0])
    st.write("Nombre de colonnes :", df.shape[1])
    st.write("Colonnes :", list(df.columns))
    st.write("Types de variables :")
    st.dataframe(df.dtypes.astype(str))

    # Valeurs manquantes
    st.markdown("### Valeurs manquantes")
    nan_df = df.isnull().sum()
    nan_df = nan_df[nan_df > 0]
    if not nan_df.empty:
        st.write(nan_df)
    else:
        st.success("Aucune valeur manquante.")

    # Filtre dynamique
    st.markdown("### Filtrer une colonne")
    colonne = st.selectbox("Choisissez une colonne pour explorer sa distribution", df.columns)
    if df[colonne].dtype in [int, float]:
        fig, ax = plt.subplots()
        sns.histplot(df[colonne], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.write(df[colonne].value_counts())

    # Corrélation (si numérique)
    st.markdown("### Matrice de corrélation")
    num_cols = df.select_dtypes(include=["int", "float"])
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(num_cols.corr(), cmap="coolwarm", annot=True, fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)
