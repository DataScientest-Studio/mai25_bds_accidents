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
accidents_path = os.path.join(DATA_RAW_DIR, "accidents_2019_2023.csv")

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
    st.markdown("### Jeu de données fusionné avant netttoyage")
    st.dataframe(df.head(5))

    # Informations générales détaillées avec détection des variables catégorielles et nombre de catégories
    st.markdown("### Informations générales détaillées")

    infos = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isna().sum()

        unique_vals = df[col].dropna().unique()

        # Détection des variables catégorielles : valeurs numériques comprises entre -1 et 99
        if pd.api.types.is_numeric_dtype(df[col]) and np.all((unique_vals >= -1) & (unique_vals <= 99)):
            is_cat = "Oui"
            nb_categories = df[col].nunique(dropna=True)
        else:
            is_cat = "Non"
            nb_categories = "-"

        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
        else:
            min_val = "-"
            max_val = "-"
            mean_val = "-"

        infos.append({
            "Colonne": col,
            "Type": dtype,
            "Catégorielle": is_cat,
            "Nb catégories": nb_categories,
            "Valeurs manquantes": missing,
            "Min": min_val,
            "Max": max_val,
            "Moyenne": mean_val
        })

    infos_df = pd.DataFrame(infos)
    st.dataframe(infos_df)



    # Filtre dynamique
    st.markdown("### Filtrer une colonne")
    colonne = st.selectbox("Choisissez une colonne pour explorer sa distribution", df.columns)
    if df[colonne].dtype in [int, float]:
        fig, ax = plt.subplots()
        sns.histplot(df[colonne], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.write(df[colonne].value_counts())

        # Corrélation avec 'grav'
    st.markdown("### Corrélation avec la variable 'grav'")

    if "grav" not in df.columns:
        st.warning("La variable 'grav' n'existe pas dans le jeu de données.")
    else:
        numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != "grav"]

        selected_cols = st.multiselect(
            "Sélectionnez les variables numériques à corréler avec 'grav'",
            options=numeric_cols,
            default=numeric_cols[:5]  # pré-sélectionne les 5 premières
        )

        if selected_cols:
            corr_df = df[["grav"] + selected_cols].corr()

            fig_corr, ax_corr = plt.subplots(figsize=(1 + len(selected_cols), 5))
            sns.heatmap(corr_df[["grav"]].loc[selected_cols], annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
            ax_corr.set_title("Corrélation des variables sélectionnées avec 'grav'")
            st.pyplot(fig_corr)
        else:
            st.info("Veuillez sélectionner au moins une variable.")

