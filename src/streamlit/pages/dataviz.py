import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run():
    st.title("Visualisation des données")
    st.markdown("## Analyse de l'âge des conducteurs impliqués dans les accidents")

    # Chargement des données nettoyées
    try:
        df = pd.read_csv("data/processed/accidents_clean.csv")
    except FileNotFoundError:
        st.error("Le fichier 'accidents_clean.csv' est introuvable.")
        return

    # Calcul de l'âge
    annee_ref = 2025
    if "an_nais" not in df.columns:
        st.error("La colonne 'an_nais' est absente du fichier.")
        return

    df["age"] = annee_ref - df["an_nais"]

    # Regrouper les âges en tranches
    bins = [0,18,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,140]
    labels = ["0-18","18-25","25-30","30-35","35-40","40-45","45-50","50-55",
              "55-60","60-65","65-70","70-75","75-80","80-85","85-90","90-95","95+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    # Graphique : nombre d'accidents par tranche d'âge
    st.markdown("### Nombre d'accidents par tranche d'âge")
    accidents_par_groupe = df["age_group"].value_counts().sort_index()

    fig1, ax1 = plt.subplots(figsize=(12,6))
    accidents_par_groupe.plot(kind="bar", ax=ax1)
    ax1.set_title("Nombre d'accidents selon la tranche d'âge")
    ax1.set_xlabel("Tranche d'âge")
    ax1.set_ylabel("Nombre d'accidents")
    st.pyplot(fig1)

    # Graphique : violon des âges selon la gravité
    if "grav" in df.columns:
        st.markdown("### Distribution des âges selon la gravité de l'accident")
        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.violinplot(data=df, x="grav", y="age", palette="muted", ax=ax2)
        ax2.set_title("Distribution de l'âge selon la gravité")
        ax2.set_xlabel("Gravité")
        ax2.set_ylabel("Âge")
        st.pyplot(fig2)
    else:
        st.warning("La colonne 'grav' n'est pas présente pour tracer le graphique de distribution.")

