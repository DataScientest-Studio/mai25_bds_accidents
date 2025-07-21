import streamlit as st
import sys

# Configuration en tout premier
st.set_page_config(page_title="Projet Accidents routiers", layout="wide")

# Import avec gestion d'erreur
try:
    from onglet import exploration, dataviz, modelisation, deeplearning
except Exception as e:
    st.sidebar.error(f"Erreur d'import : {e}")

# Barre latérale
st.sidebar.title("Sommaire")
page = st.sidebar.radio(
    "Aller vers",
    [
        "Exploration des données",
        "Visualisation",
        "Modélisation",
        "Deep Learning"
    ],
    index=0  # optionnel
)

# Routing
if page == "Exploration des données":
    exploration.run()
elif page == "Visualisation":
    dataviz.run()
elif page == "Modélisation":
    modelisation.run()
elif page == "Deep Learning":
    deeplearning.run()
