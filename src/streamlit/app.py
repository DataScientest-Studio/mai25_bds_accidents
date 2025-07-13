import streamlit as st
from pages import exploration, dataviz, modelisation, deeplearning 

@st.cache_data
st.set_page_config(page_title="Projet Accidents routiers", layout="wide")

st.sidebar.title("Sommaire")
page = st.sidebar.radio("Aller vers", ["Exploration des données", "Visualisation", "Modélisation","Deep Learning"])

if page == "Exploration des données":
    exploration.run()

elif page == "Visualisation":
    dataviz.run()

elif page == "Modélisation":
    modelisation.run()

elif page == "Deep Learning":
    deeplearning.run()
