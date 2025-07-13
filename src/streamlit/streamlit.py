import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

usagers_2019 = pd.read_csv("data/raw/usagers-2019.csv", sep=';')
caract_2019 = pd.read_csv("data/raw/caracteristiques-2019.csv", sep=';')
lieux_2019 = pd.read_csv("data/raw/lieux-2019.csv", sep=';')
vehicules_2019 = pd.read_csv("data/raw/vehicules-2019.csv", sep=';')
df_brut = pd.read_csv("data/raw/accidents_2019_2023.csv")
df = pd.read_csv("../data/processed/accidents_clean.csv")

#titre et sommaire sur le cote
st.title("Projet de prévision des accidents routiers")
st.sidebar.title("Sommaire")
pages=["Exploration de données", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

# Exploration de données
# Page Exploration de données
if page == pages[0] : 
  st.write("### Introduction")
  st.write("Ce projet a pour but de prédire les accidents routiers en France. Nous allons explorer les données, visualiser les tendances et construire des modèles de prévision.")    
  # Presentation des 4 types de fichiers usagers, caractéristiques, véhicules et lieux
  st.write(" Nous sommes parties de 4 catégories de fichiers, Usagers, Caractéristiques, Véhicules et  Lieux, que nous avons fusionnées pour obtenir un jeu de données complet. Chaque fichier allant de 2019 à 2023 ")
  if st.checkbox("Visualiser usagers"):
    st.dataframe(usagers_2019.head(10))
  if st.checkbox("Visualiser véhicules"):
    st.dataframe(vehicules_2019.head(10))  
  if st.checkbox("Visualiser lieux"):
    st.dataframe(lieux_2019.head(10))
  if st.checkbox("Visualiser caractéristiques"):
    st.dataframe(caract_2019.head(10))
  if st.checkbox("Visualiser les données brutes"):
    st.dataframe(df_brut.head(10))
  
  