import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Titre principal
st.title("🚗 Analyse et Prédiction de la Gravité des Accidents de la Route")

# Présentation du projet
st.markdown("""
### 🎯 Objectif

Ce projet vise à **analyser et prédire la gravité des accidents corporels de la route en France** à partir de données ouvertes de la sécurité routière.

### ❓ Problématique

> Peut-on prédire la **gravité** (indemne, blessé léger, blessé hospitalisé, décès) d’un accident routier à partir de ses **caractéristiques contextuelles**, des **véhicules** impliqués et des **profils d’usagers** ?

### 📌 Enjeux

- 🧠 **Métiers** : Aide à la décision pour les politiques publiques (infrastructures, prévention, sécurité routière).
- ⛑️ **Opérationnels** : Anticipation des risques graves → répartition optimisée des secours.
- 📍 **Géospatiaux** : Identification des zones à haut risque → cartographie proactive.
- 📊 **Techniques** : Modèle multiclasses avec fort déséquilibre → challenge en machine learning.

### 🗂 Source

[Base BAAC - data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023)
""")

# Présentation des données
st.markdown("## 📦 Données utilisées")

st.markdown("""
La base est composée de **plusieurs fichiers annuels (2005 à 2023)**, structurés en **quatre tables relationnelles** par année :

| Fichier        | Description                                      |
|----------------|--------------------------------------------------|
| `caracteristiques` | Info sur l’accident (lieu, heure, météo…)     |
| `vehicules`        | Infos sur chaque véhicule impliqué            |
| `usagers`          | Infos sur les personnes impliquées            |
| `lieux`            | Coordonnées géographiques                     |

### 🔢 Volumétrie (2023)

- `caracteristiques` : ~55 000 accidents
- `vehicules` : ~80 000 véhicules
- `usagers` : ~100 000 personnes

### 🧱 Schéma relationnel
""")

image = Image.open('/Users/alizeeblanchon/Documents/Data_Scientist/data_project/mai25_bds_accidents/reports/figures/Jointure tables.png')
st.image(image, caption="Structure relationnelle des données (clé : Num_Acc)")

st.markdown("""
### ⚠️ Problèmes rencontrés :

- Fichiers séparés par année → concaténation nécessaire
- Encodage différent avant/après 2018
- Données parfois mal encodées (`grav` non ordonnée, valeurs manquantes, valeurs aberrantes)
- Données très déséquilibrées : **<1% de décès**
""")

# Variable cible
st.markdown("## 🎯 Variable cible : `grav` (Gravité de l'accident)")

st.markdown("""
La variable `grav` (issue du fichier `usagers.csv`) représente la **gravité de l'atteinte corporelle** de l'usager :

| Code | Gravité              |
|------|----------------------|
| 1    | Indemne              |
| 2    | Blessé léger         |
| 3    | Blessé hospitalisé   |
| 4    | Tué                  |

Elle est utilisée comme **variable cible** dans les modèles de classification multi-classes.
""")

# Graphe de répartition des classes de gravité
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "data", "raw"))
DATA_PROCESSED_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "data", "processed"))
accidents_path = os.path.join(DATA_RAW_DIR, "accidents_2019_2023.csv")
df = pd.read_csv(accidents_path)

# Filtrage des gravités valides
# Le fichier accidents_2019_2023 n'est pas clean
# Filtrer les valeurs valides
df_valid = df[df['grav'].isin([1, 2, 3, 4])]
# Redéfinir l’ordre logique
order = [2, 3, 4, 1]
labels = ['Tué', 'Hospitalisé', 'Léger', 'Indemne']

# Calcul des proportions dans l’ordre voulu
grav_counts = df_valid['grav'].value_counts(normalize=True).reindex(order)
# Tracé du graphique
fig, ax = plt.subplots()
grav_counts.plot(kind='bar', color='salmon', ax=ax)
# Labels corrects
ax.set_xticks(range(len(order)))
ax.set_xticklabels(labels, rotation=0)
ax.set_ylabel("Proportion")
ax.set_title("Répartition des classes de gravité")

# Affichage Streamlit
st.pyplot(fig)

# Variables explicatives
st.markdown("## 🧠 Sélection des variables explicatives")

st.markdown("""
Les variables retenues sont sélectionnées en fonction :
- de leur lien plausible avec la gravité (connaissance métier)
- de leur disponibilité dans les données
- de leur pertinence statistique (analyses corrélatives ultérieures)

### 👤 Caractéristiques des usagers
- `catu` : Catégorie de l’usager (conducteur, piéton, passager)

### 🚗 Type de véhicule
- `catv` : Catégorie du véhicule (VL, moto, PL, vélo…)

### 🌧️ Conditions environnementales
- `atm` : Atmosphère (pluie, brouillard…)
- `lum` : Luminosité (jour, nuit, éclairé ou non)
- `int` : Type d’intersection
- `col` : Type de collision
- `surf` : État de la chaussée

### 📆 Dimensions temporelles
- `an` : Année
- `mois` : Mois
- `hrmn` : Heure (arrondie ou extraite à partir de `hrmn`)

### 🌍 Informations géographiques
- `dep` : Département
- `com` : Commune
- `agg` : Type d’agglomération
- `lat`, `long` : Coordonnées géographiques
""")

st.info("💡 Ces variables seront analysées en détail dans la phase exploratoire et utilisées comme entrées pour les modèles de classification.")
