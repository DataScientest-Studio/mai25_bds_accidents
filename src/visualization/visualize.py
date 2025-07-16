
import os
import pandas as pd
import folium
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
path1 = os.path.join( "..", "data", "processed", "accidents_2019_2023.csv")
accidents= pd.read_csv(path1)
path2 = os.path.join( "..", "data", "processed", "accidents_clean.csv")
accidents_clean= pd.read_csv(path2)


# Visualisation de la gravité des accidents
accidents['Num_Acc']=accidents['Num_Acc'].astype(str)
accidents['an']= accidents['Num_Acc'].str[:4]
accidents['an']=accidents['an'].astype(int)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Histogramme normalisé pour 2004–2018
sns.histplot(
    accidents_1['grav'], 
    bins=10, 
    stat="density",  # Normalisation
    kde=False, 
    color='steelblue', 
    ax=axes[0]
)
axes[0].set_title("Gravité des accidents (2004–2018)")
axes[0].set_xlabel("Gravité")
axes[0].set_ylabel("Densité")
axes[0].grid(True)

# Histogramme normalisé pour 2019–2023
sns.histplot(
    accidents_2['grav'], 
    bins=10, 
    stat="density",  # Normalisation
    kde=False, 
    color='darkorange', 
    ax=axes[1]
)
axes[1].set_title("Gravité des accidents (2019–2023)")
axes[1].set_xlabel("Gravité")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Evolution hebdomadaire du nombre d'accidents par gravité
accidents = accidents[accidents['grav'] != -1]
accidents['grav'] = accidents['grav'].replace({2: 42})
accidents['grav'] = accidents['grav'].replace({4: 2})
accidents['grav'] = accidents['grav'].replace({42: 4})

accidents['datetime_string'] = (
    accidents['an'].astype(str) + '-' +
    accidents['mois'].astype(str).str.zfill(2) + '-' +
    accidents['jour'].astype(str).str.zfill(2)
    
)
accidents['date'] = pd.to_datetime(accidents['datetime_string'], format='%Y-%m-%d')
accidents = accidents.drop('datetime_string',axis=1)
accidents['date'] = pd.to_datetime(accidents['date'], errors='coerce')
accidents['week'] = accidents['date'].dt.to_period('W').dt.start_time
accidents_weekly = accidents.groupby(['week', 'grav']).size().reset_index(name='counts')
plt.figure(figsize=(12, 6))
sns.lineplot(data=accidents_weekly, x='week', y='counts', hue='grav', errorbar=('ci', 95), palette='viridis')
plt.title("Évolution hebdomadaire du nombre d'accidents par gravité")
plt.xlabel("Semaine")
plt.ylabel("Nombre d'accidents")

plt.show()

# Visualisation de la répartition des accidents par tranche d'âge
plt.figure(figsize=(10, 6))
sns.histplot(data=accidents_clean, x='age', hue='grav', kde=True, multiple='stack', palette='viridis')
plt.title(f"Distribution de l'age en fonction de la gravité de l'accident")
plt.savefig("../reports/figures/1.0-Seb-age_gravite.png")
plt.show()




# Cluster Geo
taille_cellule = 0.05  # ~5 km

accidents_clean = accidents_clean.dropna(subset=['lat', 'long']).copy()
accidents_clean['cell_x'] = (accidents_clean['long'] // taille_cellule).astype(int)
accidents_clean['cell_y'] = (accidents_clean['lat'] // taille_cellule).astype(int)

grille = (
    accidents_clean
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


m = folium.Map(location=[46.6, 2.2], zoom_start=6)
for _, row in grille.iterrows():
    grav = int(row['grav_mediane'])
    couleur = couleurs.get(grav, 'gray')
    
    folium.Circle(
        location=[row['lat_centre'], row['long_centre']],
        # radius = np.sqrt(row['nb_accidents']) * (grav),
        radius= (row['nb_accidents'])*(np.exp(((grav-1)/2)**(grav+1))),  # rayon dépendant du nombre d'accidents pondéré par la gravité
        color=couleur,
        fill=True,
        fill_color=couleur,
        fill_opacity=0.6,
        popup=f"{row['nb_accidents']} accidents<br>Gravité médiane : {grav}"
    ).add_to(m)

m