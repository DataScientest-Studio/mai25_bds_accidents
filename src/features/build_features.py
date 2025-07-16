import pandas as pd
import numpy as np  

# Creation fichier merge 
# 2019
usagers_2019 = pd.read_csv("data/raw/usagers-2019.csv", sep=';')
caract_2019 = pd.read_csv("data/raw/caracteristiques-2019.csv", sep=';')
lieux_2019 = pd.read_csv("data/raw/lieux-2019.csv", sep=';')
vehicules_2019 = pd.read_csv("data/raw/vehicules-2019.csv", sep=';')

# 2020
usagers_2020 = pd.read_csv("data/raw/usagers-2020.csv", sep=';')
caract_2020 = pd.read_csv("data/raw/caracteristiques-2020.csv", sep=';')
lieux_2020 = pd.read_csv("data/raw/lieux-2020.csv", sep=';')
vehicules_2020 = pd.read_csv("data/raw/vehicules-2020.csv", sep=';')

#2021
usagers_2021 = pd.read_csv("data/raw/usagers-2021.csv", sep=';')
caract_2021 = pd.read_csv("data/raw/caracteristiques-2021.csv", sep=';')
lieux_2021 = pd.read_csv("data/raw/lieux-2021.csv", sep=';')
vehicules_2021 = pd.read_csv("data/raw/vehicules-2021.csv", sep=';')

# 2022
usagers_2022 = pd.read_csv("data/raw/usagers-2022.csv", sep=';')
caract_2022 = pd.read_csv("data/raw/caracteristiques-2022.csv", sep=';')
lieux_2022 = pd.read_csv("data/raw/lieux-2022.csv", sep=';')
vehicules_2022 = pd.read_csv("data/raw/vehicules-2022.csv", sep=';')

# 2023
usagers_2023 = pd.read_csv("data/raw/usagers-2023.csv", sep=';')
caract_2023 = pd.read_csv("data/raw/caracteristiques-2023.csv", sep=';')
lieux_2023 = pd.read_csv("data/raw/lieux-2023.csv", sep=';')
vehicules_2023 = pd.read_csv("data/raw/vehicules-2023.csv", sep=';')

caract_2022 = caract_2022.rename(columns={'Accident_Id': 'Num_Acc'})
caract_lieux_2023 = caract_2023.merge(lieux_2023, on='Num_Acc')
vehicules_usagers_2023 = vehicules_2023.merge(usagers_2023, on=['Num_Acc','id_vehicule','num_veh'], how='inner')
accidents_2023 = vehicules_usagers_2023.merge(caract_lieux_2023, on='Num_Acc')
caract_lieux_2022 = caract_2022.merge(lieux_2022, on='Num_Acc')
vehicules_usagers_2022 = vehicules_2022.merge(usagers_2022, on=['Num_Acc','id_vehicule','num_veh'], how='inner')
accidents_2022 = vehicules_usagers_2022.merge(caract_lieux_2022, on='Num_Acc')
caract_lieux_2021 = caract_2021.merge(lieux_2021, on='Num_Acc')
vehicules_usagers_2021 = vehicules_2021.merge(usagers_2021, on=['Num_Acc','id_vehicule','num_veh'], how='inner')
accidents_2021 = vehicules_usagers_2021.merge(caract_lieux_2021, on='Num_Acc')
caract_lieux_2020 = caract_2020.merge(lieux_2020, on='Num_Acc')
vehicules_usagers_2020 = vehicules_2020.merge(usagers_2020, on=['Num_Acc','id_vehicule','num_veh'], how='inner')
accidents_2020 = vehicules_usagers_2020.merge(caract_lieux_2020, on='Num_Acc')
caract_lieux_2019 = caract_2019.merge(lieux_2019, on='Num_Acc')
vehicules_usagers_2019 = vehicules_2019.merge(usagers_2019, on=['Num_Acc','id_vehicule','num_veh'], how='inner')
accidents_2019 = vehicules_usagers_2019.merge(caract_lieux_2019, on='Num_Acc')
accidents = pd.concat([accidents_2019,accidents_2020,accidents_2021,accidents_2022,accidents_2023], axis=0, join='inner', ignore_index=True)

# Sauvegarde fichier merger
accidents.to_csv("data/raw/accidents_2019_2023.csv", index=False)

# Nettoyage du fichier
# Suppression des doublons
accidents = accidents.drop_duplicates(subset=['Num_Acc','id_vehicule','num_veh','place','grav'])

# Conversion en float des colonnes lat et long
accidents['lat'] = accidents['lat'].str.replace(',', '.')
accidents['long'] = accidents['long'].str.replace(',', '.')
accidents['lat'] = accidents['lat'].astype(float)
accidents['long'] = accidents['long'].astype(float)

# changement ordre de gravité pour avoir un ordre logique 
accidents['grav'] = accidents['grav'].replace({2: 42})
accidents['grav'] = accidents['grav'].replace({4: 2})
accidents['grav'] = accidents['grav'].replace({42: 4})

# Suppression des colonnes inutiles
accidents_copy = accidents.drop(['occutc','secu2','secu3','locp','actp','etatp','v1','v2','vosp','lartpc','larrout'],axis=1)

# création d'une variable age
accidents_copy['age'] = accidents_copy['an']-accidents_copy['an_nais']
accidents_copy.drop(['an_nais'],axis=1,inplace = True)

# suppression des valeurs manquantes pour la variable cible
accidents_copy = accidents_copy[accidents_copy['grav'] != -1]

# Création de la variable date
accidents_copy['datetime_string'] = (
    accidents_copy['an'].astype(str) + '-' +
    accidents_copy['mois'].astype(str).str.zfill(2) + '-' +
    accidents_copy['jour'].astype(str).str.zfill(2) + ' ' +
    accidents_copy['hrmn'].astype(str)
)
accidents_copy['date'] = pd.to_datetime(accidents_copy['datetime_string'], format='%Y-%m-%d %H:%M')
accidents_copy = accidents_copy.drop('datetime_string',axis=1)

# Suppression des colonnes inutiles
accidents_copy = accidents_copy.drop(['pr','pr1','senc'],axis=1)
accidents_copy = accidents_copy.drop(['voie','adr'],axis=1)




# remplacement des valeurs manquante de sexe suivant leur proportion dans la catégorie
# grav = 1 (toutes les valeurs manquantes sont dans cette catégorie)
proportion = [1] * 71 + [2] * 29
n_manquants = (accidents_copy['sexe'] == -1).sum()
valeurs_remplacement = np.random.choice(proportion, size=n_manquants, replace=True)
accidents_copy.loc[accidents_copy['sexe'] == -1, 'sexe'] = valeurs_remplacement


# remplacement des valeurs manquantes par le mode dans les variables ou c'est possible 
accidents_copy['circ'] = accidents_copy['circ'].replace(-1, accidents_copy['circ'].mode()[0])
accidents_copy['trajet'] = accidents_copy['trajet'].replace(-1, accidents_copy['trajet'].mode()[0])
accidents_copy['secu1'] = accidents_copy['secu1'].replace(-1, accidents_copy['secu1'].mode()[0])
accidents_copy['infra'] = accidents_copy['infra'].replace(-1, accidents_copy['infra'].mode()[0])
accidents_copy['col'] = accidents_copy['col'].replace(-1, accidents_copy['col'].mode()[0])
accidents_copy['situ'] = accidents_copy['situ'].replace(-1, accidents_copy['situ'].mode()[0])
accidents_copy['surf'] = accidents_copy['surf'].replace(-1, accidents_copy['surf'].mode()[0])
accidents_copy['prof'] = accidents_copy['prof'].replace(-1, accidents_copy['prof'].mode()[0])
accidents_copy['plan'] = accidents_copy['plan'].replace(-1, accidents_copy['plan'].mode()[0])
accidents_copy['choc'] = accidents_copy['choc'].replace(-1, accidents_copy['choc'].mode()[0])
accidents_copy['obsm'] = accidents_copy['obsm'].replace(-1, accidents_copy['obsm'].mode()[0])
accidents_copy['obs'] = accidents_copy['obs'].replace(-1, accidents_copy['obs'].mode()[0])
accidents_copy['atm'] = accidents_copy['atm'].replace(-1, accidents_copy['atm'].mode()[0])
accidents_copy['int'] = accidents_copy['int'].replace(-1, accidents_copy['int'].mode()[0])
accidents_copy['place'] = accidents_copy['place'].replace(-1, accidents_copy['place'].mode()[0])

# fonction pour ramplacer des valeurs manquantes dans age par une valeur aléatoire parmis 
# les valeurs observées en respectant la distribution observée

def replace_nan_with_random(df, column_name):
    observed_values = df[column_name].dropna()
    random_values = np.random.choice(observed_values, size=df[column_name].isna().sum())
    df.loc[df[column_name].isna(), column_name] = random_values
    return df
# Remplacer les NaN dans 'age'
accidents_copy = replace_nan_with_random(accidents_copy, 'age')
# remplace les valeurs abérantes > 105 ans
accidents_copy['age'] = accidents_copy['age'].replace({106: 6})
accidents_copy['age'] = accidents_copy['age'].replace({108: 8})
accidents_copy['age'] = accidents_copy['age'].replace({109: 9})
accidents_copy['age'] = accidents_copy['age'].replace({110: 10})
accidents_copy['age'] = accidents_copy['age'].replace({118: 18})
accidents_copy['age'] = accidents_copy['age'].replace({119: 19})
accidents_copy['age'] = accidents_copy['age'].replace({120: 20})

# remplace les -1 par les valeurs de la catégorie 'autre' ou 'inconnu'
accidents_copy['motor'] = accidents_copy['motor'].replace({-1: 6})
accidents_copy['manv'] = accidents_copy['manv'].replace({-1: 0})
accidents_copy['catv'] = accidents_copy['catv'].replace({-1: 99})

# Traitement de la variable nbv
valeurs_invalides = ['#ERREUR', '#VALEURMULTI', -1, '-1', ' -1']
accidents_copy['nbv'] = accidents_copy['nbv'].replace(valeurs_invalides, np.nan)
accidents_copy['nbv'] = pd.to_numeric(accidents_copy['nbv'], errors='coerce')
accidents_copy['nbv'] = accidents_copy['nbv'].fillna(accidents_copy['nbv'].mode()[0])

# remplacement des valeurs manquantes de lum en fonction des heures de la journée
# Remplacer les -1 par 1 entre 9h et 17h
accidents_copy.loc[(accidents_copy['lum'] == -1) &
                   (accidents_copy['hrmn'] > '09:00') &
                   (accidents_copy['hrmn'] < '17:00'), 'lum'] = 1
# Remplacer les -1 par 3 avant 7h ou après 20h
accidents_copy.loc[(accidents_copy['lum'] == -1) &
                   ((accidents_copy['hrmn'] < '07:00') |
                    (accidents_copy['hrmn'] > '20:00')), 'lum'] = 3
# Remplacer les -1 restants par 2
accidents_copy['lum'] = accidents_copy['lum'].replace({-1: 2})


# remplacement des valeurs abérrantes vma
valeurs_50 = [500, 55, 520, 501, 502, 5]
valeurs_90 = [900, 901, 9]
valeurs_80 = [800, 180, 8]
valeurs_70 = [7, 75, 700, 770]
valeurs_60 = [560, 600]
valeurs_30 = [3, 300, 31, 35]
valeurs_40 = [140, 4, 42]

accidents_copy['vma'] = accidents_copy['vma'].replace(valeurs_50, 50)
accidents_copy['vma'] = accidents_copy['vma'].replace(valeurs_70, 70)
accidents_copy['vma'] = accidents_copy['vma'].replace(valeurs_80, 80)
accidents_copy['vma'] = accidents_copy['vma'].replace(valeurs_90, 90)
accidents_copy['vma'] = accidents_copy['vma'].replace(valeurs_60, 60)
accidents_copy['vma'] = accidents_copy['vma'].replace(valeurs_30, 30)
accidents_copy['vma'] = accidents_copy['vma'].replace(valeurs_40, 40)


# remplacement des -1 en fonction de la catégorie de route

accidents_copy.loc[(accidents_copy['vma'] == -1) &
                   (accidents_copy['catr'] == 1), 'vma'] = 130

accidents_copy.loc[(accidents_copy['vma'] == -1) &
                   (accidents_copy['catr'] == 2), 'vma'] = 80

accidents_copy.loc[(accidents_copy['vma'] == -1) &
                   (accidents_copy['catr'] == 6), 'vma'] = 30

accidents_copy['vma'] = accidents_copy['vma'].replace({-1: 50})

accidents_copy.loc[accidents_copy['vma'] < 30, 'vma'] = 30


# regroupement des catégories rares avec une catégorie proche

accidents_copy['vma'] = accidents_copy['vma'].replace({40: 50})
accidents_copy['vma'] = accidents_copy['vma'].replace({45: 50})
accidents_copy['vma'] = accidents_copy['vma'].replace({60: 70})
accidents_copy['vma'] = accidents_copy['vma'].replace({65: 70})
accidents_copy['vma'] = accidents_copy['vma'].replace({100: 110})
accidents_copy['vma'] = accidents_copy['vma'].replace({120: 130})                                        
# les vitesses à 130 hors autoroute ramenées à 30
accidents_copy.loc[(accidents_copy['vma'] == 130) &
                   (accidents_copy['catr'] != 1), 'vma'] = 30
#les vitesses >70 sur voies cummunales ramenées à 70
accidents_copy.loc[(accidents_copy['vma'] > 70) &
                   (accidents_copy['catr'] == 4), 'vma'] = 70
# voir pour les vitesses à 110 hors autoroute (départementales & nationales)???


# regroupement des catégories de véhicules

cat_vehicules_regroupées = {
    # 1 : 2 roues non motorisés
    1: [1, 60, 80],

    # 2 : 2 roues motorisés légers
    2: [2, 30, 31, 32, 50],

    # 3 : 2 roues motorisés puissants
    3: [33, 34],

    # 4 : 3 roues motorisés
    4: [41, 42, 43],

    # 5 : Véhicules légers
    5: [3, 7, 35, 36],

    # 6 : Véhicules utilitaires et poids lourds
    6: [10, 13, 14, 15, 16, 17],

    # 7 : Transports en commun
    7: [37, 38],

    # 8 : Engins spéciaux
    8: [20, 21, 39, 40],

    # 9 : Autres / Indéterminés
    9: [0, 99],

    # 0 : Inutilisées (pour filtre ou exclusion)
    0: [4, 5, 6, 8, 9, 11, 12, 18, 19]
}

# Inverser le dictionnaire pour mapper chaque ancien code à un nouveau groupe
mapping_cat_veh = {old: new for new, olds in cat_vehicules_regroupées.items() for old in olds}

# Appliquer le mapping
accidents_copy['catv_regroupée'] = accidents_copy['catv'].map(mapping_cat_veh)

# Export fichier nettoyé
accidents_copy.to_csv("/Users/alizeeblanchon/Documents/Data_Scientist/data_project/mai25_bds_accidents/data/processed/accidents_clean.csv", index=False)








