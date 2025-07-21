import os
import streamlit as st

# Déterminer les chemins de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Remonter de 3 dossiers
DATA_RAW_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "data", "raw"))
DATA_PROCESSED_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "data", "processed"))
REPORT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "reports", "figures"))

# Définir les chemins des fichiers CSV (non utilisés ici mais utiles plus tard)
usagers_path = os.path.join(DATA_RAW_DIR, "usagers-2019.csv")
vehicules_path = os.path.join(DATA_RAW_DIR, "vehicules-2019.csv")
lieux_path = os.path.join(DATA_RAW_DIR, "lieux-2019.csv")
caracteristiques_path = os.path.join(DATA_RAW_DIR, "caracteristiques-2019.csv")
accidents_path = os.path.join(DATA_PROCESSED_DIR, "accidents_clean.csv")

def run():
    st.title("Deep Learning")

    # Chemin vers le fichier HTML à afficher
    html_file_path = os.path.join(REPORT_DIR, "pred_grav_france.html")

    # Vérifier si le fichier existe
    if os.path.exists(html_file_path):
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Afficher dans Streamlit
        st.components.v1.html(html_content, height=700, scrolling=True)
    else:
        st.error(f"Le fichier HTML n'a pas été trouvé : {html_file_path}")
