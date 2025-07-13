
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