import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import os
# Import des modèles
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# remonter de 3 dossiers
DATA_RAW_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "data", "raw"))
DATA_PROCESSED_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "data", "processed"))

usagers_path = os.path.join(DATA_RAW_DIR, "usagers-2019.csv")
vehicules_path = os.path.join(DATA_RAW_DIR, "vehicules-2019.csv")
lieux_path = os.path.join(DATA_RAW_DIR, "lieux-2019.csv")
caracteristiques_path = os.path.join(DATA_RAW_DIR, "caracteristiques-2019.csv")
accidents_path = os.path.join(DATA_PROCESSED_DIR, "accidents_clean.csv")

def run():
    st.title("Modélisation des accidents")
    st.markdown("### Sélection du modèle et des paramètres")

    # Chargement des données
    try:
        df = pd.read_csv(accidents_path)
    except FileNotFoundError:
        st.error("Fichier 'accidents_clean.csv' introuvable.")
        return

    # Sélection des features et de la target
    if "grav" not in df.columns:
        st.error("La colonne cible 'grav' est manquante.")
        return

    target = "grav"
    features = df.drop(columns=[target]).select_dtypes(include=["int", "float"]).columns.tolist()
    X = df[features]
    y = df[target]

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choix du modèle
    model_choice = st.selectbox("Choisissez un modèle", ["Régression Logistique", "Random Forest", "XGBoost"])

    # Hyperparamètres génériques
    test_size = st.slider("Proportion de l'échantillon de test", 0.1, 0.5, 0.3, 0.05)
    random_state = st.number_input("Random seed", value=42)

    # Spécificités par modèle
    if model_choice == "Régression Logistique":
        c_value = st.number_input("C (inverse de la régularisation)", min_value=0.001, max_value=10.0, value=1.0)
        model = LogisticRegression(C=c_value, max_iter=1000, random_state=random_state)
    elif model_choice == "Random Forest":
        n_estimators = st.slider("Nombre d'estimateurs", 10, 500, 100, 10)
        max_depth = st.slider("Profondeur max", 1, 50, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    elif model_choice == "XGBoost":
        learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1)
        n_estimators = st.slider("Nombre d'estimateurs", 10, 500, 100, 10)
        model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)

    # Bouton pour entraîner
    if st.button("Entraîner le modèle"):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

        # Entraînement
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Affichage des performances
        st.markdown("### Rapport de classification")
        st.text(classification_report(y_test, y_pred))

        # Matrice de confusion
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax_cm)
        st.pyplot(fig_cm)

        # Courbe ROC (si binaire)
        if len(y.unique()) == 2:
            y_proba = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            ax_roc.plot([0, 1], [0, 1], "k--")
            ax_roc.set_xlabel("Faux positifs")
            ax_roc.set_ylabel("Vrais positifs")
            ax_roc.set_title("Courbe ROC")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

        # Importance des variables
        if hasattr(model, "feature_importances_"):
            st.markdown("### Importance des variables")
            importances = pd.Series(model.feature_importances_, index=features)
            fig_imp, ax_imp = plt.subplots(figsize=(10,6))
            importances.sort_values().plot(kind="barh", ax=ax_imp)
            ax_imp.set_title("Importance des variables")
            st.pyplot(fig_imp)
