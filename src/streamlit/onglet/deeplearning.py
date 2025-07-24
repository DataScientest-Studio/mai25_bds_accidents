import streamlit as st
import shap
import joblib
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os
import sys
from streamlit.components.v1 import html
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))

if GAT_DIR not in sys.path:
    sys.path.insert(0, GAT_DIR)

from gat import GATResNet
import torch

MAP_PATH = os.path.join(BASE_DIR, "..", "..", "..", "reports", "figures", "pred_grav_france.html")
HISTORY_PATH = os.path.join(BASE_DIR, "..", "..", "..", "models", "history_gat.joblib")
SHAP_PATH = os.path.join(BASE_DIR, "..", "..", "..", "models", "shap_values.joblib")
EXPLAINER_PATH = os.path.join(BASE_DIR, "..", "..", "..", "models", "explainer.joblib")
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "..", "models", "data.pt")
DF_PATH = os.path.join(BASE_DIR, "..", "..", "..", "data", "processed", "GAT_df.csv")
OHE_PATH = os.path.join(BASE_DIR, "..", "..", "..", "models", "ohe.joblib")
THR_PATH = os.path.join(BASE_DIR, "..", "..", "..", "models", "best_thr.joblib")
WEIGHTS_PATH = os.path.join(BASE_DIR, "..", "..", "..", "models", "GAT_weights.pt")
DF_FLAT_PATH = os.path.join(BASE_DIR, "..", "..", "..", "data", "processed", "df_flat.csv")

# === Fonction principale ===
def run():
    st.title("Prédiction de tronçons à risque d'accidents graves (GAT)")

    multi = """
    - :red_circle: Maillage de la France métropolitaine en tronçons de 10 mètres environ
    - :red_circle: Création d'une variable "nombre d'accidents"
    - :red_circle: Perte des variables liées aux usagers et véhicules
    - :red_circle: Enrichissement avec des variables géographiques et d'infrastructure (trafic, éclairage, ralentisseurs, distance hôpitaux et casernes...)
    - :red_circle: Création d'un modèle GAT (Graph Attention Network) pour prédire les tronçons à risque d'accidents graves
    """

    st.markdown(multi)

    shap_values = joblib.load(SHAP_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    data = torch.load(DATA_PATH, map_location=torch.device("cpu"), weights_only=False)
    df = pd.read_csv(DF_PATH)
    ohe = joblib.load(OHE_PATH)
    best_thr = joblib.load(THR_PATH)
    history = joblib.load(HISTORY_PATH)
    df_flat = pd.read_csv(DF_FLAT_PATH)


    NUM_INPUTS = data.x.shape[1]
    model = GATResNet(in_dim=NUM_INPUTS, hidden=32, out_dim=2, heads=8, dropout=0.3)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device("cpu")))
    model.eval()

    st.subheader(":chart_with_upwards_trend: Courbes d'apprentissage")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = list(range(10, 101, 10))
    train_losses = history["loss_tr"]
    val_losses = history["loss_va"]
    train_accuracies = history["acc_tr"]
    val_accuracies = history["acc_va"]
    # Graphe Loss
    ax1.plot(epochs, train_losses, label="train_loss")
    ax1.plot(epochs, val_losses, label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()

    # Graphe Accuracy
    ax2.plot(epochs, train_accuracies, label="train_acc")
    ax2.plot(epochs, val_accuracies, label="val_acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader(":bar_chart: Rapport de classification")
    y_true = data.y.cpu().numpy()
    y_pred = (model(data).softmax(1)[:, 1] > best_thr).int().cpu().numpy()
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()

    st.dataframe(df_report.style.format("{:.2f}"))

    model.eval()
    with torch.no_grad():
        logits_test = model(data)[data.test_mask]                  
        probas_test = logits_test.softmax(1)[:, 1].cpu().numpy()  
        preds = (probas_test > best_thr).astype(int)              
        true = data.y[data.test_mask].cpu().numpy()         
    cm = confusion_matrix(true, preds)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax) 
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    st.pyplot(fig)

    st.subheader(":honeybee: SHAP Beeswarm plot")
    plt.figure(figsize=(12,7))                         
    shap.summary_plot(shap_values, df_flat, show=False)     
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()  

    st.subheader(":world_map: Carte des prédictions")
    if os.path.exists(MAP_PATH):
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            html_data = f.read()
        html(html_data, height=700, scrolling=True)
    else:
        st.warning("Carte non trouvée. Vous pouvez la générer avec folium.")

if __name__ == "__main__":
    run()