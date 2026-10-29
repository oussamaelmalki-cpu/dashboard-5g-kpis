import numpy as np
import pandas as pd
import streamlit as st
import torch
import joblib
import sys
import os

# Permet à app.py de trouver le modele dans le dossier src
sys.path.append('src')
from model import LSTMAutoEncoder

FEATURES = ["rsrp", "sinr", "loss", "traffic", "cpu"]
SEQ_LEN  = 50

# Mise en cache pour ne pas recharger le modele à chaque clic
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMAutoEncoder().to(device)
    model.load_state_dict(torch.load("models/lstm_ae.pt", map_location=device))
    model.eval()
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler, device

@st.cache_data
def load_data():
    return pd.read_csv("data/kpis.csv")

def compute_errors(model, scaler, device, df):
    X = scaler.transform(df[FEATURES]).astype(np.float32)
    # On cree les fenetres
    xs = np.array([X[i:i + SEQ_LEN] for i in range(len(X) - SEQ_LEN + 1)])
    with torch.no_grad():
        xb = torch.from_numpy(xs).to(device)
        out = model(xb)
        err = ((out - xb) ** 2).mean(dim=(1, 2)).cpu().numpy()
    # On rajoute des zeros au debut pour que le graphique ait la meme taille
    return np.concatenate([np.zeros(SEQ_LEN - 1), err])

# --- Configuration de la page ---
st.set_page_config(page_title="AIOps 5G", layout="wide")
st.markdown("Visualisation des KPIs réseau et détection d'anomalies par Autoencodeur LSTM.")

# --- Chargement ---
model, scaler, device = load_model()
df = load_data()
df["error"] = compute_errors(model, scaler, device, df)

# --- Interface ---
st.sidebar.header("Paramètres")
# Le curseur pour le seuil (on met une valeur par défaut un peu élevée pour voir l'effet)
default_threshold = float(df["error"].max() * 0.2)
threshold = st.sidebar.slider("Seuil d'alerte (Erreur de reconstruction)", 
                              0.0, float(df["error"].max()), default_threshold)

# Application du seuil
df["alert"] = (df["error"] > threshold).astype(int)

# --- Métriques en haut ---
col1, col2, col3 = st.columns(3)
col1.metric("Anomalies détectées (Modèle)", int(df["alert"].sum()))
col2.metric("Anomalies réelles (Vérité)", int(df["anomaly"].sum()))
col3.metric("Taux de faux positifs", f"{(df['alert'].sum() / len(df) * 100):.2f} %")

# --- Graphiques ---
st.subheader("1. Séries temporelles des KPIs")
# On affiche une portion pour que ce soit lisible
st.line_chart(df[["rsrp", "sinr", "cpu", "loss"]].iloc[:5000])

st.subheader("2. Erreur de reconstruction et Alertes")
st.markdown("L'erreur augmente fortement lors des dégradations (anomalies).")
chart_data = df[["error", "alert"]].iloc[:5000]
st.line_chart(chart_data)

st.subheader("3. Aperçu des données brutes")
st.dataframe(df.head(100))