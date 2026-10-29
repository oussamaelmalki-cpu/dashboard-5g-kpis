import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

FEATURES = ["rsrp", "sinr", "loss", "traffic", "cpu"]
SEQ_LEN  = 50

def make_windows(X, y, seq_len=SEQ_LEN):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])   # etiquette alignee sur la fin
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)

def main():
    print("Chargement des donnees...")
    df = pd.read_csv("data/kpis.csv")
    
    # Separation Entrainement (70%) / Test (30%)
    split = int(0.7 * len(df))
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    print("Normalisation...")
    # Le scaler n'apprend QUE sur le fonctionnement normal (anomaly == 0)
    scaler = StandardScaler().fit(train_df.loc[train_df.anomaly == 0, FEATURES])
    joblib.dump(scaler, "models/scaler.pkl")

    print("Creation des fenetres glissantes...")
    x_tr, y_tr = make_windows(scaler.transform(train_df[FEATURES]), train_df["anomaly"].to_numpy())
    x_te, y_te = make_windows(scaler.transform(test_df[FEATURES]), test_df["anomaly"].to_numpy())

    # L'ensemble d'entrainement ne contient QUE du normal (pour l'autoencodeur)
    x_tr = x_tr[y_tr == 0]

    # Sauvegarde des fichiers prets pour PyTorch
    np.save("data/x_train.npy", x_tr)
    np.save("data/x_test.npy",  x_te)
    np.save("data/y_test.npy",  y_te)
    
    print(f"Termine !")
    print(f"Taille du jeu d'entrainement (normal uniquement) : {x_tr.shape}")
    print(f"Taille du jeu de test (avec anomalies) : {x_te.shape}")

if __name__ == "__main__":
    main()