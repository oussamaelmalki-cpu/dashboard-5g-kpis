import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score)
from model import LSTMAutoEncoder

def main():
    print("Chargement des donnees de test...")
    x_tr = np.load("data/x_train.npy")
    x_te = np.load("data/x_test.npy")
    y_te = np.load("data/y_test.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Charger le modele
    print("Chargement du modele...")
    model = LSTMAutoEncoder().to(device)
    model.load_state_dict(torch.load("models/lstm_ae.pt", map_location=device))
    model.eval()

    # 2. Fonction pour calculer l'erreur de reconstruction
    def compute_errors(x):
        with torch.no_grad():
            xb = torch.from_numpy(x).to(device)
            out = model(xb)
            # Moyenne de l'erreur au carre sur les fenetres
            return ((out - xb) ** 2).mean(dim=(1, 2)).cpu().numpy()

    # 3. Calcul du seuil (moyenne + 3 ecarts-types sur les donnees normales d'entrainement)
    err_tr = compute_errors(x_tr)
    mu, sd = err_tr.mean(), err_tr.std()
    threshold = mu + 3 * sd
    print(f"\nSeuil d'alerte calcule : {threshold:.6f} (Moyenne={mu:.6f}, Ecart-type={sd:.6f})")

    # 4. Evaluation sur le jeu de test
    err_te = compute_errors(x_te)
    y_pred = (err_te > threshold).astype(int)

    print("\n--- RESULTATS ---")
    print(f"Precision : {precision_score(y_te, y_pred):.4f}")
    print(f"Rappel    : {recall_score(y_te, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_te, y_pred):.4f}")
    
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_te, y_pred))
    print("(Rappel : Ligne 0 = Normal, Ligne 1 = Anomalie)")
    print("Colonnes : [Vrais Negatifs, Faux Positifs]")
    print("           [Faux Negatifs, Vrais Positifs]")

if __name__ == "__main__":
    main()