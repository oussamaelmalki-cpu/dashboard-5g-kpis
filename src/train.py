import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model import LSTMAutoEncoder

def main():
    print("Chargement des donnees d'entrainement...")
    x = np.load("data/x_train.npy")
    
    # On crée un DataLoader (num_workers=0 pour eviter les bugs sous Windows)
    ds = TensorDataset(torch.from_numpy(x))
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

    # On utilise le GPU s'il y en a un, sinon le CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Appareil utilise : {device}")

    model = LSTMAutoEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    print("Debut de l'entrainement (20 epoques)...")
    for epoch in range(1, 21):
        model.train()
        total = 0.0
        # tqdm affiche une jolie barre de progression
        for (xb,) in tqdm(dl, desc=f"Epoque {epoch:02d}", leave=False):
            xb = xb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            
        print(f"Epoque {epoch:02d} | Perte (MSE) = {total / len(ds):.6f}")

    # Sauvegarde du modele entraine
    torch.save(model.state_dict(), "models/lstm_ae.pt")
    print("Modele sauvegarde dans models/lstm_ae.pt !")

if __name__ == "__main__":
    main()