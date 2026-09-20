import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MultiFaultPantoDataset
from model import MultiDeepSAD
import os

def lancer_entrainement():
    # 1. Configuration du matériel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Entraînement sur : {device} ---")
    
    # 2. Chemins d'accès
    base_data = r"E:\projet_pfe\railway_train_test_splits"
    simulink_data = r"E:\01_Modeles_Simulink\panto_oncf\dataset_panto"
    
    # 3. Initialisation du Dataset et du DataLoader
    train_dataset = MultiFaultPantoDataset(base_dir=base_data, simulink_dir=simulink_data, split="train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    
    # 4. Modèle Deep SAD
    model = MultiDeepSAD(input_dim_signal=250, rep_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    centre_c = torch.zeros(64, device=device)
    
    # 5. Boucle d'entraînement
    model.train()
    for epoch in range(1, 21):
        loss_epoch = 0.0
        for signals, images, labels in train_loader:
            signals, images, labels = signals.to(device), images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            z = model(signals, images)
            
            # Calcul de la perte Deep SAD
            dist = torch.sum((z - centre_c) ** 2, dim=1)
            loss = torch.sum(dist[labels == 1]) + torch.sum(1.0 / (dist[labels != 1] + 1e-6))
            loss = loss / 32
            
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            
        print(f"Époque [{epoch}/20] | Perte Moyenne: {loss_epoch/len(train_loader):.4f}")

    # 6. Sauvegarde robuste
    dossier_sortie = "Modeles_Sortie"
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
    
    chemin_sauvegarde = os.path.join(dossier_sortie, "panto_model_weights.pth")
    torch.save(model.state_dict(), chemin_sauvegarde)
    
    print(f"🎯 Entraînement terminé. Modèle sauvegardé dans : {chemin_sauvegarde}")

if __name__ == "__main__":
    lancer_entrainement()