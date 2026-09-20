import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report
from dataset import MultiFaultPantoDataset
from model import MultiDeepSAD

def evaluer_jumeau_numerique_complet():
    print("==========================================================")
    print("   ÉVALUATION DIAGNOSTIC MULTI-PANNES (SEUIL DE SÛRETÉ)   ")
    print("==========================================================")
    
    # Chemins complets
    chemin_usb = r"E:\projet_pfe\railway_train_test_splits"
    simulink_dir = r"E:\01_Modeles_Simulink\panto_oncf\dataset_panto"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Chargement du dataset de test avec le chemin Simulink
    print("Chargement du dataset de test...")
    dataset_test = MultiFaultPantoDataset(base_dir=chemin_usb, simulink_dir=simulink_dir, split="test")
    dataloader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    
    # 2. Chargement du modèle
    model = MultiDeepSAD(input_dim_signal=250, rep_dim=64).to(device)
    model.load_state_dict(torch.load("panto_model_weights.pth", map_location=device))
    model.eval()
    
    centre_c = torch.ones(64, device=device) * 0.1
    
    tous_les_scores = []
    tous_les_labels = []
    
    # 3. Calcul des scores d'anomalie
    with torch.no_grad():
        for signals, images, labels in dataloader:
            signals, images, labels = signals.to(device), images.to(device), labels.to(device)
            
            sorties_latentes = model(signals, images)
            distances = torch.sum((sorties_latentes - centre_c) ** 2, dim=1)
            
            tous_les_scores.extend(distances.cpu().numpy())
            tous_les_labels.extend(labels.cpu().numpy())
            
    tous_les_scores = np.array(tous_les_scores)
    tous_les_labels = np.array(tous_les_labels)
    
    # 4. Calcul du seuil sur la classe "Sain" (Label = 1)
    seuil_alerte = np.percentile(tous_les_scores[tous_les_labels == 1], 97.5)
    
    # 5. Métriques
    predictions_binaires = np.where(tous_les_scores > seuil_alerte, 1, 0)
    labels_binaires = np.where(tous_les_labels == 1, 0, 1) # 0 = Sain, 1 = Panne
    
    print(f"\nSeuil d'alerte configuré : {seuil_alerte:.6f}")
    print(classification_report(labels_binaires, predictions_binaires, target_names=["Sain", "Alerte Panne"]))
    
    # 6. Analyse détaillée
    for name, code_label in [("Arc Électrique", -1), ("Fissure Bras", -2), ("Fuite Pneumatique", -3)]:
        masque = (tous_les_labels == code_label)
        if np.sum(masque) > 0:
            taux = np.mean(tous_les_scores[masque] > seuil_alerte) * 100
            print(f"➔ {name:<18} : Détection réussie à {taux:.1f}%")

if __name__ == "__main__":
    evaluer_jumeau_numerique_complet()