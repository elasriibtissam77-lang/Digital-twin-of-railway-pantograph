import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from dataset import MultiFaultPantoDataset
from model import MultiDeepSAD

def generer_visuels_pfe():
    print("==========================================================")
    print("   GÉNÉRATION DES GRAPHIQUES POUR LE RAPPORT DE PFE       ")
    print("==========================================================")
    
    chemin_usb = r"E:\projet_pfe\railway_train_test_splits"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Chargement des données de test
    dataset_test = MultiFaultPantoDataset(base_dir=chemin_usb, split="test")
    dataloader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    
    # 2. Chargement du modèle
    model = MultiDeepSAD(input_dim_signal=250, rep_dim=64).to(device)
    model.load_state_dict(torch.load("panto_model_weights.pth", map_location=device))
    model.eval()
    
    centre_c = torch.ones(64, device=device) * 0.1
    tous_les_scores = []
    tous_les_labels = []
    
    with torch.no_grad():
        for signals, images, labels in dataloader:
            signals = signals.to(device)
            images = images.to(device)
            sorties_latentes = model(signals, images)
            distances = torch.sum((sorties_latentes - centre_c) ** 2, dim=1)
            tous_les_scores.extend(distances.cpu().numpy())
            tous_les_labels.extend(labels.numpy())
            
    tous_les_scores = np.array(tous_les_scores)
    tous_les_labels = np.array(tous_les_labels)
    
    # Réglage parfait à 97.5%
    seuil_alerte = np.percentile(tous_les_scores[tous_les_labels == 1], 97.5)
    predictions_binaires = np.where(tous_les_scores > seuil_alerte, 1, 0)
    labels_binaires = np.where(tous_les_labels == 1, 0, 1) # 0 = Sain, 1 = Panne

    # ----------------------------------------------------------------
    # GRAPHIQUE 1 : LA MATRICE DE CONFUSION
    # ----------------------------------------------------------------
    print("Génération de la Matrice de Confusion...")
    cm = confusion_matrix(labels_binaires, predictions_binaires)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matrice de Confusion - Jumeau Numérique Pantographe")
    plt.colorbar()
    
    classes = ['Sain (Nominal)', 'Alerte Panne']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Ajout des chiffres à l'intérieur du tableau
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, fontweight='bold')
            
    plt.ylabel('Réalité Terrain (ONCF)')
    plt.xlabel('Prédiction du Modèle IA')
    plt.tight_layout()
    plt.savefig('matrice_confusion.png', dpi=300) # Sauvegarde en haute qualité
    plt.close()
    print("➔ Image 'matrice_confusion.png' enregistrée.")

    # ----------------------------------------------------------------
    # GRAPHIQUE 2 : LA COURBE ROC
    # ----------------------------------------------------------------
    print("Génération de la Courbe ROC...")
    fpr, tpr, _ = roc_curve(labels_binaires, tous_les_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taux de Fausses Alertes (FPR)")
    plt.ylabel("Taux de Détection des Pannes (TPR)")
    plt.title("Performance Globale du Séparateur (Courbe ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('courbe_roc.png', dpi=300)
    plt.close()
    print("➔ Image 'courbe_roc.png' enregistrée.")
    print("==========================================================")

if __name__ == "__main__":
    generer_visuels_pfe()
    