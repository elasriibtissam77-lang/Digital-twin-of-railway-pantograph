import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiDeepSAD(nn.Module):
    def __init__(self, input_dim_signal=250, rep_dim=64):
        super(MultiDeepSAD, self).__init__()
        
        # --- BRANCHE 1 : Traitement du Signal (FFT des capteurs) ---
        # Reçoit le spectre fréquentiel (250 points) et extrait les caractéristiques
        self.signal_encoder = nn.Sequential(
            nn.Linear(input_dim_signal, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, rep_dim),
            nn.ReLU()
        )
        
        # --- BRANCHE 2 : Traitement d'Image (Vision du pantographe) ---
        # Réseau de neurones convolutif (CNN) simplifié pour traiter les images (ex: 64x64)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Sortie: 16 x 32 x 32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Sortie: 32 x 16 x 16
            
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, rep_dim),
            nn.ReLU()
        )
        
        # --- COUCHE DE FUSION MULTIMODALE ---
        # Fusionne les représentations du signal et de la vision vers l'espace latent final
        self.fusion_layer = nn.Linear(rep_dim * 2, rep_dim)
        
    def forward(self, signal, image):
        # 1. Extraction des caractéristiques de chaque modalité
        feat_signal = self.signal_encoder(signal)
        feat_image = self.image_encoder(image)
        
        # 2. Concaténation des deux vecteurs
        combined = torch.cat((feat_signal, feat_image), dim=1)
        
        # 3. Projection dans l'espace de représentation final (Espace Latent)
        latent_space = self.fusion_layer(combined)
        return latent_space

# --- ZONE DE TEST DU MODÈLE ---
if __name__ == "__main__":
    print("--- Test de l'architecture MultiDeepSAD (PyTorch) ---")
    
    # Initialisation du modèle
    model = MultiDeepSAD(input_dim_signal=250, rep_dim=64)
    print("Modèle initialisé avec succès.")
    
    # Simulation d'un mini-batch de données (Taille du batch = 4)
    # 4 signaux FFT de 250 points chacun
    fake_signal = torch.randn(4, 250) 
    # 4 images couleur (3 canaux) de dimensions 64x64 pixels
    fake_image = torch.randn(4, 3, 64, 64)
    
    # Passage des données fictives dans le réseau (Forward Pass)
    output_latent = model(fake_signal, fake_image)
    
    print(f"Forme de la sortie dans l'espace latent : {output_latent.shape}")
    print("-> Attendu: [4, 64] (4 échantillons, projetés en 64 dimensions caractéristiques).")