import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms

class MultiFaultPantoDataset(Dataset):
    def __init__(self, base_dir, simulink_dir, split="train", transform=None):
        self.base_dir = base_dir
        self.simulink_dir = simulink_dir
        self.split = split
        
        # Chemins de base
        if split == "train":
            self.force_dir_sain = os.path.join(base_dir, "force_train", "0")
            self.img_dir_sain = os.path.join(base_dir, "imgs_train", "0")
        else:
            self.force_dir_sain = os.path.join(base_dir, "force_test", "0")
            self.img_dir_sain = os.path.join(base_dir, "imgs_test", "0")
            
        self.force_dir_arc = os.path.join(base_dir, "arcing_forces")
        self.img_dir_arc = os.path.join(base_dir, "arcings")

        self.files_sains = sorted([f for f in os.listdir(self.force_dir_sain) if f.endswith('.npy')])
        self.files_arcs = sorted([f for f in os.listdir(self.force_dir_arc) if f.endswith('.npy')])
        
        self.samples = []
        
        # 1. Ajout des cas Sains (Label = 1)
        for f in self.files_sains:
            self.samples.append((f, 1, self.force_dir_sain, self.img_dir_sain, "reel"))
            
        # 2. Ajout des Arcs Électriques (Label = -1)
        for f in self.files_arcs:
            self.samples.append((f, -1, self.force_dir_arc, self.img_dir_arc, "reel"))
            
        # 3. Ajout des Fissures (Label = -2) et Fuites (Label = -3)
        for i, f in enumerate(self.files_sains[:400]): 
            if i % 2 == 0:
                self.samples.append((f, -2, self.force_dir_sain, self.img_dir_sain, "fissure"))
            else:
                self.samples.append((f, -3, self.force_dir_sain, self.img_dir_sain, "fuite"))

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label, force_folder, img_folder, type_panne = self.samples[idx]
        
        # Chargement image
        img_filename = filename.replace(".npy", ".jpg")
        img_path = os.path.join(img_folder, img_filename)
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".png")
            
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.fromarray(np.uint8(np.zeros((64, 64, 3))))

        # Chargement signal : Réel ou Simulink
        if type_panne == "reel":
            force_path = os.path.join(force_folder, filename)
            signal = np.load(force_path).astype(np.float32)
        else:
            rand_idx = np.random.randint(1, 101)
            prefix = "Fissure" if type_panne == "fissure" else "Fuite"
            simu_path = os.path.join(self.simulink_dir, f"{prefix}_{rand_idx}.csv")
            
            # Correction : gestion des fichiers CSV multi-colonnes
            signal = np.loadtxt(simu_path, delimiter=',').astype(np.float32)
            if signal.ndim > 1:
                signal = signal[:, 0]
            
            if type_panne == "fuite":
                image = transforms.functional.adjust_brightness(image, 0.8)

        # Standardisation de la taille du signal à 250 points
        if len(signal) > 250:
            signal = signal[:250]
        elif len(signal) < 250:
            signal = np.pad(signal, (0, 250 - len(signal)), mode='constant')

        # Calcul FFT
        fft_vals = np.abs(np.fft.fft(signal)) / 250
        signal_fft = torch.tensor(fft_vals[:250], dtype=torch.float32)
            
        if self.transform:
            image = self.transform(image)
            
        return signal_fft, image, torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    base_data = r"E:\projet_pfe\railway_train_test_splits"
    simulink_data = r"E:\01_Modeles_Simulink\panto_oncf\dataset_panto"
    
    dataset = MultiFaultPantoDataset(base_dir=base_data, simulink_dir=simulink_data)
    print(f"Dataset prêt. {len(dataset)} échantillons chargés.")