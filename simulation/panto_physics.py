import numpy as np

class PantoPhysics:
    def __init__(self):
        # Paramètres physiques (Basés sur le modèle à 2 masses)
        self.M1 = 9.0    # Archet
        self.M2 = 18.0   # Bras
        self.K1 = 7000.0 # Raideur
        self.C1 = 30.0   # Amortissement
        self.F_uplift = 70.0 # Force de levage
        self.dt = 0.001 
        self.state = np.zeros(4) # [pos1, vit1, pos2, vit2]

    def step(self, fault_type, severity):
        # --- Logique des pannes ---
        k_eff = self.K1 * (1.0 - 0.5 * severity) if fault_type == "Fissure" else self.K1
        f_up = self.F_uplift * (1.0 - 0.7 * severity) if fault_type == "Fuite" else self.F_uplift
        
        # --- Équations de Newton ---
        f_contact = max(0, 1000.0 * self.state[0]) 
        accel_bras = (f_up - k_eff * (self.state[2]-self.state[0]) - self.C1*(self.state[3]-self.state[1])) / self.M2
        accel_archet = (k_eff * (self.state[2]-self.state[0]) + self.C1*(self.state[3]-self.state[1]) - f_contact) / self.M1
        
        # --- Mise à jour ---
        self.state[3] += accel_bras * self.dt
        self.state[2] += self.state[3] * self.dt
        self.state[1] += accel_archet * self.dt
        self.state[0] += self.state[1] * self.dt
        
        return f_contact, accel_archet
    