import socket
import time
import numpy as np
import os
import json

UDP_IP = "" #ip adress
UDP_PORT_DASH = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

PATH_REAL_NORMAL = r"" #directory of your dataset located in your laptop
PATH_REAL_ARC = r"" #directory of your dataset located in your laptop
PATH_SIMU = r"" #directory of your dataset located in your laptop 
JSON_BLENDER = r"" #directory of your dataset_panto located in your laptop 

DT = 0.05
H_CATENAIRE = 3.20
H_PLANCHER = 1.80
VITESSE_TRAIN = 2.0
PORTEE_CAT = 60.0
DERIVE_X_MAX = 0.20

_y_train = 0.0

def position_catenaire(dt=DT):
    global _y_train
    _y_train += VITESSE_TRAIN * dt
    phase = (2.0 * np.pi / (2.0 * PORTEE_CAT)) * _y_train
    x_cat = DERIVE_X_MAX * np.sin(phase)
    return _y_train, x_cat, H_CATENAIRE

def ecrire_json_blender(y_panto, x_panto, z_panto, x_cat, z_cat, hi, nom_panne=""):
    """
    Écriture JSON robuste sous Windows.

    Blender peut ouvrir donnees_panto.json exactement au moment où Python
    essaie de le remplacer. Sous Windows cela peut provoquer WinError 5.
    On effectue donc plusieurs tentatives très courtes avant d'abandonner
    uniquement l'image courante.
    """
    payload = {
        "hauteur": round(float(z_panto), 4),
        "deviation_x": round(float(x_panto), 4),
        "avance_y": round(float(y_panto), 4),
        "x_cat": round(float(x_cat), 4),
        "z_cat": round(float(z_cat), 4),
        "hi": round(float(hi), 4),
        "panne": str(nom_panne),
        "ts": time.time(),
    }

    dossier = os.path.dirname(JSON_BLENDER)
    if dossier:
        os.makedirs(dossier, exist_ok=True)

    tmp = JSON_BLENDER + ".tmp"

    # 1) Écriture complète dans le fichier temporaire
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        print(f"⚠️ Impossible d'écrire le fichier temporaire JSON : {e}")
        return False

    # 2) Remplacement avec plusieurs tentatives
    for tentative in range(10):
        try:
            os.replace(tmp, JSON_BLENDER)
            return True

        except PermissionError:
            # Blender est probablement en train de lire le fichier.
            time.sleep(0.01)

        except OSError as e:
            print(f"⚠️ Erreur JSON Blender : {e}")
            break

    # 3) Si Blender garde le fichier trop longtemps,
    # on ignore seulement cette mise à jour et on continue le simulateur.
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except OSError:
        pass

    return False

def diffuser_donnees(f_val, vibration_val, pression_val, temp_val, hi_val, rul_val, nom_panne="", dt=DT):
    y_train, x_cat, z_cat = position_catenaire(dt)

    y_panto = y_train
    x_panto = x_cat
    z_panto = z_cat

    intensite = max(0.0, 1.0 - hi_val)
    panne_lower = nom_panne.lower()

    if nom_panne == "Normal" or nom_panne == "":
        x_panto = x_cat
        z_panto = z_cat

    elif "fuite" in panne_lower:
        chute = (H_CATENAIRE - H_PLANCHER) * intensite
        z_panto = max(z_cat - chute + np.random.normal(0.0, 0.01), H_PLANCHER)
        x_panto = x_cat + np.random.normal(0.0, max(0.005, intensite * 0.02))

    elif "fissure" in panne_lower:
        x_panto = x_cat + 0.55 * intensite + np.random.normal(0.0, max(0.005, intensite * 0.05))
        z_panto = z_cat - 0.25 * intensite + np.random.normal(0.0, max(0.005, intensite * 0.03))

    elif "arc" in panne_lower or "electrique" in panne_lower:
        x_panto = x_cat + np.random.normal(0.0, 0.015)
        z_panto = z_cat + np.random.normal(0.0, 0.01)

    ecrire_json_blender(
        y_panto, x_panto, z_panto, x_cat, z_cat, hi_val, nom_panne
    )

    packet_dash = (
        f"{float(f_val):.4f};"
        f"{float(vibration_val):.4f};"
        f"{float(pression_val):.4f};"
        f"{float(temp_val):.4f};"
        f"{float(hi_val):.4f};"
        f"{float(rul_val):.2f};"
        f"{float(z_panto):.4f};"
        f"{nom_panne}"
    )

    sock.sendto(packet_dash.encode("utf-8"), (UDP_IP, UDP_PORT_DASH))

class PHMEngine:
    def __init__(self, window=30):
        self.hi_history = []
        self.window = window

    def update(self, hi, dt=DT):
        self.hi_history.append(float(hi))
        if len(self.hi_history) > self.window:
            self.hi_history.pop(0)

        if len(self.hi_history) >= 2:
            dhi_dt = (self.hi_history[-1] - self.hi_history[0]) / (len(self.hi_history) * dt)
            if dhi_dt < -1e-4:
                rul = max(0.0, (hi / abs(dhi_dt)) / 60.0)
            else:
                rul = min(200.0, hi * 200.0)
        else:
            rul = hi * 200.0

        return round(float(rul), 2)

def lister_fichiers(dossier, extension):
    if not os.path.isdir(dossier):
        return []
    return sorted(f for f in os.listdir(dossier) if f.lower().endswith(extension.lower()))

def charger_signal_npy(path):
    data = np.load(path)
    return np.asarray(data).reshape(-1)

def charger_signal_csv(path):
    data = np.loadtxt(path, delimiter=",")
    if np.ndim(data) > 1:
        data = data[:, 0]
    return np.asarray(data).reshape(-1)

def jouer_normal(files_normal, phm):
    if not files_normal:
        return

    fname = np.random.choice(files_normal)
    data = charger_signal_npy(os.path.join(PATH_REAL_NORMAL, fname))
    print(f"Etat : Normal - {fname}")

    for val in data[:400]:
        f_val = float(val)
        vibration = 0.50 + np.random.normal(0.0, 0.03)
        pression = 7.00 + np.random.normal(0.0, 0.05)
        temperature = 60.0 + np.random.normal(0.0, 1.0)
        hi = 1.0
        rul = phm.update(hi, DT)

        diffuser_donnees(
            f_val, vibration, pression, temperature, hi, rul, "Normal", DT
        )
        time.sleep(DT)

def jouer_arc(files_arc, phm):
    if not files_arc:
        print("Aucun fichier ARC trouve. Verifie PATH_REAL_ARC.")
        return

    fname = np.random.choice(files_arc)
    data = charger_signal_npy(os.path.join(PATH_REAL_ARC, fname))
    print(f"Etat : Arc electrique - {fname}")

    for val in data[:400]:
        f_val = float(val)
        vibration = 1.5 + abs(np.random.normal(0.0, 0.30))
        pression = 6.8 + np.random.normal(0.0, 0.10)
        temperature = 185.0 + np.random.normal(0.0, 5.0)
        hi = float(np.clip(0.20 + np.random.normal(0.0, 0.02), 0.0, 1.0))
        rul = phm.update(hi, DT)

        diffuser_donnees(
            f_val, vibration, pression, temperature, hi, rul, "Arc électrique", DT
        )
        time.sleep(DT)

def jouer_defaut_simulink(files_simu, phm):
    fault_files = [f for f in files_simu if ("Fissure" in f or "Fuite" in f)]

    if not fault_files:
        print("Aucun CSV Fissure/Fuite trouve.")
        return

    panne_file = np.random.choice(fault_files)
    data_simu = charger_signal_csv(os.path.join(PATH_SIMU, panne_file))

    if "Fissure" in panne_file:
        nom_panne = "Fissure"
        hi_target = 0.55
        pression = 7.0 + np.random.normal(0.0, 0.10)
        temperature = 65.0 + np.random.normal(0.0, 2.0)
        print(f"Etat : Fissure - {panne_file}")
    else:
        nom_panne = "Fuite d'air"
        hi_target = 0.25
        pression = 4.5 + np.random.normal(0.0, 0.15)
        temperature = 62.0 + np.random.normal(0.0, 1.0)
        print(f"Etat : Fuite d'air - {panne_file}")

    for val in data_simu[:600]:
        f_val = float(val)
        hi = float(np.clip(hi_target + np.random.normal(0.0, 0.02), 0.0, 1.0))

        if nom_panne == "Fissure":
            vibration = 4.5 + abs(np.random.normal(0.0, 0.25))
        else:
            vibration = 1.0 + abs(np.random.normal(0.0, 0.10))

        rul = phm.update(hi, DT)

        diffuser_donnees(
            f_val, vibration, pression, temperature, hi, rul, nom_panne, DT
        )
        time.sleep(DT)

def simulateur_hybride():
    print("=" * 72)
    print("PANTOPROGNOSIS - SIMULATEUR HYBRIDE SYNCHRONISE")
    print("=" * 72)

    files_normal = lister_fichiers(PATH_REAL_NORMAL, ".npy")
    files_arc = lister_fichiers(PATH_REAL_ARC, ".npy")
    files_simu = lister_fichiers(PATH_SIMU, ".csv")

    print(f"Normal .npy : {len(files_normal)}")
    print(f"Arc .npy    : {len(files_arc)}")
    print(f"Simulink CSV: {len(files_simu)}")

    if not files_normal:
        print(f"Aucun fichier normal dans : {PATH_REAL_NORMAL}")
        return

    if not files_simu:
        print(f"Aucun fichier CSV dans : {PATH_SIMU}")
        return

    while True:
        phm = PHMEngine(window=30)
        jouer_normal(files_normal, phm)
        time.sleep(1.0)

        phm = PHMEngine(window=30)
        jouer_defaut_simulink(files_simu, phm)
        print("Maintenance simulee - retour au nominal...")
        time.sleep(2.0)

        phm = PHMEngine(window=30)
        jouer_normal(files_normal, phm)
        time.sleep(1.0)

        if files_arc:
            phm = PHMEngine(window=30)
            jouer_arc(files_arc, phm)
            print("Maintenance simulee - retour au nominal...")
            time.sleep(2.0)

if __name__ == "__main__":
    simulateur_hybride()
