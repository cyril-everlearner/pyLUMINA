
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox

# Boîte de dialogue pour les paramètres
root = tk.Tk()
root.withdraw()  # Cacher la fenêtre principale

# Valeurs par défaut pour l'inox (paramètres C3 dans l'article)
default_delta_nm = 18  # nm
default_Fth = 0.055     # J/cm²

try:
    delta = float(simpledialog.askstring("Paramètre delta (nm)",
        f"Entrez la valeur de delta (nm) [défaut = {default_delta_nm}]:") or default_delta_nm)
    Fth = float(simpledialog.askstring("Fluence seuil Fth (J/cm²)",
        f"Entrez la valeur de Fth [défaut = {default_Fth} J/cm²]:") or default_Fth)
except Exception as e:
    messagebox.showerror("Erreur", f"Erreur de saisie : {e}")
    exit()

# Conversion de delta en microns pour être cohérent avec l'axe Y
delta_um = delta / 1000  # nm → µm

# Plage de fluence pour le tracé
F = np.linspace(Fth * 1.01, 10, 500)  # évite ln(0) et va jusqu'à 10 J/cm²

# Formule d'ablation par impulsion
z = delta_um * np.log(F / Fth)  # z en µm

# Tracé
plt.figure(figsize=(8, 5))
plt.plot(F, z, label=f"δ = {delta} nm, Fth = {Fth} J/cm²", color='navy')
plt.xlabel("Fluence F (J/cm²)")
plt.ylabel("Profondeur par impulsion z (µm)")
plt.title("Modèle de profondeur d’ablation par impulsion (inox)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

