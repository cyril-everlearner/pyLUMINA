'''
Looking at diffraction of apertured gaussian laser beam with or without focusing through a lens. Fluence maps are in J/cm2.
Beware of the aperture size with respect to the Gaussian waist depending on what you want.
e.g for a uniform disk, make sure to set apertur width >> gaussian waist and so on. 
You can choose the colormap, berlin or coolwarm are particularly appropriate to highlight
low intensity fluence rings.

Author: Cyril Mauclair, LPhiA Angers, France.

License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
This work is licensed under the Creative Commons Attribution 4.0 International License.
You are free to:
- Share: Copy and redistribute the material in any medium or format.
- Adapt: Remix, transform, and build upon the material for any purpose, even commercially.
Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, 
  and indicate if changes were made. You may do so in any reasonable manner, 
  but not in any way that suggests the licensor endorses you or your use.


The GUI Tkinter theme is Azure-ttk-theme, under the MIT License

Copyright (c) 2021 rdbende
'''

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.constants import c, pi
import tkinter as tk
from tkinter import ttk



from laser_prop_functions import *

def run_simulation():
    try:
        nbpixel = int(entry_nbpixel.get())
        waist = float(entry_waist.get())
        taillefenetre = float(entry_taillefenetre.get())
        z = float(entry_z.get())
        landa = float(entry_landa.get())
        pulse_energy = float(entry_pulse_energy.get())
        pulse_FWHM = float(entry_pulse_FWHM.get())
        nbplane = int(entry_nbplane.get())
        aperture_width = float(entry_aperture_width.get())
        aperture_type = aperture_var.get()  # Récupération du type d'ouverture

        # Génération du champ source
        source = gaussian_source(nbpixel, waist, taillefenetre, pulse_energy, pulse_FWHM)

        # Application de l'ouverture choisie si activée
        if apply_aperture_var.get():            
            source = apply_aperture(source, aperture_type, aperture_width, taillefenetre)

        # Ajout de la phase parabolique si activé
        if apply_lens_phase_var.get():
            f = float(entry_focal_length.get())
            source = apply_lens_phase(source, f, taillefenetre, landa)

        # Calcul du rayon théorique de la tâche d'Airy (effet lentille négligé))
        airy_diameter = airy_disk_radius(landa, aperture_width, z)
        print(f"Airy radius (without lens!): {airy_diameter:.2e} m")

        # Propagation
        z_planes = np.linspace(0, z, nbplane)
        propagated_fields = np.zeros((nbplane, nbpixel, nbpixel), dtype=np.complex128)

        for i, z in enumerate(z_planes):
            propagated_fields[i] = propagation(source, z, landa, nbpixel, taillefenetre)

        # Calcul de la fluence
        fluence3D = (pulse_FWHM * 3e8 * 8.85e-12 * np.abs(propagated_fields)**2) / (2 * 0.94) * 1e4

        # Sélection de la colormap
        cmap_selected = colormap_var.get()

        # Visualisation
        plot_propagation_2D(fluence3D, z_planes, taillefenetre, cmap_selected)

        if wanna_go_3D.get():
            plot_propagation_3D(fluence3D, z_planes, taillefenetre, recordGIF=True, parameters=parameters)

    except ValueError:
        tk.messagebox.showerror("Input Error", "Please enter valid numerical values.")


root = tk.Tk()

# Définition de l'icône
icon_path = "logo.png"  # Fichier ICO (Windows) ou PNG (Linux/Mac)
icon = tk.PhotoImage(file=icon_path)
root.iconphoto(True, icon)

# Thème de la GUI
# Définir le chemin vers le fichier azure.tcl
theme_path = os.path.join(os.path.dirname(__file__), "Azure-ttk-theme-main", "azure.tcl")

# Charger le thème Azure
root.tk.call("source", theme_path)

# Définir le thème en mode clair ou sombre
root.tk.call("set_theme", "light")  # Ou "light"

root.title("Laser Beam Propagation Simulator 3D")

# Interface
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

parameters = [
    ("Number of pixels", 256),
    ("Beam waist (m)", 1e-1),
    ("Window size (m)", 0.02),
    ("Propagation distance (m)", 5.0),
    ("Wavelength (m)", 1030e-9),
    ("Pulse energy (J)", 1e-6),
    ("Pulse FWHM (s)", 1e-13),
    ("Number of planes", 10),
    ("Aperture width (m)", 2e-3),
    ("Focal Length (m):", 1),
]

entries = []
for i, (label, default) in enumerate(parameters):
    ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W)
    entry = ttk.Entry(frame, width=15)
    entry.grid(row=i, column=1)
    entry.insert(0, str(default))
    entries.append(entry)

entry_nbpixel, entry_waist, entry_taillefenetre, entry_z, entry_landa, entry_pulse_energy, entry_pulse_FWHM, entry_nbplane, entry_aperture_width, entry_focal_length = entries

# Choix du type d'ouverture
aperture_var = tk.StringVar(value="disk")
aperture_label = ttk.Label(frame, text="Aperture Type:")
aperture_label.grid(row=len(parameters), column=0, sticky=tk.W)

aperture_options = ["disk", "square", "triangle", "annulus"]
aperture_menu = ttk.Combobox(frame, textvariable=aperture_var, values=aperture_options, state="readonly")
aperture_menu.grid(row=len(parameters), column=1)

apply_aperture_var = tk.BooleanVar(value=True)
checkbox_lens = ttk.Checkbutton(frame, text="Apply Aperture?", variable=apply_aperture_var)
checkbox_lens.grid(row=len(parameters), column=2,columnspan=1, sticky=tk.W)

# Checkbox pour affichage 3D
wanna_go_3D = tk.BooleanVar(value=False)
checkbox = ttk.Checkbutton(frame, text="Wanna go 3D?", variable=wanna_go_3D)
checkbox.grid(row=len(parameters) + 2, column=2, sticky=tk.W)

# Choix de la colormap
colormap_var = tk.StringVar(value="coolwarm")
colormap_label = ttk.Label(frame, text="Colormap:")
colormap_label.grid(row=len(parameters) + 2, column=0, sticky=tk.W)

colormap_options = ["inferno", "viridis", "plasma", "magma", "cividis", "jet", "gray", "berlin", "coolwarm"]
colormap_menu = ttk.Combobox(frame, textvariable=colormap_var, values=colormap_options, state="readonly")
colormap_menu.grid(row=len(parameters) + 2, column=1)

# Ajout des éléments de la GUI pour la focale
apply_lens_phase_var = tk.BooleanVar(value=False)
checkbox_lens = ttk.Checkbutton(frame, text="Apply Lens Phase?", variable=apply_lens_phase_var)
checkbox_lens.grid(row=len(parameters) - 1, column=2,columnspan=1, sticky=tk.W)

# Ajout d'un bouton pour changer de thème
def toggle_theme():
    current_theme = root.tk.call("ttk::style", "theme", "use")
    new_theme = "light" if current_theme == "azure-dark" else "dark"
    root.tk.call("set_theme", new_theme)
theme_button = ttk.Button(frame, text="Dark/light Theme", command=toggle_theme)
theme_button.grid(row=0, column=2)

# Bouton pour lancer la simulation
button_run = ttk.Button(frame, text="Run Simulation",style='Accent.TButton', command=run_simulation)
button_run.grid(row=len(parameters) + 6, column = 1)



root.mainloop()

