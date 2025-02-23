'''
Looking at diffraction of an apertured Gaussian laser beam with or without focusing through a lens.
Fluence maps are in J/cm². The laser electric field is calculated in V/m.
Be careful with the aperture size relative to the Gaussian waist, depending on what you want to study.

The default values shows a Bessel-type beam applied with an SLM where the blur value in pixel represents the pixel cross talk. Carefull, the angle value and bessel beam length have NOT been verified yet!!!

By clicking on "Wanna go 3D": You can save the stack of propagated planes as a GIF in both false color and grayscale (8-bit).
A "GIF_readme.txt" file is also saved, providing additional details for anyone who wants to analyze the files later.

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

The GUI Tkinter theme is Azure-ttk-theme, under the MIT License.

Copyright (c) 2021 rdbende
'''

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for compatibility with Tkinter
import matplotlib.pyplot as plt
from scipy.constants import c, pi
import tkinter as tk
from tkinter import ttk

from laser_prop_functions import *  # Importing external functions related to laser propagation

def run_simulation():
    """Runs the laser beam propagation simulation based on user-defined parameters."""
    try:
        # Retrieving user inputs from the GUI
        nbpixel = int(entry_nbpixel.get())
        waist = float(entry_waist.get())
        taillefenetre = float(entry_taillefenetre.get())
        z = float(entry_z.get())
        landa = float(entry_landa.get())
        pulse_energy = float(entry_pulse_energy.get())
        pulse_FWHM = float(entry_pulse_FWHM.get())
        nbplane = int(entry_nbplane.get())
        aperture_width = float(entry_aperture_width.get())
        aperture_type = aperture_var.get()  # Get selected aperture type

        # Generating the Gaussian beam source
        source = gaussian_source(nbpixel, waist, taillefenetre, pulse_energy, pulse_FWHM)

        # Applying axiconic phase
        if apply_axicon_phase_var.get():
            axicon_angle = float(entry_axicon_angle.get())
            source = apply_axiconic_phase(source, axicon_angle, taillefenetre, landa)
        
        # Applying blur if enabled
        if apply_blur_var.get():
            blur_sigma = float(entry_blur_sigma.get())
            source = apply_blur(source, blur_sigma)

        # Applying the selected aperture if enabled
        if apply_aperture_var.get():
            source = apply_aperture(source, aperture_type, aperture_width, taillefenetre)

        # Adding lens phase if enabled
        if apply_lens_phase_var.get():
            f = float(entry_focal_length.get())
            source = apply_lens_phase(source, f, taillefenetre, landa)

        # Propagation simulation over multiple planes
        z_planes = np.linspace(0, z, nbplane)
        propagated_fields = np.zeros((nbplane, nbpixel, nbpixel), dtype=np.complex128)

        for i, z in enumerate(z_planes):
            propagated_fields[i] = propagation(source, z, landa, nbpixel, taillefenetre)

        # Compute fluence distribution
        fluence3D = (pulse_FWHM * c * 8.85e-12 * np.abs(propagated_fields)**2) / (2 * 0.94) * 1e4  # Convert to J/cm²

        # Retrieve colormap selection from the user
        cmap_selected = colormap_var.get()

        # Display 2D fluence maps
        plot_propagation_2D(fluence3D, z_planes, taillefenetre, cmap_selected)
        
        # Display Phase maps
        plot_phase_2D(propagated_fields, z_planes, taillefenetre, cmap="gray")
        
        # If 3D visualization is selected, generate and save 3D fluence distribution
        if wanna_go_3D.get():
            plot_propagation_3D(fluence3D, z_planes, taillefenetre, recordGIF=True, parameters=parameters, apply_lens=apply_lens_phase_var.get(), cmap=cmap_selected)
            

    except ValueError:
        tk.messagebox.showerror("Input Error", "Please enter valid numerical values.")

# Initialize Tkinter root window
root = tk.Tk()

# Set application icon
icon_path = "logo.png"  # Use PNG for Linux/Mac, ICO for Windows
icon = tk.PhotoImage(file=icon_path)
root.iconphoto(True, icon)

# Load the Azure ttk theme
theme_path = os.path.join(os.path.dirname(__file__), "Azure-ttk-theme-main", "azure.tcl")
root.tk.call("source", theme_path)
root.tk.call("set_theme", "light")  # Set default theme to light

root.title("Laser Beam Propagation Simulator 3D")

# GUI Main Frame
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Simulation parameters
parameters = [
    ("Number of pixels", 256),
    ("Beam waist (m)", 0.005),
    ("Window size (m)", 0.02),
    ("Propagation distance (m)", 3.0),
    ("Wavelength (m)", 1030e-9),
    ("Pulse energy (J)", 1e-6),
    ("Pulse FWHM (s)", 1e-13),
    ("Number of planes", 10),
    ("Aperture width (m)", 1e-2),
    ("Focal Length (m):", 1),
]

# Creating GUI input fields for each parameter
entries = []
for i, (label, default) in enumerate(parameters):
    ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W)
    entry = ttk.Entry(frame, width=15)
    entry.grid(row=i, column=1)
    entry.insert(0, str(default))
    entries.append(entry)

entry_nbpixel, entry_waist, entry_taillefenetre, entry_z, entry_landa, entry_pulse_energy, entry_pulse_FWHM, entry_nbplane, entry_aperture_width, entry_focal_length = entries

# Aperture selection menu
aperture_var = tk.StringVar(value="disk")
aperture_label = ttk.Label(frame, text="Aperture Type:")
aperture_label.grid(row=len(parameters), column=0, sticky=tk.W)
aperture_options = ["square", "disk", "triangle", "annulus"]
aperture_menu = ttk.Combobox(frame, textvariable=aperture_var, values=aperture_options, state="readonly")
aperture_menu.grid(row=len(parameters), column=1)

# Aperture enable checkbox
apply_aperture_var = tk.BooleanVar(value=False)
checkbox_aperture = ttk.Checkbutton(frame, text="Apply Aperture?", variable=apply_aperture_var)
checkbox_aperture.grid(row=len(parameters), column=2, sticky=tk.W)

# 3D visualization checkbox
wanna_go_3D = tk.BooleanVar(value=False)
checkbox_3D = ttk.Checkbutton(frame, text="Wanna go 3D?", variable=wanna_go_3D)
checkbox_3D.grid(row=len(parameters) + 3, column=2, sticky=tk.W)

# Colormap selection menu
colormap_var = tk.StringVar(value="berlin")
colormap_label = ttk.Label(frame, text="Colormap:")
colormap_label.grid(row=len(parameters) + 3, column=0, sticky=tk.W)
colormap_options = ["inferno", "viridis", "plasma", "magma", "cividis", "jet", "gray", "berlin", "coolwarm"]
colormap_menu = ttk.Combobox(frame, textvariable=colormap_var, values=colormap_options, state="readonly")
colormap_menu.grid(row=len(parameters) + 3, column=1)

# Lens phase checkbox
apply_lens_phase_var = tk.BooleanVar(value=False)
checkbox_lens = ttk.Checkbutton(frame, text="Apply Lens Phase?", variable=apply_lens_phase_var)
checkbox_lens.grid(row=len(parameters) - 1, column=2, sticky=tk.W)

# Champ pour l'angle physique de l'axicon
ttk.Label(frame, text="Axicon Angle (deg):").grid(row=len(parameters) + 1, column=0, sticky=tk.W)
entry_axicon_angle = ttk.Entry(frame, width=15)
entry_axicon_angle.grid(row=len(parameters) + 1, column=1)
entry_axicon_angle.insert(0, "0.1")  # Valeur par défaut

# Checkbox pour activer/désactiver la phase axiconique
apply_axicon_phase_var = tk.BooleanVar(value=True)
checkbox_axicon = ttk.Checkbutton(frame, text="Apply Axicon Phase?", variable=apply_axicon_phase_var)
checkbox_axicon.grid(row=len(parameters) + 1, column=2, sticky=tk.W)


# Checkbox pour activer/désactiver blur
apply_blur_var = tk.BooleanVar(value=True)
checkbox_blur = ttk.Checkbutton(frame, text="Apply Blur?", variable=apply_blur_var)
checkbox_blur.grid(row=len(parameters) + 2, column=2, sticky=tk.W)

# Blur sigma value (gaussian filter)
ttk.Label(frame, text="Sigma (pixels):").grid(row=len(parameters) + 2, column=0, sticky=tk.W)
entry_blur_sigma = ttk.Entry(frame, width=15)
entry_blur_sigma.grid(row=len(parameters) + 2, column=1)
entry_blur_sigma.insert(0, "1.2")  # Valeur par défaut

# Theme toggle button
def toggle_theme():
    current_theme = root.tk.call("ttk::style", "theme", "use")
    new_theme = "light" if current_theme == "azure-dark" else "dark"
    root.tk.call("set_theme", new_theme)

theme_button = ttk.Button(frame, text="Dark/light Theme", command=toggle_theme)
theme_button.grid(row=0, column=2)

# Simulation run button
button_run = ttk.Button(frame, text="Run Simulation", style='Accent.TButton', command=run_simulation)
button_run.grid(row=len(parameters) + 6, column=1)

root.mainloop()

