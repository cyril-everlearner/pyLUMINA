"""
Scénario simple de vérification de propagation d'un faisceau Gaussien
Calcul de sa fluence.
Approximation non paraxiale


Author: Cyril Mauclair
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
This work is licensed under the Creative Commons Attribution 4.0 International License.
You are free to:
- Share: Copy and redistribute the material in any medium or format.
- Adapt: Remix, transform, and build upon the material for any purpose, even commercially.
Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, 
  and indicate if changes were made. You may do so in any reasonable manner, 
  but not in any way that suggests the licensor endorses you or your use.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from scipy.constants import c, pi
import tkinter as tk
from tkinter import ttk

from laser_prop_functions import *


def run_simulation():
    """
    Scénario simple de vérification de propagation d'un faisceau Gaussien
    Calcul de sa fluence.
    Approximation non paraxiale
    """
    try:
        nbpixel = int(entry_nbpixel.get())
        waist = float(entry_waist.get())
        taillefenetre = float(entry_taillefenetre.get())
        z = float(entry_z.get())
        landa = float(entry_landa.get())
        pulse_energy = float(entry_pulse_energy.get())
        pulse_FWHM = float(entry_pulse_FWHM.get())

        # Generate source and propagate
        source = gaussian_source(nbpixel, waist, taillefenetre, pulse_energy,pulse_FWHM)
        beam_field = propagation(source, z, landa, nbpixel, taillefenetre)

        # Calculate fluence
        beam_profile = np.abs(beam_field)**2
        fluence = (pulse_FWHM*3e8*8.85e-12*np.abs(beam_field)**2)/(2*0.94)*1e4 #calcul à partir du chp électrique V/m
        theoretical = theoretical_waist(waist, z, landa)

        # Plot results
        x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
        y = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x, y, fluence, shading='auto', cmap='inferno')
        plt.colorbar(label='Fluence (J/cm^2)')
        plt.title(f"Beam Fluence after {z} m propagation\nTheoretical waist: {theoretical:.2e} m\nMax Fluence: {np.max(np.max(fluence))} J/cm^2")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis('equal')
        plt.show()

#        print('max fluence1 : ',np.max(np.max(fluence)),' J/cm2')
#        print('max fluence2 : ',np.max(np.max(fluence2)),' J/cm2')

    except ValueError:
        tk.messagebox.showerror("Input Error", "Please enter valid numerical values.")

root = tk.Tk()
root.title("Laser Beam Propagation Simulator")  

# Input fields
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

parameters = [
    ("Number of pixels", 256),
    ("Beam waist (m)", 1e-3),
    ("Window size (m)", 0.01),
    ("Propagation distance (m)", 1.0),
    ("Wavelength (m)", 1030e-9),
    ("Pulse energy (J)", 1e-6),
    ("Pulse FWHM (s)", 1e-13),
]

entries = []
for i, (label, default) in enumerate(parameters):
    ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W)
    entry = ttk.Entry(frame, width=15)
    entry.grid(row=i, column=1)
    entry.insert(0, str(default))
    entries.append(entry)

entry_nbpixel, entry_waist, entry_taillefenetre, entry_z, entry_landa, entry_pulse_energy, entry_pulse_FWHM = entries

# Run button
button_run = ttk.Button(frame, text="Run Simulation", command=run_simulation)
button_run.grid(row=len(parameters), columnspan=2)

root.mainloop()

