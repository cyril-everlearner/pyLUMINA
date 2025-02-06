import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from scipy.constants import c, pi
import tkinter as tk
from tkinter import ttk


def gaussian_source(nbpixel, waist, taillefenetre, pulse_energy):
    """
    Génère un champ électrique gaussien normalisé en V/m,
    de telle sorte que l'énergie totale de l'impulsion corresponde à pulse_energy.

    Paramètres :
    - nbpixel : int, nombre de pixels de la grille
    - waist : float, taille du waist du faisceau en mètres
    - taillefenetre : float, taille de la fenêtre physique en mètres
    - pulse_energy : float, énergie de l'impulsion en Joules

    Retour :
    - champ_gaussien : ndarray, champ électrique en V/m
    """

    # Grille spatiale
    x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    y = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    X, Y = np.meshgrid(x, y)

    # Profil gaussien non normalisé (champ électrique en amplitude)
    E_field = np.exp(-2 * (X**2 + Y**2) / waist**2)

    # Calcul de l'énergie totale dans la fenêtre
    area_per_pixel = (taillefenetre / nbpixel)**2  # Aire d'un pixel en m²
    intensity = E_field**2  # Intensité proportionnelle à |E|²
    total_energy = np.sum(intensity) * area_per_pixel  # Énergie en Joules

    # Normalisation pour obtenir l'énergie totale correcte
    scaling_factor = np.sqrt(pulse_energy / total_energy)  # √ car E_field² donne l'intensité
    E_field *= scaling_factor  # Mise à l'échelle du champ électrique
    print(np.max(np.max(E_field)))
    print(np.max(np.max(intensity)))
    print(total_energy)
 # Plot results
    x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    y = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x, y, intensity, shading='auto', cmap='inferno')
    plt.colorbar(label='intensity (J/cm^2)')
    plt.axis('equal')
    plt.show()

    return E_field  # Retourne un champ électrique en V/m


def propagation(source, z, landa, nbpixel, taillefenetre):
    """
    Fonction de propagation en approximation scalaire non paraxiale.
    vérifie si conservation énergie à 1% près
    
    Paramètres :
    - source : np.ndarray, la matrice représentant le plan objet.
    - z : float, la distance de propagation (en mètres).
    - landa : float, la longueur d'onde (en mètres).
    - nbpixel : int, le nombre de pixels dans une dimension.
    - taillefenetre : float, la taille physique de la fenêtre de calcul (en mètres).
    
    Retourne :
    - image : np.ndarray, la matrice représentant le plan image après propagation.
    """
    nb = nbpixel // 2  # Position centrale
    energy_input = np.sum(np.abs(source)**2) # Calcul Energie (somme des pixels)

    # Transformée de Fourier de la source
    TFsource = np.fft.fft2(np.fft.fftshift(source))
    
    # Création de la grille de coordonnées
    m, n = np.meshgrid(range(1, nbpixel + 1), range(1, nbpixel + 1), indexing='ij')
    
    # Noyau de propagation
    noyau1 = np.exp(1j * (2 * np.pi / landa) * z * np.sqrt(1 - landa**2 * 
              (((m - nb - 1)**2 + (n - nb - 1)**2) / taillefenetre**2)))
    
    # FFT shift du noyau
    noyau2 = np.fft.fftshift(noyau1)
    
    # Plan image
    image = np.fft.ifftshift(np.fft.ifft2(TFsource * noyau2))
    
    energy_output = np.sum(np.abs(image)**2) # Calcul Energie (somme des pixels)

    # test conservation energie
    if (energy_output>energy_input*1.01) or (energy_output<energy_input*0.99):
        tk.messagebox.showerror("Calculus Error", "Breaking the energy conservation law (see propagation function)")
    print(energy_input)
    print(energy_output)
    return image

def calculate_fluence(beam_profile, pulse_energy, taillefenetre):
    """
    Calcule la fluence (J/cm²) à partir du profil d'intensité du faisceau.

    Paramètres :
    - beam_profile : ndarray, matrice du profil d'intensité (normalisé ou en intensité relative)
    - pulse_energy : float, énergie de l'impulsion en Joules
    - taillefenetre : float, taille physique de la fenêtre d'analyse en mètres

    Retour :
    - fluence : ndarray, fluence en J/cm²
    """

    # Taille d'un pixel en m²
    area_per_pixel = (taillefenetre / beam_profile.shape[0])**2  # en m²

    # Normalisation de l'intensité pour que la somme corresponde bien à l'énergie totale
    total_intensity = np.sum(beam_profile) * area_per_pixel  # en Joules

    # Facteur d'échelle pour ramener l'énergie totale au niveau souhaité
    scaling_factor = pulse_energy / total_intensity  # Normalisation correcte

    # Calcul de la fluence (conversion m² → cm² avec 1e4)
    fluence = beam_profile * scaling_factor / (area_per_pixel * 1e4)  # en J/cm²

    return fluence

def theoretical_waist(waist, z, landa):
    z_r = pi * waist**2 / landa
    return waist * np.sqrt(1 + (z / z_r)**2)

# --- GUI Setup ---
def run_simulation():
    try:
        nbpixel = int(entry_nbpixel.get())
        waist = float(entry_waist.get())
        taillefenetre = float(entry_taillefenetre.get())
        z = float(entry_z.get())
        landa = float(entry_landa.get())
        pulse_energy = float(entry_pulse_energy.get())

        # Generate source and propagate
        source = gaussian_source(nbpixel, waist, taillefenetre, pulse_energy)
        beam_field = propagation(source, z, landa, nbpixel, taillefenetre)

        # Calculate fluence
        beam_profile = np.abs(beam_field)**2
        fluence = calculate_fluence(beam_profile, pulse_energy, taillefenetre)
        theoretical = theoretical_waist(waist, z, landa)

        # Plot results
        x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
        y = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x, y, fluence, shading='auto', cmap='inferno')
        plt.colorbar(label='Fluence (J/cm^2)')
        plt.title(f"Beam profile after {z} m propagation\nTheoretical waist: {theoretical:.2e} m")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis('equal')
        plt.show()

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
]

entries = []
for i, (label, default) in enumerate(parameters):
    ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W)
    entry = ttk.Entry(frame, width=15)
    entry.grid(row=i, column=1)
    entry.insert(0, str(default))
    entries.append(entry)

entry_nbpixel, entry_waist, entry_taillefenetre, entry_z, entry_landa, entry_pulse_energy = entries

# Run button
button_run = ttk.Button(frame, text="Run Simulation", command=run_simulation)
button_run.grid(row=len(parameters), columnspan=2)

root.mainloop()

