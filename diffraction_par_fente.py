# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:31:48 2025

@author: Etudiant
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from scipy.constants import c, pi
import tkinter as tk
from tkinter import ttk

from laser_prop_functions import *

def calculate_fwhm(x, profile):
    """
    Calcule la largeur à mi-hauteur (FWHM) du pic central.
    
    Paramètres :
    - x : array, positions des points (en mètres)
    - profile : array, valeurs du profil
    
    Retourne :
    - fwhm : float, largeur à mi-hauteur (en mètres)
    - left_x, right_x : positions des points à mi-hauteur
    """
    # Trouver le maximum et la moitié du maximum
    max_val = np.max(profile)
    half_max = max_val / 2
    
    # Trouver le pic central (indice du maximum)
    peak_idx = np.argmax(profile)
    
    # Trouver les croisements à gauche du pic
    left_side = profile[:peak_idx]
    left_crossings = np.where(left_side <= half_max)[0]
    if len(left_crossings) > 0:
        left_idx = left_crossings[-1]  # Dernier point avant dépassement
    else:
        left_idx = 0
    
    # Trouver les croisements à droite du pic
    right_side = profile[peak_idx:]
    right_crossings = np.where(right_side <= half_max)[0]
    if len(right_crossings) > 0:
        right_idx = peak_idx + right_crossings[0]  # Premier point après dépassement
    else:
        right_idx = len(profile) - 1
    
    # Interpolation linéaire pour une meilleure précision
    def interpolate_crossing(x1, x2, y1, y2, y_target):
        return x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
    
    # Calcul précis des positions à mi-hauteur
    left_x = interpolate_crossing(x[left_idx], x[left_idx + 1], 
                                 profile[left_idx], profile[left_idx + 1], half_max)
    right_x = interpolate_crossing(x[right_idx - 1], x[right_idx], 
                                  profile[right_idx - 1], profile[right_idx], half_max)
    
    fwhm = right_x - left_x
    return fwhm, left_x, right_x


def propagation_trou(source,z,largeur_fente, landa, nbpixel, taillefenetre):
    """
    Fonction de propagation en approximation scalaire non paraxiale.
    vérifie si conservation énergie à 1% près
    Champ électrique en V/m entrée et en sortie
    
    Paramètres :
    - source : np.ndarray, la matrice représentant le plan objet.
    - z : float, la distance de propagation (en mètres).
    - landa : float, la longueur d'onde (en mètres).
    - nbpixel : int, le nombre de pixels dans une dimension.
    - taillefenetre : float, la taille physique de la fenêtre de calcul (en mètres).
    
    Retourne :
    - image : np.ndarray, la matrice représentant le plan image après propagation.
    """
    x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    y = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    X, Y = np.meshgrid(x, y)
   
    # Application d'un masque ''trou''
    masque = (np.abs(X) <= largeur_fente / 2).astype(np.complex64)
    source_apres_trou = source * masque

    nb = nbpixel // 2  # Position centrale
    energy_input = np.sum(np.abs(source_apres_trou)**2) # Calcul Energie (somme des pixels)

   
    # Transformée de Fourier de la source
    TFsource = np.fft.fft2(np.fft.fftshift( source_apres_trou))
    
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
#    print('verif conservation energie input',energy_input)
#    print('verif conservation energie output',energy_output)
    return image



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
        largeur_fente=float(entry_largeur_fente.get())

        # Generate source and propagate
        source = gaussian_source(nbpixel, waist, taillefenetre, pulse_energy,pulse_FWHM)
        beam_field = propagation_trou(source, z, largeur_fente, landa, nbpixel, taillefenetre)

        # Calculate fluence
        beam_profile = np.abs(beam_field)**2
        fluence = (pulse_FWHM*3e8*8.85e-12*np.abs(beam_field)**2)/(2*0.94)*1e4 #calcul à partir du chp électrique V/m
        theoretical = theoretical_waist(waist, z, landa)
        # Calculate rayon tache d'airy
        
        # Création de la figure
        fig = plt.figure(figsize=(15, 6))
        x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
        y = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
        center_y = nbpixel // 2
        horizontal_profile = fluence[center_y, :]
     
     # Calcul de la FWHM
        fwhm, left_x, right_x = calculate_fwhm(x, horizontal_profile)
     
     # Graphique principal (fluence)
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        im = ax1.pcolormesh(x, y, fluence, shading='auto', cmap='inferno')
        plt.colorbar(im, ax=ax1, label='Fluence (J/cm²)')
        ax1.set_title(f"Beam Fluence after {z} m propagation\nTheoretical waist: {theoretical:.2e} m\nMax Fluence: {np.max(fluence):.2e} J/cm^2 ")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.axis('equal')
       
     # Lignes centrales
        ax1.axhline(y=y[center_y], color='r', linestyle='--', linewidth=1)
    
     
     # Profil horizontal avec FWHM
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        ax2.plot(x * 1e3, horizontal_profile, 'r-', label='Profil')
        ax2.axhline(y=np.max(horizontal_profile)/2, color='g', linestyle='--', label='Mi-hauteur')
        ax2.axvline(x=left_x * 1e3, color='b', linestyle=':', label=f'distance mi hauteur(rayon) = {fwhm*1e3:.3e} mm')
        ax2.axvline(x=right_x * 1e3, color='b', linestyle=':')
        ax2.set_title('Profil Horizontal')
        ax2.set_xlabel('Position (mm)')
        ax2.set_ylabel('Fluence (J/cm²)')
        ax2.legend()
        ax2.grid(True)
        
      
     
        plt.tight_layout()
        plt.show()
        
      

     
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
    ("Number of pixels", 600),
    ("Beam waist (m)", 5e-3),
    ("Window size (m)", 0.002),
    ("Propagation distance (m)", 0.05),
    ("Wavelength (m)", 1030e-9),
    ("Pulse energy (J)", 1e-6),
    ("Pulse FWHM (s)", 1e-13),
    ("largeur_fente", 150e-5),
]


entries = []
for i, (label, default) in enumerate(parameters):
    ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W)
    entry = ttk.Entry(frame, width=15)
    entry.grid(row=i, column=1)
    entry.insert(0, str(default))
    entries.append(entry)

entry_nbpixel, entry_waist, entry_taillefenetre, entry_z, entry_landa, entry_pulse_energy, entry_pulse_FWHM,entry_largeur_fente = entries

# Run button
button_run = ttk.Button(frame, text="Run Simulation", command=run_simulation)
button_run.grid(row=len(parameters), columnspan=2)

root.mainloop()

def convergence_test():
    pixels = [500, 1000, 2000, 4000]
    results = []
    
    for n in pixels:
        source = gaussian_source(n, waist, taillefenetre, pulse_energy, pulse_FWHM)
        beam_field = propagation_trou(source, z, largeur_fente, landa, n, taillefenetre)
        fluence = calculate_fluence(beam_field)
        fwhm = calculate_fwhm(x, fluence[center_y,:])
        results.append(fwhm)
    
    plt.plot(pixels, results, 'o-')
    plt.xlabel('Number of pixels')
    plt.ylabel('FWHM (mm)')
    plt.title('Convergence Analysis')
    plt.grid(True)
    plt.show()

