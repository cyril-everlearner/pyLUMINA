"""
Functions for laser propagation simulations

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
import matplotlib.ticker as ticker

import matplotlib.pyplot as plt
from scipy.constants import c, pi
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from datetime import datetime
import os
import imageio
from scipy.signal import fftconvolve

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.constants import pi
from scipy.ndimage import gaussian_filter

import plotly.graph_objects as go  # pour la fig 3D


def apply_lens_phase(source, f, taillefenetre, landa):
    """
    Apply a parabolic phase shift to simulate a thin lens of focal length f.
    
    Parameters:
    source (numpy array): The input electric field distribution (complex).
    f (float): Focal length of the lens (m).
    taillefenetre (float): Spatial window size (m).
    landa (float): Wavelength of the beam (m).

    Returns:
    numpy array: The modified electric field after passing through the lens.
    """
    nbpixel = source.shape[0]  # Assuming square grid
    k = 2 * np.pi / landa  # Wavenumber
    
    # Create spatial coordinate grid
    x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    X, Y = np.meshgrid(x, x)
    
    # Apply the quadratic phase shift of a thin lens
    phase_lens = np.exp(-1j * k * (X**2 + Y**2) / (2 * f))
    
    return source * phase_lens

def apply_tilt_phase(source, tilt_range, taillefenetre, landa):
    """
    Apply a parabolic phase shift to simulate a tilt of tilt_range rad
    over the pupil.
    
    Parameters:
    source (numpy array): The input electric field distribution (complex).
    tilt_range (float): tilt in rad.
    taillefenetre (float): Spatial window size (m).
    landa (float): Wavelength of the beam (m).

    Returns:
    numpy array: The modified electric field after adding the tilt.
    """
    nbpixel = source.shape[0]  # Assuming square grid
    k = 2 * np.pi / landa  # Wavenumber
    
    # Create spatial coordinate grid
    x = np.linspace(0, tilt_range, nbpixel)
    X, Y = np.meshgrid(x, x)
        
    # Apply the tilt phase
    phase_tilt = np.exp(-1j * k * tilt_range *Y)
    
    return source * phase_tilt


def gaussian_source(nbpixel, waist, taillefenetre, pulse_energy, pulse_FWHM):
    """
    Génère un champ électrique gaussien normalisé en V/m,
    de telle sorte que l'énergie totale de l'impulsion corresponde à pulse_energy.

    Paramètres :
    - nbpixel : int, nombre de pixels de la grille
    - waist : float, taille du waist du faisceau en mètres
    - taillefenetre : float, taille de la fenêtre physique en mètres
    - pulse_energy : float, énergie de l'impulsion en Joules

    Retour :
    - E_field_Vpm : ndarray, champ électrique en V/m
    """

    # Grille spatiale
    x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    y = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    X, Y = np.meshgrid(x, y)

    # Profil gaussien normalisé (champ électrique en amplitude, pas de facteur 2 en haut de l'argument de la gaussienne')
    E_field = np.exp(- (X**2 + Y**2) / waist**2)
    E_field = E_field/np.max(E_field)

    # Profil d'intensité
    beam_int_profile = np.abs(E_field)**2

    # Taille d'un pixel en m²
    area_per_pixel_cm = (taillefenetre*100 / nbpixel**2)  # en cm²

    # Somme de tous les ndg -> cela correspond à toute l'énergie de l'impulsion
    total_intensity = np.sum(beam_int_profile)

    # le quantum d'énergie (par pixel et par niveau de gris)
    quantum = pulse_energy/(total_intensity*area_per_pixel_cm)

    # Calcul de la fluence (J/cm²)
    fluence = beam_int_profile * quantum  # en J/cm²

    # Calcul du champ élec en V/m
    E_field_Vpm = np.sqrt(2*0.94*fluence*0.0001/(2.6550e-03*pulse_FWHM))


# # Pour les tests, pour voir ce qu'il se passe. 
#    print('max elec field : ',np.max(np.max(E_field_Vpm)),' V/m')
#    print('max fluence : ',np.max(np.max(fluence)),' J/cm2')
#    
# # Plot results
#    x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
#    y = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
#    plt.figure(figsize=(10, 8))
#    plt.pcolormesh(x, y, fluence, shading='auto', cmap='inferno')
#    plt.colorbar(label='fluence de la source (J/cm^2)')
#    plt.axis('equal')
#    plt.show()
#
#    x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
#    y = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
#    plt.figure(figsize=(10, 8))
#    plt.pcolormesh(x, y, fluence, shading='auto', cmap='inferno')
#    plt.colorbar(label='E_field_Vpm de la source (V/m)')
#    plt.axis('equal')
#    plt.show()

    return E_field_Vpm  # Retourne un champ électrique en V/m


def propagation(source, z, landa, nbpixel, taillefenetre):
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
#    print('verif conservation energie input',energy_input)
#    print('verif conservation energie output',energy_output)
    return image


def propagation_n2(source, z, landa, nbpixel, taillefenetre, n2=3.2e-19, dz=1e-6):
    """
    Fonction de propagation scalaire non paraxiale avec effet non linéaire n2.
    Utilise la méthode BPM (Beam Propagation Method) en ajoutant une phase non linéaire à chaque pas dz.
    
    Paramètres :
    - source : np.ndarray, champ électrique initial (V/m).
    - z : float, distance de propagation (m).
    - landa : float, longueur d'onde (m).
    - nbpixel : int, nombre de pixels dans une dimension.
    - taillefenetre : float, taille physique de la fenêtre de calcul (m).
    - n2 : float, coefficient de Kerr (m²/W) (défaut : valeur élevée du CS₂).
    - dz : float, pas de propagation (m) (défaut : 1 µm).

    Retourne :
    - image : np.ndarray, champ électrique après propagation.
    """
    nb = nbpixel // 2  # Position centrale
    k0 = 2 * pi / landa  # Nombre d'onde dans le vide
    energy_input = np.sum(np.abs(source)**2)  # Énergie d'entrée

    # Grille spatiale
    m, n = np.meshgrid(range(1, nbpixel + 1), range(1, nbpixel + 1), indexing='ij')

    # Noyau de propagation scalaire non paraxiale
    noyau1 = np.exp(1j * k0 * dz * np.sqrt(1 - landa**2 * 
              (((m - nb - 1)**2 + (n - nb - 1)**2) / taillefenetre**2)))
    noyau2 = np.fft.fftshift(noyau1)

    # Initialisation du champ
    E = source.astype(np.complex128)
    
    # Nombre d'étapes en dz
    steps = int(z / dz)

    for _ in range(steps):
        # Appliquer la phase non linéaire
        I = np.abs(E)**2  # Intensité locale
        E *= np.exp(1j * k0 * n2 * I * dz)

        # Transformée de Fourier
        TF_E = np.fft.fft2(np.fft.fftshift(E))

        # Propagation dans l'espace de Fourier
        TF_E *= noyau2

        # Retour à l'espace direct
        E = np.fft.ifftshift(np.fft.ifft2(TF_E))

    energy_output = np.sum(np.abs(E)**2)  # Énergie de sortie

    # Vérification conservation d'énergie
    if not (0.99 * energy_input <= energy_output <= 1.01 * energy_input):
        tk.messagebox.showerror("Calculus Error", "Breaking the energy conservation law (see propagationn2 function)")

    return E


def circular_pupil(source, width, taillefenetre):
    """
    Tronque le champ électrique laser par un disque de diamètre 'width'.
    
    Paramètres :
        source (numpy.ndarray) : Matrice du champ électrique.
        width (float) : Diamètre du disque de troncature en mètres.
        taillefenetre (float) : Taille de la matrice source en mètres.
    
    Retourne :
        source_truncated (numpy.ndarray) : Matrice du champ après troncature.
        energy_loss (float) : Pourcentage d'énergie perdue par la troncature.
    """
    nbpixel = source.shape[0]  # Nombre de pixels dans chaque dimension
    x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    y = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    X, Y = np.meshgrid(x, y)
    
    # Création du masque circulaire
    radius = width / 2
    mask = (X**2 + Y**2) <= radius**2

    # Calcul de l'énergie initiale et après troncature
    initial_energy = np.sum(source**2)
    truncated_source = np.copy(source)
    truncated_source[~mask] = 0
    final_energy = np.sum(truncated_source**2)

    # Calcul du pourcentage d'énergie perdue
    energy_loss = 100 * (1 - final_energy / initial_energy)
    print(f"Énergie perdue par troncation : {energy_loss:.2f} %")

    return truncated_source

def airy_disk_radius(wavelength, aperture_diameter, distance):
    """Calcule la largeur de la tache d'Airy théorique après diffraction."""
    return 1.22 * wavelength * distance / aperture_diameter

def theoretical_waist(waist, z, landa):
    z_r = pi * waist**2 / landa
    return waist * np.sqrt(1 + (z / z_r)**2)

def apply_axiconic_phase(field, angle, taillefenetre, landa):
    """
    Applique une phase axiconique au champ laser.
    
    :param field: Champ laser complexe.
    :param angle: Angle physique de l'axicon en degrés.
    :param taillefenetre: Taille de la fenêtre spatiale (m).
    :param landa: Longueur d'onde du laser (m).
    :return: Champ modifié avec la phase axiconique.
    """
    nbpixel = field.shape[0]
    k = 2 * np.pi / landa  # Nombre d'onde
    alpha = np.radians(angle)  # Conversion en radians

    # Création des coordonnées spatiales
    x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    y = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    X, Y = np.meshgrid(x, y)

    # Calcul du rayon en coordonnées polaires
    R = np.sqrt(X**2 + Y**2)

    # Phase axiconique : exp(i k R sin(alpha))
    phase_axicon = np.exp(-1j * k * R * np.sin(alpha))

    return field * phase_axicon


def apply_aperture(source, aperture_type, width, taillefenetre):
    """ Tronque le champ selon l'ouverture choisie (disk, square, triangle, annulus). """
    nbpixel = source.shape[0]
    x = np.linspace(-taillefenetre/2, taillefenetre/2, nbpixel)
    y = np.linspace(-taillefenetre/2, taillefenetre/2, nbpixel)
    X, Y = np.meshgrid(x, y)
    
    aperture = np.zeros_like(source)

    if aperture_type == "disk":
        aperture = (X**2 + Y**2 <= (width/2)**2).astype(float)
    elif aperture_type == "square":
        aperture = (np.abs(X) <= width/2) & (np.abs(Y) <= width/2)
    elif aperture_type == "triangle":
        H = width * np.sqrt(3) / 2  # Hauteur du triangle équilatéral
        aperture = (Y >= -H/2) & (Y <= (H/width) * (width/2 - np.abs(X)))
    elif aperture_type == "annulus":
        outer_radius = width / 2
        inner_radius = outer_radius * 0.9  # Épaisseur = 10% du diamètre
        aperture = ((X**2 + Y**2 <= outer_radius**2) & (X**2 + Y**2 >= inner_radius**2)).astype(float)

    return source * aperture

def apply_cubic_phase(field, taillefenetre, nbpixel, cubic_coeff):
    """Applique une phase cubique pour générer un Airy beam."""
    x = np.linspace(-1/2, 1/2, nbpixel)
    X, Y = np.meshgrid(x, x)
    r3 = X**3 + Y**3  # Phase cubique sur pupille normalisée
    phi_cubic = cubic_coeff * r3
    return field * np.exp(1j * phi_cubic)

def apply_parabolic_phase(field, taillefenetre, nbpixel, Second_order_coeff):
    """adds a parabolic phase modulation."""
    x = np.linspace(-1/2, 1/2, nbpixel)
    X, Y = np.meshgrid(x, x)
    r2 = X**2  # Phase parabolique sur pupille normalisée
    phi_parabol = Second_order_coeff * r2
    return field * np.exp(1j * phi_parabol)

def apply_helical_phase(field, l, taillefenetre, nbpixel):
    """Applies a helical phase to the laser field."""
    x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
    X, Y = np.meshgrid(x, x)
    theta = np.angle(X + 1j * Y)
    return field * np.exp(1j * l * theta)

def apply_IFTA_phase(source, taillefenetre, landa, f, n_spots, iterations=50):
    """Computes the phase mask for multispot generation using the IFTA method."""
    nbpixel = source.shape[0]
    k = 2 * pi / landa

    # Compute focal spot size (Airy disk radius)
    waist_focal = 1.22 * landa * f / (2 * (taillefenetre / nbpixel))
    #print("taille spot (m)",waist_focal)

    # Define target amplitude in the Fourier plane
    target = np.zeros((nbpixel, nbpixel), dtype=np.complex128)
    center = nbpixel // 2
    spacing = 20#int(50 * waist_focal / (taillefenetre / nbpixel))  # Separation between spots

    for i in range(n_spots):
        x_pos = center + (i - (n_spots - 1) / 2) * spacing
        if 0 <= x_pos < nbpixel:
            target[nbpixel // 2, int(x_pos)] = 1  # Line of spots along X

    # Initialize phase with backpropagated phase of the target
    field = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(target)))
    phase = np.angle(field)

    # IFTA iterations
    amplitude_source = np.abs(source)
    for _ in range(iterations):
        field = amplitude_source * np.exp(1j * phase)  # Apply phase to initial amplitude
        field_f = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field)))  # Forward FFT
        field_f = target * np.exp(1j * np.angle(field_f))  # Enforce target amplitude
        field = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(field_f)))  # Backward FFT
        phase = np.angle(field)  # Extract updated phase

    return source * np.exp(1j * phase)  # Apply optimized phase




def apply_IFTA_top_hat(source, taillefenetre, landa, f, ratio_radius_on_waist, iterations=50):
    """
    Computes the phase mask for top-hat generation using the IFTA method, 
    with a soft-edged target obtained by convolving a binary top-hat 
    with a Gaussian approximation of the focal spot (PSF).
    
    Parameters:
    - source: 2D complex array, initial field amplitude at SLM.
    - taillefenetre: physical size of simulation window (in meters).
    - landa: wavelength (in meters).
    - f: focal length of the lens (in meters).
    - ratio_radius_on_waist: scaling factor of top-hat radius over Airy disk radius.
    - iterations: number of IFTA iterations.

    Returns:
    - final field with optimized phase applied: 2D complex array.
    """
    nbpixel = source.shape[0]
    k = 2 * pi / landa

    # Compute Airy disk (focal spot) size
    dx = taillefenetre / nbpixel
    waist_focal = 1.22 * landa * f / (2 * dx)

    # Coordinate system
    x = np.linspace(-taillefenetre/2, taillefenetre/2, nbpixel)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    # Binary top-hat
    radius = ratio_radius_on_waist * waist_focal
    binary_target = (R <= radius).astype(float)

    # Gaussian approximation of PSF
    sigma = waist_focal / 2.3548  # FWHM to sigma
    PSF = np.exp(-R**2 / (2 * sigma**2))
    PSF /= PSF.sum()

    # Convolve to obtain soft target
    target = fftconvolve(binary_target, PSF, mode='same')
    target /= target.max()

    # Initialize phase using back-propagated target
    field = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(target)))
    phase = np.angle(field)

    # IFTA iterations
    amplitude_source = np.abs(source)
    for _ in range(iterations):
        field = amplitude_source * np.exp(1j * phase)
        field_f = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field)))
        field_f = target * np.exp(1j * np.angle(field_f))
        field = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(field_f)))
        phase = np.angle(field)

    return source * np.exp(1j * phase)


def apply_blur(field, blur_sigma):
    """
    Applique un flou gaussien à la phase spatiale d'un faisceau laser.
    
    Paramètres :
    - field : np.ndarray, champ laser (complexe).
    - blur_sigma : float, largeur du flou gaussien (en pixels).
    
    Retourne :
    - np.ndarray : champ avec phase floutée.
    """
    # Extraire l'amplitude et la phase
    amplitude = np.abs(field)
    phase = np.angle(field)
    
    # Appliquer un flou gaussien à la phase
    blurred_phase = gaussian_filter(phase, sigma=blur_sigma)
    
    # Reconstruire le champ avec la phase floutée
    return amplitude * np.exp(1j * blurred_phase)


def plot_propagation_2D(fields, z_planes, taillefenetre, cmap="inferno"):
    """ Affiche une visualisation de la fluence propagée."""
    # Création de la figure
    global fig, ax
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    
    # Affichage d'une coupe transverse du champ à z=z_final
    im = ax[0, 0].imshow(fields[-1], extent=[-taillefenetre/2, taillefenetre/2, -taillefenetre/2, taillefenetre/2], cmap=cmap)
    ax[0, 0].set_title(f"Fluence (J/cm²) @ z={z_planes[-1]:.2f} m")
    ax[0, 0].set_xlabel("y (m)")
    ax[0, 0].set_ylabel("x (m)")
    plt.colorbar(im, ax=ax[0, 0])
    
    # Coupe 2D du profil axial (YZ) avec inversion des axes
    y_index = fields.shape[1] // 2  # Coupe au centre
    im2 = ax[0, 1].imshow(
        np.transpose(fields[:, y_index, :]),  # Transposition pour faire correspondre les axes
        aspect='auto',
        extent=[z_planes[0], z_planes[-1], -taillefenetre/2, taillefenetre/2],  # Inversion des axes
        cmap=cmap
    )
    ax[0, 1].set_title("Lateral view (J/cm² ) (YZ)")
    ax[0, 1].set_xlabel("Propagation axis (m)")
    ax[0, 1].set_ylabel("x (m)")
    plt.colorbar(im2, ax=ax[0, 1])

   # Ajout des contours pour les iso fluences
    #isovalues = [0.01353 * fields.max(), 0.1353 * fields.max(), 0.5 * fields.max(), 0.9 * fields.max()]
    x = np.linspace(-taillefenetre/2, taillefenetre/2, fields.shape[2])
    y = np.linspace(-taillefenetre/2, taillefenetre/2, fields.shape[1])
    z = np.linspace(z_planes[0], z_planes[-1], fields.shape[0])
  
    CS1 = ax[1, 0].contour(fields[-1], extent=[-taillefenetre/2, taillefenetre/2, -taillefenetre/2, taillefenetre/2])
    ax[1, 0].clabel(CS1, fontsize=10)
    ax[1, 0].set_title(f"Iso-fluences @ z={z_planes[-1]:.2f} m")
    ax[1, 0].set_xlabel("y (m)")
    ax[1, 0].set_ylabel("y (m)")
    ax[1, 0].grid(True, linestyle='dotted')
    
    CS2 = ax[1, 1].contour(np.transpose(fields[:, y_index, :]), extent=[-taillefenetre/2, taillefenetre/2, z_planes[0], z_planes[-1]])
    ax[1, 1].clabel(CS2, fontsize=10)
    ax[1, 1].set_title("Iso-fluences (YZ cut)")
    ax[1, 1].set_xlabel("y (m)")
    ax[1, 1].set_ylabel("z (m)")
    ax[1, 1].grid(True, linestyle='dotted')
    
    plt.tight_layout()
    plt.show()

def plot_phase_2D(cplx_fields, z_planes, taillefenetre, cmap="inferno"):
    """ Affiche une visualisation de la phase propagée."""
    # Création de la figure
    global fig, ax
    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    
    # Affichage de la phase du champ à z=0
    phase = np.angle(cplx_fields)
    im = ax[0].imshow(phase[0], extent=[-taillefenetre/2, taillefenetre/2, -taillefenetre/2, taillefenetre/2], cmap=cmap)
    ax[0].set_title(f"Phase (rad) @ z={z_planes[0]:.2f} m")
    ax[0].set_xlabel("y (m)")
    ax[0].set_ylabel("x (m)")
    plt.colorbar(im, ax=ax[0])
    
    # Coupe 2D du profil axial (YZ) avec inversion des axes
    y_index = phase.shape[1] // 2  # Coupe au centre
    im2 = ax[1].imshow(
        np.transpose(phase[:, y_index, :]),  # Transposition pour faire correspondre les axes
        aspect='auto',
        extent=[z_planes[0], z_planes[-1], -taillefenetre/2, taillefenetre/2],  # Inversion des axes
        cmap=cmap
    )
    ax[1].set_title("Lateral view (rad) (YZ)")
    ax[1].set_xlabel("Propagation axis (m)")
    ax[1].set_ylabel("x (m)")
    plt.colorbar(im2, ax=ax[1])
       
    plt.tight_layout()
    plt.show()



def plot_propagation_3D(fields, z_planes, taillefenetre, recordGIF=False, parameters=None, apply_lens=False, cmap="inferno"):
    """ Affiche la fluence propagée en 3D et enregistre un GIF si demandé. """
    
    # Définition des axes
    max_fluence = fields.max()
    min_fluence = fields.min()
    nbpixel = fields.shape[1]
    
    X, Y, Z = np.meshgrid(
        z_planes,
        np.linspace(-taillefenetre/2, taillefenetre/2, nbpixel),
        np.linspace(-taillefenetre/2, taillefenetre/2, nbpixel),
        indexing='ij'
    )
    
    # Création de la figure 3D
    fig_3d = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=fields.flatten(),
        isomin=0.01353 * max_fluence,  # iso surface at 0.1/e2
        isomax=0.9 * max_fluence,
        opacity=0.2,  # needs to be small to see through all surfaces
        surface_count=30,  # nombre des isosurfaces
        colorbar=dict(title="Fluence (J/cm²)")
    ))
    
    fig_3d.update_layout(
        title=f"Fluence max: {max_fluence:.2e} J/cm²",
        scene=dict(
            xaxis_title="Propagation Axis (m)",
            yaxis_title="y (m)",
            zaxis_title="x (m)"
        )
    )

    # Affichage
    fig_3d.show()

    # Si enregistrement GIF demandé
# Si enregistrement GIF demandé
    if recordGIF:
        # Boîte de dialogue pour choisir le dossier de sauvegarde
        root = tk.Tk()
        root.withdraw()  # Ne pas afficher la fenêtre principale Tkinter
        save_dir = filedialog.askdirectory(title="Choisir le dossier de sauvegarde")
        root.destroy()
        if not save_dir:  # Si l'utilisateur annule, ne rien faire
            print("Sauvegarde annulée.")
            return

        # Création du sous-dossier pour les PNGs des colormaps
        png_save_dir = os.path.join(save_dir, "GIFSpngs")
        os.makedirs(png_save_dir, exist_ok=True)

        # Générer le nom des fichiers avec date et heure
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        color_gif_filename = os.path.join(save_dir, f"GIFstack_colormap_{timestamp}.gif")
        grey_gif_filename = os.path.join(save_dir, f"GIFstack_greylevels_{timestamp}.gif")
        readme_filename = os.path.join(save_dir, f"GIF_readme_{timestamp}.txt")

        # Création des images pour le GIF
        images_color = []
        images_grey = []

        for i in range(fields.shape[0]):  # Boucle sur les plans de propagation
            fluence_map = fields[i]

            # Image en fausse couleur
            fig, ax = plt.subplots()
            cax = ax.imshow(fluence_map, cmap=cmap, extent=[-taillefenetre/2, taillefenetre/2, -taillefenetre/2, taillefenetre/2])
            plt.colorbar(cax, label="Fluence (J/cm²)")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            plt.title(f"Fluence - Plane {i+1}/{fields.shape[0]}")
            
            temp_color_path = "temp_color.png"
            plt.savefig(temp_color_path, dpi=100)
            plt.close()

            images_color.append(imageio.imread(temp_color_path))

            # Sauvegarde de l'image colormap en PNG dans GIFSpngs
            png_filename = os.path.join(png_save_dir, f"frame_{i+1}.png")
            os.rename(temp_color_path, png_filename)  # Déplace le fichier temporaire
    
            # Image en niveaux de gris (8 bits)
            fluence_map_normalized = (fluence_map - min_fluence) / (max_fluence - min_fluence) * 255
            fluence_map_8bit = fluence_map_normalized.astype(np.uint8)
            imageio.imwrite("temp_grey.png", fluence_map_8bit)
            images_grey.append(imageio.imread("temp_grey.png"))

        # Création du GIF
        imageio.mimsave(color_gif_filename, images_color, duration=0.1)
        imageio.mimsave(grey_gif_filename, images_grey, duration=0.1)

        # Suppression du fichier temporaire grey
        os.remove("temp_grey.png")

        # Création du fichier README
        with open(readme_filename, "w") as f:
            f.write(f"GIF Simulation - {timestamp}\n")
            f.write(f"Min Fluence: {min_fluence:.2e} J/cm²\n")
            f.write(f"Max Fluence: {max_fluence:.2e} J/cm²\n\n")
            f.write(f"Lens is applied? : {'Yes' if apply_lens else 'No'}\n\n")
            f.write("Simulation Parameters:\n")
            for param_name, param_value in parameters:
                f.write(f"{param_name}: {param_value}\n")

        print(f"GIF en fausse couleur sauvegardé sous : {color_gif_filename}")
        print(f"GIF en niveaux de gris sauvegardé sous : {grey_gif_filename}")
        print(f"Images colormap sauvegardées dans : {png_save_dir}")
        print(f"Fichier README sauvegardé sous : {readme_filename}")













