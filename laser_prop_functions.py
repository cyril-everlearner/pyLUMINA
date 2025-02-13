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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
# # Intégration dans l'interface Tkinter
#    global canvas
#    if 'canvas' in globals():
#        canvas.get_tk_widget().destroy()  # Supprime l'ancien graphique si présent
#  
#    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
#    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
#    canvas.draw()


def plot_propagation_3D(fields, z_planes, taillefenetre):
    """ Affiche une visualisation de la fluence propagée en 3D"""
    # rendu 3D
    max_fluence = fields.max()
    nbpixel = fields.shape[1]
   
    X, Y, Z = np.meshgrid(
            z_planes,
            np.linspace(-taillefenetre/2, taillefenetre/2, nbpixel),
            np.linspace(-taillefenetre/2, taillefenetre/2, nbpixel),
            indexing='ij'
    )
    fig_3d = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=fields.flatten(),
        isomin=0.01353*max_fluence, # iso surface at 0.1/e2
        isomax=0.9*max_fluence,
        opacity=0.2, # needs to be small to see through all surfaces
        surface_count=30, # nombre des isosurfaces
        colorbar=dict(title="Fluence (J/cm²)")
        ))
"""    fig_3d = go.Figure()  # pour des iso surfaces à des fluences précises
#    isovalues = [0.01353 * max_fluence, 0.1353 * max_fluence, 0.5 * max_fluence, 0.9 * max_fluence]
#    colors = ['black', 'purple', 'red', 'yellow']
#
#    for iso, color in zip(isovalues, colors):
#        fig_3d.add_trace(go.Isosurface(
#            x=X.flatten(),
#            y=Y.flatten(),
#            z=Z.flatten(),
#            value=fields.flatten(),
#            isomin=iso,
#            isomax=iso,
#            opacity=0.3,
#            surface_count=1,
#            caps=dict(x_show=False, y_show=False, z_show=False),
#            colorscale=[[0, color], [1, color]],
#            showscale=True,
#            colorbar=dict(title="Fluence (J/cm²)")
#        ))
#    
#        
    fig_3d.update_layout(
        title=f"Fluence max: {max_fluence:.2e} J/cm²",
        scene=dict(
            xaxis_title="Propagation Axis (m)",
            yaxis_title="y (m)",
            zaxis_title="x (m)"
        )
    )


    plt.show()
    fig_3d.show()
"""
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import imageio
import os
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

def plot_propagation_3D(fields, z_planes, taillefenetre, recordGIF=False, parameters=None):
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
    if recordGIF:
        # Boîte de dialogue pour choisir le dossier de sauvegarde
        root = tk.Tk()
        root.withdraw()  # Ne pas afficher la fenêtre principale Tkinter
        save_dir = filedialog.askdirectory(title="Choisir le dossier de sauvegarde")
        if not save_dir:  # Si l'utilisateur annule, ne rien faire
            print("Sauvegarde annulée.")
            return
        
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
            cax = ax.imshow(fluence_map, cmap="inferno", extent=[-taillefenetre/2, taillefenetre/2, -taillefenetre/2, taillefenetre/2])
            plt.colorbar(cax, label="Fluence (J/cm²)")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            plt.title(f"Fluence - Plane {i+1}/{fields.shape[0]}")
            plt.savefig("temp_color.png", dpi=100)
            plt.close()
            images_color.append(imageio.imread("temp_color.png"))

            # Image en niveaux de gris (8 bits)
            fluence_map_normalized = (fluence_map - min_fluence) / (max_fluence - min_fluence) * 255
            fluence_map_8bit = fluence_map_normalized.astype(np.uint8)
            imageio.imwrite("temp_grey.png", fluence_map_8bit)
            images_grey.append(imageio.imread("temp_grey.png"))

        # Création du GIF
        imageio.mimsave(color_gif_filename, images_color, duration=0.1)
        imageio.mimsave(grey_gif_filename, images_grey, duration=0.1)

        # Suppression des fichiers temporaires
        os.remove("temp_color.png")
        os.remove("temp_grey.png")

        # Création du fichier README
        with open(readme_filename, "w") as f:
            f.write(f"GIF Simulation - {timestamp}\n")
            f.write(f"Min Fluence: {min_fluence:.2e} J/cm²\n")
            f.write(f"Max Fluence: {max_fluence:.2e} J/cm²\n\n")
            f.write("Simulation Parameters:\n")
            for param_name, param_value in parameters:
                f.write(f"{param_name}: {param_value}\n")

        print(f"GIF en fausse couleur sauvegardé sous : {color_gif_filename}")
        print(f"GIF en niveaux de gris sauvegardé sous : {grey_gif_filename}")
        print(f"Fichier README sauvegardé sous : {readme_filename}")














