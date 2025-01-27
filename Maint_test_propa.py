#erreur sur les fluences!!!
import numpy as np
import matplotlib.pyplot as plt
from propagation import propagation

# Paramètres
landa = 1030e-9  # Longueur d'onde en mètres (1030 nm)
waist_0 = 1e-3  # Waist initial (1 mm)
z = 1.0  # Distance de propagation en mètres
nbpixel = 512  # Nombre de pixels (grille NxN)
taillefenetre = 0.01  # Taille de la fenêtre physique (10 mm)
energie_impulsion = 1e-6  # Énergie par impulsion (1 microjoule)

# Calcul du waist théorique après propagation
z_r = np.pi * waist_0**2 / landa  # Distance de Rayleigh
waist_z = waist_0 * np.sqrt(1 + (z / z_r)**2)  # Waist après propagation

# Calcul de la fluence maximale théorique au centre du faisceau initial
fluence_max_theorique = (2 * energie_impulsion) / (np.pi * waist_0**2) * 1e-4  # Conversion en J/cm²

# Création de la source gaussienne (front d'onde plan)
x = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
y = np.linspace(-taillefenetre / 2, taillefenetre / 2, nbpixel)
X, Y = np.meshgrid(x, y)
r2 = X**2 + Y**2
source = np.exp(-2 * r2 / waist_0**2)  # Source gaussienne (facteur 2 pour une définition d'intensité)

# Normalisation de la source pour obtenir une énergie totale de 1 µJ
aire_totale = np.sum(source) * (taillefenetre / nbpixel)**2  # Aire sous la courbe
source = source * (energie_impulsion / aire_totale)  # Normalisation en énergie

# Propagation
image = propagation(source, z, landa, nbpixel, taillefenetre)

# Calcul des fluences (en J/cm²)
dx = taillefenetre / nbpixel  # Taille d'un pixel en mètres
fluence_initiale = source / (dx**2) * 1e-4  # Conversion en J/cm²
fluence_finale = np.abs(image)**2 / (dx**2) * 1e-4  # Conversion en J/cm²

# Affichage des résultats
plt.figure(figsize=(10, 6))

# Fluence initiale
plt.subplot(1, 2, 1)
plt.imshow(fluence_initiale, extent=[x[0], x[-1], y[0], y[-1]], cmap='hot')
plt.colorbar(label="Fluence (J/cm²)")
plt.title("Fluence initiale (waist = 1 mm)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

# Fluence après propagation
plt.subplot(1, 2, 2)
plt.imshow(fluence_finale, extent=[x[0], x[-1], y[0], y[-1]], cmap='hot')
plt.colorbar(label="Fluence (J/cm²)")
plt.title(f"Fluence après propagation (z = {z} m)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.tight_layout()
plt.show()

# Résultat théorique
print(f"Waist initial : {waist_0 * 1e3:.2f} mm")
print(f"Waist théorique après propagation : {waist_z * 1e3:.2f} mm")
print(f"Énergie totale par impulsion : {energie_impulsion * 1e6:.2f} µJ")
print(f"Fluence maximale théorique initiale : {fluence_max_theorique:.2e} J/cm²")
print(f"Fluence maximale initiale mesurée : {np.max(fluence_initiale):.2e} J/cm²")

