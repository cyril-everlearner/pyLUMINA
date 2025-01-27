import numpy as np

def propagation(source, z, landa, nbpixel, taillefenetre):
    """
    Fonction de propagation en approximation scalaire non paraxiale.
    
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
    
    return image

