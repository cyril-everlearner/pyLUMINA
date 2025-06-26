import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi

# --- Paramètres physiques ---
wavelength = 800e-9        # [m]
w0 = 1.1e-6                  # Waist [m]
tau = 100e-15              # Pulse duration [s]
P_Pcr = 1                 # Power ratio
n0 = 1.45                  # Refractive index of fused silica
n2 = 2.5e-20               # Nonlinear index of fused silica [m^2/W]
f = 1e-3                   # Focal length [m]
k0 = 2 * pi * n0 / wavelength
zr = pi * w0**2 * n0 / wavelength

# --- Puissance critique et énergie ---
Pcr = 0.148 * wavelength**2 / (n0 * n2)    # [W]
P = P_Pcr * Pcr
E_pulse = P * tau         # [J]
E_pulse_uJ = E_pulse * 1e6

# --- Distance de Marburger ---
sqrt_term = np.sqrt( (np.sqrt(P_Pcr) - 0.852)**2-0.0219)
znf = 0.367 * zr / sqrt_term
delta = (f**2)/(f+znf)

print(f"Zr = {zr*1e3:.2f} mm")
print(f"Pcr = {Pcr:.2e} W")
print(f"P = {P:.2e} W")
print(f"E_pulse = {E_pulse_uJ:.2f} µJ")
print(f"znf = {znf*1e6:.2f} um")
print(f"delta = {delta*1e6:.2f} um")
