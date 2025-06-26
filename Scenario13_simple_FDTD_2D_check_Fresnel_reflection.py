'''
Looking at simple FDTD 1D


This script simulates the propagation of a 10 fs laser pulse at 1030 nm using the 2D FDTD method. It includes the vacuum-glass interface and allows for observation of reflection and transmission. The animation shows the temporal evolution of the electric field.

You can now verify whether the reflected fraction corresponds to the calculated Fresnel coefficient (≈3.37%).
We can also verify the speed drop when reaching the fused silica sample by the 1.45 factor which is the refractive index at 1030nm

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



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Physical constants
c = 3e8  # Speed of light in vacuum (m/s)
dx = dy = 50e-9  # Spatial step (50 nm)
dt = dx / (2 * c)  # Time step to satisfy Courant condition

# Simulation parameters
Nx = Ny = 400  # Grid size (square grid)
Nt = 1000  # Number of time steps

# Field definitions
Ez = np.zeros((Nx, Ny))  # Electric field (z-component)
Hx = np.zeros((Nx, Ny))  # Magnetic field (x-component)
Hy = np.zeros((Nx, Ny))  # Magnetic field (y-component)

# Source parameters
lambda0 = 1030e-9  # Wavelength (m)
T0 = 10e-15  # Pulse duration (s)
omega0 = 2 * np.pi * c / lambda0  # Angular frequency
source_pos = (100, 200)  # Source position (x, y)
T_peak = 100  # Temporal peak shift

def gaussian_pulse(t):
    return np.exp(-((t - T_peak) * dt / T0) ** 2) * np.sin(omega0 * (t - T_peak) * dt)

# Material properties (vacuum on left, fused silica on right)
n_silica = 1.45  # Refractive index of fused silica
mid_x = Nx // 2  # Interface position
C1 = np.ones((Nx, Ny)) * (c * dt / dx)
C2 = np.ones((Nx, Ny)) * (c * dt / dx)
C1[mid_x:, :] /= n_silica  # Adjusted for fused silica
C2[mid_x:, :] /= n_silica

# Absorptive region on the left
absorption_length = int(5e-6 / dx)  # 5 µm absorbing layer
absorption_factor = 0.5

# Storage for animation
Ez_list = []

# FDTD loop
for t in range(Nt):
    # Update magnetic field
    Hx[:, :-1] -= C1[:, :-1] * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] += C1[:-1, :] * (Ez[1:, :] - Ez[:-1, :])
    
    # Update electric field
    Ez[1:, 1:] += C2[1:, 1:] * ((Hy[1:, 1:] - Hy[:-1, 1:]) - (Hx[1:, 1:] - Hx[1:, :-1]))
    
    # Apply source
    Ez[source_pos] += gaussian_pulse(t)
    
    # Apply absorption on the left region
    Ez[:absorption_length, :] *= absorption_factor
    
    # Store field for animation
    Ez_list.append(Ez.copy())

# Animation setup
fig, ax = plt.subplots()
cax = ax.imshow(Ez_list[0], extent=[0, Nx*dx*1e6, 0, Ny*dy*1e6], cmap='bwr', vmin=-0.1, vmax=0.11)
ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")
ax.set_title("2D FDTD Simulation: Electric Field Ez")
fig.colorbar(cax)

def update(frame):
    cax.set_data(Ez_list[frame])
    return cax,

ani = animation.FuncAnimation(fig, update, frames=Nt, interval=2, blit=True)
plt.show()

