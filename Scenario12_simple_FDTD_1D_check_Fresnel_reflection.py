'''
Looking at simple FDTD 1D


This script simulates the propagation of a 10 fs laser pulse at 1030 nm using the 1D FDTD method. It includes the vacuum-glass interface and allows for observation of reflection and transmission. The animation shows the temporal evolution of the electric field.

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
from matplotlib.widgets import Button

# Physical constants
c = 3e8  # Speed of light in vacuum (m/s)
dx = 50e-9  # Spatial step (50 nm)
dt = dx / (2 * c)  # Time step to satisfy the Courant condition

# Simulation parameters
Nx = 800  # Number of grid points
Nt = 3000  # Number of time steps

# Definition of electric and magnetic fields
Ez = np.zeros(Nx)  # Electric field
Hy = np.zeros(Nx)  # Magnetic field

# Pulse properties
source_pos = 100
lambda0 = 1030e-9  # Central wavelength (m)
T0 = 10e-15  # Pulse duration (s)
omega0 = 2 * np.pi * c / lambda0  # Angular frequency

# Gaussian pulse function
T_peak = 200  # Offset to avoid cut-off
gaussian_pulse = lambda t: np.exp(-((t - T_peak) * dt / T0) ** 2) * np.sin(omega0 * (t - T_peak) * dt)

# FDTD coefficients
C1 = np.ones(Nx) * (c * dt / dx)
C2 = np.ones(Nx) * (c * dt / dx)

# Absorption layer on the left
absorption_region_length = int(5e-6 / dx)  # 5 µm absorbing region
absorption_factor = 0.5  # Absorption factor

# Fused silica parameters
epsilon_r_silica = 2.1  # Relative permittivity
mu_r_silica = 1.0  # Relative permeability

# Apply fused silica parameters to the right half of the grid
silica_start = Nx // 2
for i in range(silica_start, Nx):
    C1[i] = (c * dt / dx) / np.sqrt(epsilon_r_silica * mu_r_silica)
    C2[i] = (c * dt / dx) / np.sqrt(epsilon_r_silica * mu_r_silica)

# Store fields for animation
Ez_list = []
Hy_list = []

# FDTD time loop
for t in range(Nt):
    for i in range(Nx - 1):
        Hy[i] += C1[i] * (Ez[i + 1] - Ez[i])
    
    for i in range(1, Nx):
        Ez[i] += C2[i] * (Hy[i] - Hy[i - 1])
    
    Ez[source_pos] += gaussian_pulse(t)
    
    for i in range(absorption_region_length):
        Ez[i] *= absorption_factor
    
    Ez_list.append(Ez.copy())
    Hy_list.append(Hy.copy())

# Plot setup
fig, ax = plt.subplots()
ax.set_xlim(0, Nx * dx * 1e6)
ax.set_ylim(-1, 1)
ax.set_xlabel("Position (µm)")
ax.grid(True)

# Set horizontal ticks every 1 µm
tick_positions = np.arange(0, Nx * dx * 1e6, 1)  # Step of 1 µm
ax.set_xticks(tick_positions)
ax.set_xticklabels([f"{x:.1f}" for x in tick_positions])

# Add time label in ns
time_text = ax.text(0.02, 0.95, f'Time: 0.000 ns', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Create plot lines
line_ez, = ax.plot([], [], lw=2, label='Ez (Electric Field)')
line_hy, = ax.plot([], [], lw=2, label='Hy (Magnetic Field)', color='red')

# Add fused silica interface line
silica_line = ax.axvline(x=silica_start * dx * 1e6, color='green', linestyle='--', label="Fused Silica Interface")

# Pause button
paused = False

def toggle_pause(event):
    global paused
    paused = not paused
    button.label.set_text("Resume" if paused else "Pause")

# Create pause button
ax_button = plt.axes([0.85, 0.02, 0.1, 0.05])
button = Button(ax_button, 'Pause', color='lightgoldenrodyellow')
button.on_clicked(toggle_pause)


# Function to update the plot for animation
#def update(frame):
#    if not paused:
#        # Update the data for the electric and magnetic fields
#        line_ez.set_data(np.linspace(0, Nx * dx * 1e6, Nx), Ez_list[frame])  # Electric field
#        line_hy.set_data(np.linspace(0, Nx * dx * 1e6, Nx), Hy_list[frame])  # Magnetic field
#        time_text.set_text(f'Time step: {frame}')  # Update the text for the time step
#    return line_ez, line_hy, time_text

# Update function
def update(frame):
    if not paused:
        line_ez.set_data(np.linspace(0, Nx * dx * 1e6, Nx), Ez_list[frame])
        line_hy.set_data(np.linspace(0, Nx * dx * 1e6, Nx), Hy_list[frame])

        # Update time in ns
        time_ns = frame * dt * 1e15  # Convert time to fs
        time_text.set_text(f'Time: {time_ns:.3f} fs')

    return line_ez, line_hy, time_text

# Animation (blit=False to ensure text updates correctly)
ani = animation.FuncAnimation(fig, update, frames=Nt, interval=1, blit=True)

# Legend
ax.legend(loc='upper right')

plt.show()

