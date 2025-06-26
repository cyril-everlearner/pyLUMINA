import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 512  # grid size
iterations = 60
wavelength = 1.0  # normalized
beam_waist = 100  # pixels

# Coordinate system
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)
R2 = X**2 + Y**2

# Input Gaussian beam
input_amplitude = np.exp(-R2 * (N / (2 * beam_waist))**2)

# Target: Top-hat beam in focal plane
def generate_tophat_target(radius=0.1):
    mask = np.sqrt(X**2 + Y**2) <= radius
    return mask.astype(float)

target_intensity = generate_tophat_target(radius=0.2)
target_amplitude = np.sqrt(target_intensity)

# Initial input phase (random)
#input_phase = 0*np.exp(1j * 2 * np.pi * np.random.rand(N, N))
input_phase = np.exp(1j * 5 * np.pi * (X**2 + Y**2))
input_phase1=input_phase
# Simulate optical aberration: Astigmatism
def add_astigmatism(phase, amount=0.5):
    return phase * np.exp(1j * amount * (X**2 - Y**2))

# IFTA with aberration added at each iteration
for i in range(iterations):
    # Forward FFT to focal plane
    U_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(input_amplitude * input_phase)))
    
    # Replace amplitude with target, keep phase
    U_focal_phase = np.angle(U_focal)
    U_focal_new = target_amplitude * np.exp(1j * U_focal_phase)

    # Backward FFT to input plane
    U_input = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_focal_new)))

    # Update phase and apply astigmatism
    input_phase = np.exp(1j * np.angle(U_input))
    #input_phase = add_astigmatism(input_phase, amount=0.5)

# Final field in Fourier plane
U_final = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(input_amplitude * input_phase)))
I_final = np.abs(U_final)**2

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Target Intensity (Top-hat)")
plt.imshow(target_intensity, cmap='inferno')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Input Phase with Astigmatism")
plt.imshow(np.angle(input_phase1), cmap='twilight')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Simulated Output Intensity")
plt.imshow(I_final, cmap='inferno')
plt.colorbar()

plt.tight_layout()
plt.show()

