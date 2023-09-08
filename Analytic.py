import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

t0 = 0.0  # Initial time


x0_1 = -10.0  # Mean position
sigma_1 = 1 # Standard deviation 
p_1 = -5 # Momentum 
theta_1 = 0 # Phase
w_1 = 0.6 # Weight of first gaussian

x0_2 = 10.0  # Mean position of the second Gaussian
sigma_2 = 1 # Standard deviation of the second Gaussian
p_2 = 10  # Momentum of the second Gaussian
theta_2 = 0 # Phase of the second Gaussian
w_2 = 1 - w_1 # Weight of second gaussian

x_min = -40.0  # Minimum position
x_max = 40.0  # Maximum position
num_points = 100000  # Number of points in the grid
x = np.linspace(x_min, x_max, num_points)

num_steps_time = 300  # Number of time steps
dt = 0.01  # Time step size
dx = 0.01 

k_values = np.fft.fftshift(np.fft.fftfreq(len(x), d=x[1]-x[0]))

def L2_norm(psi, dx):
    norm = np.sqrt(np.trapz(np.abs(psi) ** 2, dx=dx))
    return norm

def psi_analytic(x, t, x0, p, sigma):
    normalization = 1 / np.sqrt(2 * np.pi)
    denominator = np.sqrt(sigma**2 + 1j * t)
    exponent = (-0.5 * (x - x0 + 1j * p * sigma**2)**2) / (sigma**2 + 1j * t)
    phase_factor = x0 * p
    gaussian_factor = np.exp(-0.5 * sigma**2 * p**2)
    
    return normalization * (1 / denominator) * np.exp2(exponent) * np.exp(1j * phase_factor) * gaussian_factor

# Calculate the wave function for each Gaussian separately
psi_1 = psi_analytic(x, 0, x0_1, p_1, sigma_1) * np.exp(1j * theta_1)
psi_2 = psi_analytic(x, 0, x0_2, p_2, sigma_2) * np.exp(1j * theta_2)

# Add the wave functions together
psi_unnorm = w_1*psi_1 + w_2*psi_2

N = L2_norm(psi_unnorm,dx)

psi = psi_unnorm/N

fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi) ** 2)
ax.set_xlabel('Position')
ax.set_ylabel('Probability Density')
ax.set_title('Gaussian Wave Function')
ax.grid(True)

frames = []

def update(frame):
    t = (frame + 1) * dt
    psi_a1 = psi_analytic(x, t, x0_1, p_1, sigma_1) * np.exp(1j * theta_1)
    psi_a2 = psi_analytic(x, t, x0_2, p_2, sigma_2) * np.exp(1j * theta_2)

    psi_a = (1/N) * (w_1*psi_a1 + w_2*psi_a2)  # Add the wave functions together

    line.set_data(x, np.abs(psi_a) ** 2)

    if frame % 100 == 0:
        # Calculate and print the L2 norm at every 100th frame
        norm = L2_norm(psi_a, dx)
        print("L2 Norm at frame", frame + 1, ":", norm)

    return line,  # Add a comma after line to return it as a sequence


# Define the total duration of the animation in seconds
total_duration = 30.0

# Define the desired animation speed by adjusting the interval value
animation_speed = 0.5  # Adjust this value to control the animation speed

interval = 1000 / 24 

# Calculate the number of frames needed for the desired duration
num_frames = int(total_duration * 1000 / interval)

# Adjust the number of frames to fit within the available data
num_frames = min(num_frames, num_steps_time)

# Create an array of frame indices for the animation
frames = np.linspace(0, num_steps_time - 1, num_frames, dtype=int)

# Create the animation
animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

# Save the animation as an MP4 file
animation.save('wave_animation.mp4', writer='ffmpeg', dpi=100)




plt.close(fig)