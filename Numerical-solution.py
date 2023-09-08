import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_min = -50.0  # Minimum position
x_max = 50.0  # Maximum position
num_points = 100000  # Number of points in the grid
x = np.linspace(x_min, x_max, num_points)

dt = 0.1  # Time step size
dx = (x_max - x_min) / num_points
num_steps_time = 1000

k_values = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx))

dk = 2 * np.pi / (x_max - x_min)

def fourier_transform(psi, dx):
    ft = np.fft.fftshift(np.fft.fft(psi)) * dx
    return ft

def inverse_fourier_transform(ft, dk):
    psi = np.fft.ifft(np.fft.ifftshift(ft)) * dk
    return psi

def L2_norm(psi, dx):
    norm = np.sqrt(np.trapz(np.abs(psi) ** 2, dx=dx))
    return norm
def gaussian_wave(x, x0, sigma, p):
    psi = np.exp(-0.5 * ((x - x0) / sigma) ** 2)*np.exp(p*x*1j)
    return psi


def box(x, L):
    ans = np.zeros(len(x))
    dx = x[1] - x[0]
    start_ind = int(((-L / 2.0) - x[0]) / dx)
    end_ind = int(((L / 2.0) - x[0]) / dx)
    ans[start_ind: end_ind + 1] = 1.0 / np.sqrt(L)
    return ans


def solve_schrodinger_eq(psi_initial, x, t_values, dt):
    psi = np.copy(psi_initial)
    psi_transformed = fourier_transform(psi, dx)

    psi_evolution = []  # Copy the initial wave function
    
    for t in t_values:
        # Fourier domain evolution (kinetic term)
        psi_transformed *= np.exp(-1j * k_values**2 * t / 2)

        # Inverse Fourier transform
        psi = inverse_fourier_transform(psi_transformed, dk)

        # Normalize the wave function
        psi /= L2_norm(psi, dx)

        # Append the current wave function to the list
        psi_evolution.append(psi.copy())  # Copy the current wave function

        # Fourier transform
        psi_transformed = fourier_transform(psi, dx)

    return np.array(psi_evolution)

psi = box(x,10)

t_values = np.linspace(0, num_steps_time * dt, num_steps_time)
psi_history = solve_schrodinger_eq(np.copy(psi), x, t_values, dt)


fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi_history[0]) ** 2)

# Function to update the plot for each animation frame
def update(frame):
    line.set_ydata(np.abs(psi_history[frame]) ** 2)

# Create the animation
animation = FuncAnimation(fig, update, frames=num_steps_time)

# Display the animation
plt.show()
