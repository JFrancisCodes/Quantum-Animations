import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-20, 20)  # Adjust the x-axis limits to the desired window

x_min = -300.0  # Minimum position
x_max = 300.0  # Maximum position
num_points = 10000  # Number of points in the grid
x = np.linspace(x_min, x_max, num_points)

dt = 0.01  # Time step size
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


def box(x, L, center):
    ans = np.zeros(len(x))
    dx = x[1] - x[0]
    start_ind = int(((center - L / 2.0) - x[0]) / dx)
    end_ind = int(((center + L / 2.0) - x[0]) / dx)
    ans[start_ind: end_ind + 1] = 1.0 / np.sqrt(L)
    return ans

def simple_quadratic(x,c,L):
    y = np.where(np.logical_and(x >= -L, x <= L), (3/4) * (1 - (x-c)**2), 0)
    return y

def simple_line(x,L):
    n = (L^3)/3
    y = (1/n)*np.where(np.logical_and(x >= -L/2, x <= L/2), (x + L/2), 0)
    return y

def other_line(x,L):
    n = (L^2)/2
    y = (1/n)*np.where(np.logical_and(x >= -L/2, x <= L/2), np.sqrt(np.maximum(x + L/2, 0)), 0)
    return y

def sinebump(x,L,n):
    if n % 2 == 0:
        y = np.where(np.logical_and(x >= -L*np.pi/2, x <= L*np.pi/2), np.sin(n*x/L), 0)
    else:
        y = np.where(np.logical_and(x >= -L*np.pi/2, x <= L*np.pi/2), np.cos(n*x/L), 0)
    return y



def solve_schrodinger_eq(psi_initial, x, t_values, dt):
    psi = np.copy(psi_initial)
    psi_transformed = fourier_transform(psi, dx)

    psi_evolution = []  # Copy the initial wave function

    for t in t_values:
        # Fourier domain evolution (kinetic term)
        psi_transformed *= np.exp(-1j * k_values ** 2 * t / 2)

        # Inverse Fourier transform
        psi = inverse_fourier_transform(psi_transformed, dk)

        # Normalize the wave function
        psi /= L2_norm(psi, dx)

        # Append the current wave function to the list
        psi_evolution.append(psi.copy())  # Copy the current wave function

        # Fourier transform
        psi_transformed = fourier_transform(psi, dx)

    return np.array(psi_evolution)


psi = sinebump(x,1,4)


t_values = np.linspace(0, num_steps_time * dt, num_steps_time)
psi_history = solve_schrodinger_eq(np.copy(psi), x, t_values, dt)


y_max = np.max(np.abs(psi_history) ** 2)
line, = ax.plot(x, np.abs(psi_history[0]) ** 2)
ax.set_ylim(0, y_max) 

# Function to update the plot for each animation frame
def update(frame):
    line.set_data(x, np.abs(psi_history[frame]) ** 2)
    ax.set_title('Time (t={:.2f})'.format(t_values[frame]))

    return line,


# Create the animation with a specific frame rate
frame_rate = 30  # Frame rate in frames per second
animation = FuncAnimation(fig, update, frames=num_steps_time, interval=1000/frame_rate, blit=True)

# Save the animation as an MP4 file using ffmpeg
animation.save('box.mp4', writer='ffmpeg', dpi=100)


# Display the animation
plt.show()
