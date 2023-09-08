import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_min, x_max = -2, 2
num_points = 10000
x = np.linspace(x_min, x_max, num_points)
dx = (x_max-x_min)/num_points
zero = 0*x
counter = 0

t_min, t_max = 0, 0.001
num_time = 1000
dt = (t_max - t_min)/num_time
t = t_min 
t_values = np.linspace(t_min,t_max, num_time)

k_values = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx))
dk = 2 * np.pi / (x_max - x_min)

terms_to_include = 300

fig, ax = plt.subplots()
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, 1) 

line, = ax.plot(x, zero, lw=2)


def non_phys_exp_2(x):
    return np.exp (-np.abs(x))

def non_phys_exp(x,t):
    return np.exp(t-x)

def fourier_transform(psi, dx):
    ft = np.fft.fftshift(np.fft.fft(psi)) * dx
    return ft

def inverse_fourier_transform(ft, dk):
    psi = np.fft.ifft(np.fft.ifftshift(ft)) * dk
    return psi

def L2_norm(psi, dx):
    norm = np.sqrt(np.trapz(np.abs(psi) ** 2, dx=dx))
    return norm

psi = non_phys_exp_2(x)

def coefficients(psi,terms_to_include):
    fft_result = np.fft.fft(psi)
    a0 = np.real(fft_result[0]) / num_points
    cn = fft_result[1:] / num_points
    coefficients = np.zeros(terms_to_include, dtype= complex)
    coefficients[0] = a0 / 2  # Store the DC (constant) coefficient
    for n in range(1, terms_to_include):
        coefficients[n] = cn[n - 1]
    return coefficients

def evo_coeff(coeff, t):
    evo_coeff = np.zeros(terms_to_include, dtype= complex)
    for n in range(0,len(coeff)):
        evo_coeff[n] = np.exp(-1j*n*t)*coeff[n]
    
    return evo_coeff


def evolved_function(psi,t):
    coeff = coefficients(psi,terms_to_include)
    evo = evo_coeff(coeff,t)
    psit = evo[0]
    for i in range(1,len(coeff)):
        psit += evo[i]*np.exp(1j*i*x)
    return psit
    
def update(t):
    global counter
    line.set_ydata(2*evolved_function(psi, t))
    if counter % 10 == 0:
        print("L2 Norm:", L2_norm(psi, dx))
    counter = counter + 1
    return line,

ani = FuncAnimation(fig, update, frames=num_time, interval=100, blit=True)
ani.save('good_bdry2.mp4', writer='ffmpeg', dpi=100)

plt.show()