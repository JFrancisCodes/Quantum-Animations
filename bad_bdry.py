import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_min, x_max = -2, 2
num_points = 1000
x = np.linspace(x_min, x_max, num_points)
dx = (x_max-x_min)/num_points
zero = 0*x
length = x_max- x_min

t_min, t_max = 0, 1.5
num_time = 500
dt = (t_max - t_min)/num_time
t = t_min 
counter = 0

fig, ax = plt.subplots()
ax.set_xlim(x_min, x_max)


line, = ax.plot(x, zero, lw=2)


def non_phys_exp(x,t):
    return np.exp(t-x)

def non_phys_exp_2(x,t):
    return np.exp(x-t)

def phys_exp(x,t):
    a = ((x - t) - x_min) % length + x_min 
    return np.exp(-np.abs(a))

def L2_norm(psi, dx):
    norm = np.sqrt(np.trapz(np.abs(psi) ** 2, dx=dx))
    return norm

ax.set_ylim(0, non_phys_exp_2(x_max,t_min)) 

def update(frame):
    global t, counter
    if t>t_max:
        t= t_min
    psi = non_phys_exp(x,t)
    line.set_data(x, psi)
    t += dt
    if counter % 10 == 0:
        print("L2 Norm:", L2_norm(psi, dx))
    counter = counter + 1
              
    return line,

ani = FuncAnimation(fig, update, frames=num_time, interval=10, blit=True)

ani.save('bad_bdry1.mp4', writer='ffmpeg', dpi=100)

plt.show()