import numpy as np
import matplotlib.pyplot as plt

def linear_convection(nx, L=2., c=1., sigma=0.5, nt=20):
    dx = L / (nx -1)
    x = np.linspace(0., L, num=nx)
    dt = sigma * dx / c
    u0 = np.ones(nx)
    mask = np.where(np.logical_and(x >= 0.5, x <= 1.0))
    u0[mask] = 2.0
    u = u0.copy()
    for n in range(1, nt):
        u[1:] = u[1:] - c * dt / dx * (u[1:] - u[:-1])
    plt.title('Initial Conditions')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid
    plt.plot(x, u0, label='Initial', color='C0', linestyle='--')
    plt.plot(x, u, label='nt = {}'.format(nt), color='C2', linestyle=':')
    plt.legend()
    plt.xlim(0., L)
    plt.ylim(0., 2.5)
    plt.show()


linear_convection(300)
