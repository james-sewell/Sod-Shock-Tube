import numpy as np


def vector_u(array, gamma):
    rho = array[0]
    u = array[1]
    p = array[2]
    e_t = p / ((gamma - 1) * rho) + .5 * u ** 2
    return np.array([rho, rho * u, rho * e_t])


def vector_f(u, gamma):
    u_1 = u[0]
    u_2 = u[1]
    u_3 = u[2]

    return np.array([u_2, u_2 ** 2 / u_1 + (gamma - 1) * (u_3 - .5 * u_2 ** 2 / u_1),
                        (u_3 + (gamma - 1) * (u_3 - .5 * u_2 ** 2 / u_1)) * u_2 / u_1])


def do_the_math(array, nt, nx, dt, dx, gamma):
    u = np.zeros((nt, 3, nx))
    u[0] = vector_u(array, gamma)

    for t in range(nt - 1):
        un = u[t]
        f = vector_f(un, gamma)
        u_star = 0.5 * (un[:, 1:] + un[:, :-1]) - dt / (2 * dx) * (f[:, 1:] - f[:, :-1])
        f_star = vector_f(u_star, gamma)
        u[t + 1][:, 1:-1] = un[:, 1:-1] - dt / dx * (f_star[:, 1:] - f_star[:, :-1])
        u[t + 1][:, 0] = un[:, 0]
        u[t + 1][:, -1] = un[:, -1]

    return u, nt, nx, gamma


nx = 81
dx = 0.25
dt = 0.0002
gamma = 1.4

T = 0.2             # time considered for the simulation
nt = int(T/dt)+1     # number of time steps
rho_left = 1.        # density in kg m^{-3}
u_left = 0           # velocity in m/s
p_left = 100      # pressure in kg m^{-2}
rho_right = 0.125    # density in kg m^{-3}
u_right = 0          # velocity in m/s
p_right = 10      # pressure in kg m^{-2}

# Boundaries list used with the function shocktube()
condition_left = [rho_left,u_left,p_left]
condition_right = [rho_right,u_right,p_right]

