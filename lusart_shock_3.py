import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def vector_u(array, gamma_in):
    rho = array[0]
    u = array[1]
    p = array[2]
    m = rho * u
    ep = p / (gamma_in - 1) / rho
    e = rho * ep + 0.5 * rho * u * u
    return np.array([rho, m, e])


def vector_f(u, gamma_in):
    u_1 = u[0]
    u_2 = u[1]
    u_3 = u[2]
    return np.array([u_2, u_2 ** 2 / u_1 + (gamma_in - 1) * (u_3 - .5 * u_2 ** 2 / u_1),
                        (u_3 + (gamma_in - 1) * (u_3 - .5 * u_2 ** 2 / u_1)) * u_2 / u_1])


def vector_a(u, nt, nx, gamma):
    initial = np.zeros((nt, 3, nx))
    for t in range(nt):
        u_1 = u[t][0]
        u_2 = u[t][1]
        u_3 = u[t][2]
        initial[t] = np.array([u_1, u_2 / u_1, (gamma - 1) * (u_3 - .5 * u_2 ** 2 / u_1)])
    return initial


def region_fill_initial(nx, cdt_left, cdt_right):
    x = np.ones(nx)

    middle = int((nx - 1) / 2)

    rho = x.copy() * cdt_left[0]
    rho[middle:] = cdt_right[0]

    u = x.copy() * cdt_left[1]
    u[middle:] = cdt_right[1]

    p = x.copy() * cdt_left[2]
    p[middle:] = cdt_right[2]

    T = x.copy() * cdt_left[3]
    T[middle:] = cdt_right[3]

    gamma = x.copy() * cdt_left[3]
    gamma[middle:] = cdt_right[3]

    array = np.array([rho, u, p, T, gamma])

    return array


def sound_speed(gamma_in, p, rho):
    """
    CHECKED
    :param gamma_in:
    :param p:
    :param rho:
    :return:
    """
    return np.sqrt(gamma_in * p / rho)


def godunov(array, nt, nx, dt, dx, gamma):
    u = np.zeros((nt, 3, nx))
    u[0] = vector_u(array, gamma)

    for t in range(nt - 1):
        un = u[t]
        f = vector_f(un, gamma)
        u_star = 0.5 * (un[:, 1:] + un[:, :-1]) - dt / dx * (f[:, 1:] - f[:, :-1])
        f_star = vector_f(u_star, gamma)
        u[t + 1][:, 1:-1] = un[:, 1:-1] - dt / dx * (f_star[:, 1:] - f_star[:, :-1])
        u[t + 1][:, 0] = un[:, 0]
        u[t + 1][:, -1] = un[:, -1]

    return vector_a(u, nt, nx, gamma)


def lax_wendroff(array, nt, nx, dt, dx, gamma):
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

    return vector_a(u, nt, nx, gamma)


def maccormack(array, nt, nx, dt, dx, gamma):
    u = np.zeros((nt, 3, nx))
    u[0] = vector_u(array, gamma)

    for t in range(nt - 1):
        un = u[t]
        f = vector_f(un, gamma)
        u_star = un[:, 1:-1] - dt / dx * (f[:, 1:] - f[:, :-1])
        f_star = vector_f(u_star, gamma)
        u[t + 1][:, 1:-1] = 0.5 * (un[:, 1:] + un[:, :-1]) - dt / (2 * dx) * (f_star[:, 1:] - f_star[:, :-1])
        u[t + 1][:, 0] = un[:, 0]
        u[t + 1][:, -1] = un[:, -1]

    return vector_a(u, nt, nx, gamma)


def temp_calc(array, nt, gamma, R):
    u_1 = array[nt-1][0]
    u_2 = array[nt-1][1]
    u_3 = array[nt-1][2]
    return np.array(gamma - 1) * (u_3 - .5 * u_2 ** 2 / u_1) / u_1 / R


#Basic code parameters
nx = 500
dx = 0.6
gamma = 1.4
T = 0.2
nt_lax = 250
nt_god = 500
dt_lax = T / nt_lax
dt_god = T / nt_god

condition_left = [1., 0., 100.e3, 300., 1.4]
condition_right = [0.125, 0., 10.e3, 273., 1.67]

A = region_fill_initial(nx, condition_left, condition_right)

x = np.linspace(-150, 150, nx)

lax = lax_wendroff(A, nt_lax, nx, dt_lax, dx, gamma)

T_lax = temp_calc(lax, nt_lax, gamma, 287.)

god = godunov(A, nt_god, nx, dt_god, dx, gamma)

# a_ = maccormack(A, nt, nx, dt, dx, gamma)

plt.figure(0)
plt.plot(x, lax[nt_lax-1][0])
plt.ylabel('Density')
plt.xlabel('Distance')

plt.figure(1)
plt.plot(x, god[nt_god-1][0])
plt.ylabel('Density')
plt.xlabel('Distance')

plt.figure(2)
plt.plot(x, T_lax)
plt.ylabel('Density')
plt.xlabel('Distance')

# plt.figure(0)
# plt.plot(x, a[nt-1][0])
# plt.ylabel('Density')
# plt.xlabel('Distance')

# plt.figure(1)
# plt.plot(x, a[nt-1][1])
# plt.ylabel('Velocity')
# plt.xlabel('Distance')
#
# plt.figure(2)
# plt.plot(x, a[nt-1][2], 'x')
# plt.ylabel('Pressure')
# plt.xlabel('Distance')
# plt.savefig('shock_tube_lusart_p', bbox_inches='tight')

plt.show()
