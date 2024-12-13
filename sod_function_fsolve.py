import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


# state properties are p, rho, u, gamma, T
left_state = (1, 1, 0, 1.4, 273)
right_state = (0.1, 0.125, 0., 1.4, 273)
geometry = (-1, 1, 0)
t = 0.2
num_points = 500
max_iter = 100


def sound_speed(gamma_in, p, rho):
    """
    CHECKED
    :param gamma_in:
    :param p:
    :param rho:
    :return:
    """
    return np.sqrt(gamma_in * p / rho)


def shock_tube_function(p2_in, p1_in, p5_in, rho1_in, rho5_in, gamma_1_in, gamma_5_in):
    a1 = sound_speed(gamma_1_in, p1_in, rho1_in)
    a5 = sound_speed(gamma_5_in, p5_in, rho5_in)
    prm1 = (p2_in / p1_in) - 1
    g1p1 = gamma_1 + 1.
    g5m1 = gamma_5 - 1.
    g5_2 = 2. * gamma_5_in
    g1_2 = 2. * gamma_1_in
    k1 = g5m1 / g1_2 * (a1 / a5)
    k2 = g1p1 / g1_2
    k3 = g5_2 / g5m1
    return p5_in * (1 - (k1 * prm1 / np.sqrt(1 + k2 * prm1)))**k3 - p2_in


p1 = right_state[0]
rho1 = right_state[1]
gamma_1 = right_state[3]
p5 = left_state[0]
rho5 = left_state[1]
gamma_5 = left_state[3]
p2 = opt.fsolve(shock_tube_function, x0=np.linspace(p1, p5, max_iter), args=(p1, p5, rho1, rho5, gamma_1, gamma_5),
                xtol=1e-6)
x_initial = geometry[2]
xl = geometry[0]
xr = geometry[1]
x_array = np.linspace(xl, xr, num_points)

x_shock = 0.
x_cont_surf = 0.5
p = np.zeros(num_points)
u = np.zeros(num_points)

for i in range(0, num_points):
    if x_array[i] < x_shock:
        p[i] = p2[0]
        u[i] = 2. * ((x_array[i] - x_initial) / t)
        k13 = 1. - 0.5 * u[i] / 5
    elif x_array[i] < x_cont_surf:
        p[i] = 0.5
    else:
        p[i] = 1.


print(p2)
print(u)
np.savetxt('sod_fun_fsolve.txt', np.c_[x_array, p])

plt.plot(x_array, p)
plt.show()