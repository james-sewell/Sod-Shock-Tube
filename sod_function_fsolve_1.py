import numpy as np
import scipy.optimize as opt


# state properties are p, rho, u, gamma
left_state = (1, 1, 0, 1.4)
right_state = (0.1, 0.125, 0., 1.4)
geometry = (-0.5, 0.5, 0)
t = 0.2
npts = 500
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
    return p5_in * (1 - ((k1 * prm1) / np.sqrt(1 + k2 * prm1)))**k3 - p2_in


p1 = right_state[0]
rho1 = right_state[1]
gamma_1 = right_state[3]
p5 = left_state[0]
rho5 = left_state[1]
gamma_5 = left_state[3]
p2 = opt.fsolve(shock_tube_function, x0=np.linspace(p1, p5, max_iter), args=(p1, p5, rho1, rho5, gamma_1, gamma_5),
                xtol=1e-6)

print(p2)
np.savetxt('sod_fun_fsolve.txt', (p2[0], p1))