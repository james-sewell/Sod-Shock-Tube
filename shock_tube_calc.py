import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# state properties are p, rho, u, gamma
red_side = (100e3, 1, 0, 273, 1.4)
blue_side = (10e3, 0.125, 0., 273, 1.4)
geometry = (-150, 150, 0)
t = 0.2
num_points = 500
max_iter = 10


def sound_speed(gamma_in, p, rho):
    """
    CHECKED
    :param gamma_in:
    :param p:
    :param rho:
    :return:
    """
    return np.sqrt(gamma_in * p / rho)


def shock_tube_function(p2, p1, p5, rho1, rho5, gamma_1, gamma_5):
    """
    CHECKED
    :param p2:
    :param p1:
    :param p5:
    :param rho1:
    :param rho5:
    :param gamma_1:
    :param gamma_5:
    :return:
    """
    a1 = sound_speed(gamma_1, p1, rho1)
    a5 = sound_speed(gamma_5, p5, rho5)
    prm1 = (p2 / p1) - 1
    g1p1 = gamma_1 + 1.
    g5m1 = gamma_5 - 1.
    g5_2 = 2. * gamma_5
    g1_2 = 2. * gamma_1
    k1 = g5m1 / g1_2 * (a1 / a5)
    k2 = g1p1 / g1_2
    k3 = g5_2 / g5m1
    return p5 * (1 - (k1 * prm1 / np.sqrt(1 + k2 * prm1)))**k3 - p2


p1 = blue_side[0]
rho1 = blue_side[1]
u1 = blue_side[2]
T1 = blue_side[3]
gamma_1 = blue_side[4]
p5 = red_side[0]
rho5 = red_side[1]
u5 = red_side[2]
T5 = red_side[3]
gamma_5 = red_side[4]

result = opt.fsolve(shock_tube_function, x0=np.linspace(p1, p5, max_iter), args=(p1, p5, rho1, rho5, gamma_1, gamma_5),
                xtol=1e-6)
p2 = result[0]

# calculate region 1
a1 = sound_speed(gamma_1, p1, rho1)
g1p1 = gamma_1 + 1.
g1m1 = gamma_1 - 1.
g1_2 = 2*gamma_1
p2p1 = p2 / p1
k4 = g1p1 / g1_2
k5 = g1m1 / g1_2
s = a1 * np.sqrt(k4 * p2p1 + k5)

# calculate region 2
k6 = g1p1 / g1m1
k7 = k6 * p2p1 + 1
k8 = k6 + p2p1
rho2 = rho1 * k7 / k8
k9 = (p2 / p1) - 1
u2 = a1 * k9 * np.sqrt((2 / gamma_1) / (g1p1 * p2p1 + g1m1))
k10 = g1m1 / g1p1
k11 = 1 + k10 * p2p1
k12 = 1 + k10 * p1 / p2
T2 = T1 * (k11 / k12)

# calculate region 3
p3 = p2
k13 = 1 / gamma_5
rho3 = rho5 * (p3 / p5)**k13
a3 = sound_speed(gamma_5, p3, rho3)
u3 = u2
p5p1 = p5 / p1
g5m1 = gamma_5 - 1.
k14 = g5m1 / gamma_5
T3 = T5 * (p2p1 / p5p1)**k14

# calculate region 5
a5 = sound_speed(gamma_5, p5, rho5)
# u5 = a5
g5p1 = gamma_5 + 1.

regions = np.array([[rho2, u2, p2, T2], [rho3, u3, p3, T3]])

x_initial = geometry[2]
x_total = np.abs(geometry[0]) + geometry[1]
x_shock = x_initial + s * t
x_contact_surf = x_initial + u2 * t
x_rare_ft = x_initial + (u3 - a3) * t
x_rare_hd = x_initial - a5 * t


# create arrays for plots
x_array = np.linspace(geometry[0], geometry[1], num_points)
rho = np.zeros(num_points)
p = np.zeros(num_points)
rho = np.zeros(num_points)
u = np.zeros(num_points)
T = np.zeros(num_points)

for i in range(0, num_points):
    if x_array[i] < x_rare_hd:
        p[i] = p5
        rho[i] = rho5
        u[i] = u5
        T[i] = T5
    elif x_array[i] < x_rare_ft:
        u[i] = 2. / g5p1 * (a5 + (x_array[i] - x_initial) / t)
        k13 = 1. - 0.5 * g5m1 * u[i] / a5
        rho[i] = rho5 * k13 ** (2. / g5m1)
        p[i] = p5 * k13 ** (2. * gamma_5 / g5m1)
        T[i] = T5 * (p[i] / rho[i]/1e5)
    elif x_array[i] < x_contact_surf:
        p[i] = p3
        rho[i] = rho3
        u[i] = u3
        T[i] = T3
    elif x_array[i] < x_shock:
        p[i] = p2
        rho[i] = rho2
        u[i] = u2
        T[i] = T2
    else:
        p[i] = p1
        rho[i] = rho1
        u[i] = u1
        T[i] = T1

# plot values

plt.figure(0)
plt.title('Shock Tube Fluid Pressure at t=0.2s')
plt.plot(x_array, p)
plt.xlabel('Position')
plt.ylabel('Pressure (bar)')
plt.savefig('shock_tube_p', bbox_inches='tight')

plt.figure(1)
plt.title('Shock Tube Fluid Density at t=0.2')
plt.plot(x_array, rho)
plt.xlabel('Position')
plt.ylabel('Density (kg/m^3)')
plt.savefig('shock_tube_rho', bbox_inches = 'tight')

plt.figure(2)
plt.title('Shock Tube Fluid Velocity at t=0.2s')
plt.plot(x_array, u)
plt.xlabel('Position')
plt.ylabel('Velocity (m/s)')
plt.savefig('shock_tube_u', bbox_inches = 'tight')

plt.figure(3)
plt.title('Shock Tube Fluid Temperature at t=0.2s')
plt.plot(x_array, T)
plt.xlabel('Position')
plt.ylabel('Temperature (K)')
plt.savefig('shock_tube_T', bbox_inches = 'tight')

# plt.show()

print(regions[1][1])
