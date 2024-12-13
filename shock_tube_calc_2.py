import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# state properties are p, rho, u, gamma
rho_left = 5.75
u_left = 0
p_left = 500e3
T_left = 303
gamma_left = 1.4
rho_right = 0.23
u_right = 0
p_right = 20e3
T_right = 303
gamma_right = 1.4

region_left = [rho_left, u_left, p_left, T_left, gamma_left]
region_right = [rho_right, u_right, p_right, T_right, gamma_right]

geometry = (-150, 150, 0, 500)
t = 0.2
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


def exact_sol_pr(p2, p1, p5, rho1, rho5, gamma_1, gamma_5):
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


def exact_region_props(left, right):
    rho5 = left[0]
    u5 = left[1]
    p5 = left[2]
    T5 = left[3]
    gamma_5 = left[4]
    rho1 = right[0]
    u1 = right[1]
    p1 = right[2]
    T1 = right[3]
    gamma_1 = right[4]

    result = opt.fsolve(exact_sol_pr, x0=np.linspace(p1, p5, max_iter), args=(p1, p5, rho1, rho5, gamma_1, gamma_5),
                        xtol=1e-6)
    p2 = result[0]

    # calculate region 1
    a1 = sound_speed(gamma_1, p1, rho1)
    g1p1 = gamma_1 + 1.
    g1m1 = gamma_1 - 1.
    g1_2 = 2 * gamma_1
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
    rho3 = rho5 * (p3 / p5) ** k13
    u3 = u2
    p5p1 = p5 / p1
    g5m1 = gamma_5 - 1.
    k15 = g5m1 / gamma_5
    T3 = T5 * (p2p1 / p5p1) ** k15
    print(s)

    return np.array([[rho1, u1, p1, T1, gamma_1], [rho2, u2, p2, T2, gamma_1], [rho3, u3, p3, T3, gamma_5],
                     [rho5, u5, p5, T5, gamma_5]]), s


def region_boundaries(properties, geometry_in, shock_speed, time):
    u2 = properties[1][1]
    rho3 = properties[2][0]
    u3 = properties[2][1]
    p3 = properties[2][2]

    gamma_5 = properties[3][4]
    rho5 = properties[3][0]
    p5 = properties[3][2]

    a3 = sound_speed(gamma_5, p3, rho3)
    a5 = sound_speed(gamma_5, p5, rho5)

    x_initial = geometry_in[2]

    x_shock = x_initial + shock_speed * time
    x_contact_surf = x_initial + u2 * time
    x_rare_ft = x_initial + (u3 - a3) * time
    x_rare_hd = x_initial - a5 * time

    return np.array([x_shock, x_contact_surf, x_rare_ft, x_rare_hd])


def lin_region_geom(properties, geometry_in, region_in):
    num_points = geometry_in[3]
    x_initial = geometry_in[2]
    x_array = np.linspace(geometry_in[0], geometry_in[1], num_points)
    rho = np.zeros(num_points)
    u = np.zeros(num_points)
    p = np.zeros(num_points)
    T = np.zeros(num_points)

    x_shock = region_in[0]
    x_contact_surf = region_in[1]
    x_rare_ft = region_in[2]
    x_rare_hd = region_in[3]
    # print(x_shock, x_contact_surf, x_rare_ft, x_rare_hd)

    rho1 = properties[0][0]
    u1 = properties[0][1]
    p1 = properties[0][2]
    T1 = properties[0][3]
    # print(rho1, u1, p1, T1)
    rho2 = properties[1][0]
    u2 = properties[1][1]
    p2 = properties[1][2]
    T2 = properties[1][3]
    # print(rho2, u2, p2, T2)
    rho3 = properties[2][0]
    u3 = properties[2][1]
    p3 = properties[2][2]
    T3 = properties[2][3]
    # print(rho3, u3, p3, T3)
    rho5 = properties[3][0]
    u5 = properties[3][1]
    p5 = properties[3][2]
    T5 = properties[3][3]
    gamma_5 = properties[3][4]
    # print(rho5, u5, p5, T5)

    g5m1 = gamma_5 - 1.
    g5p1 = gamma_5 + 1.
    a5 = sound_speed(gamma_5, p5, rho5)

    for i in range(0, num_points):
        if x_array[i] < x_rare_hd:
            p[i] = p5
            rho[i] = rho5
            u[i] = u5
            T[i] = T5
        elif x_array[i] < x_rare_ft:
            u[i] = 2. / g5p1 * (a5 + (x_array[i] - x_initial) / t)
            k16 = 1. - 0.5 * g5m1 * u[i] / a5
            rho[i] = rho5 * k16 ** (2. / g5m1)
            p[i] = p5 * k16 ** (2. * gamma_5 / g5m1)
            T[i] = T5 * (p[i] / rho[i] / 1e5)
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

    return np.array([x_array, rho, u, p, T])


def exact_solution(rl, rr, geo_in, t_in):
    region_results, s = exact_region_props(rl, rr)
    bounds = region_boundaries(region_results, geo_in, s, t_in)
    return lin_region_geom(region_results, geo_in, bounds)


x, rho, u, p, T = exact_solution(region_left, region_right, geometry, t)

# plt.figure(0)
# plt.title('Case A, Exact Solution Pressure at t=0.2')
# plt.plot(x, p/1e3)
# plt.xlabel('Position (m)')
# plt.ylabel('Pressure (kPa)')
# plt.savefig('exact_p', bbox_inches='tight')
#
# plt.figure(1)
# plt.title('Case A, Exact Solution Density at t=0.2')
# plt.plot(x, rho)
# plt.xlabel('Position (m)')
# plt.ylabel('Density (kg/m^3)')
# plt.savefig('exact_rho', bbox_inches = 'tight')
#
# plt.figure(2)
# plt.title('Case A, Exact Solution Velocity at t=0.2s')
# plt.plot(x, u)
# plt.xlabel('Position (m)')
# plt.ylabel('Velocity (m/s)')
# plt.savefig('exact_u', bbox_inches = 'tight')
#
# plt.figure(3)
# plt.title('Case A, Exact Solution Temperature at t=0.2s')
# plt.plot(x, T)
# plt.xlabel('Position (m)')
# plt.ylabel('Temperature (K)')
# plt.savefig('exact_T', bbox_inches = 'tight')

# plt.show()
