import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Shock tube characteristics
geometry = (-150, 150, 0, 500)
t = 0.2
max_iter = 10

nx = geometry[3]
dx = 0.6
gamma = 1.4
T = 0.2
nt = 250
dt = T / nt

# Case A initial conditions
case_a_left = [1., 0., 100e3, 273., 1.4]
case_a_right = [0.125, 0., 10e3, 273., 1.4]

# Case B initial conditions
case_b_left = [1., 0., 100.e3, 300., 1.4]
case_b_right = [0.125, 0., 10.e3, 273, 1.67]


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


def temp_calc(rho_in, p_in, gamma_in):

    return


case_a_exact = exact_solution(case_a_left, case_a_right, geometry, t)

case_b_exact = exact_solution(case_b_left, case_b_right, geometry, t)

A = region_fill_initial(nx, case_b_left, case_b_right)

case_b_x = np.linspace(geometry[0], geometry[1], nx)

nt_lax = 250
dt_lax = T / nt_lax
lax = lax_wendroff(A, nt_lax, nx, dt_lax, dx, gamma)

nt_god = 500
dt_god = T / nt_god
god = godunov(A, nt_god, nx, dt_god, dx, gamma)


plt.figure(0)
plt.plot(case_a_exact[0], case_a_exact[1])
plt.xlabel('Position (m)')
plt.ylabel('Density (kg/m^3)')
plt.savefig('case_a_exact_rho')

plt.figure(1)
plt.plot(case_a_exact[0], case_a_exact[2])
plt.xlabel('Position (m)')
plt.ylabel('Velocity (m/s)')
plt.savefig('case_a_exact_u')

plt.figure(2)
plt.plot(case_a_exact[0], case_a_exact[3]/1e3)
plt.xlabel('Position (m)')
plt.ylabel('Pressure (kPa)')
plt.savefig('case_a_exact_p')

plt.figure(3)
plt.plot(case_a_exact[0], case_a_exact[4])
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.savefig('case_a_exact_T')

plt.figure(4)
plt.plot(case_a_exact[0], case_b_exact[1])
plt.plot(case_a_exact[0], lax[nt_lax-1][0], 'x')
plt.ylabel('Density (kg/m^3)')
plt.xlabel('Distance (m)')
plt.savefig('lax_wen_rho')

plt.figure(5)
plt.plot(case_a_exact[0], case_b_exact[2])
plt.plot(case_a_exact[0], lax[nt_lax-1][1], 'x')
plt.ylabel('Velocity (m/s)')
plt.xlabel('Distance (m)')
plt.savefig('lax_wen_u')

plt.figure(6)
plt.plot(case_a_exact[0], case_b_exact[3]/1e3)
plt.plot(case_a_exact[0], lax[nt_lax-1][2]/1e3, 'x')
plt.ylabel('Pressure (kPa)')
plt.xlabel('Distance (m)')
plt.savefig('lax_wen_p')

plt.figure(7)
plt.plot(case_a_exact[0], case_b_exact[4])
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.savefig('lax_wen_T')

plt.figure(8)
plt.plot(case_a_exact[0], case_b_exact[1])
plt.plot(case_a_exact[0], god[nt_god-1][0], 'x')
plt.ylabel('Density (kg/m^3)')
plt.xlabel('Distance (m)')
plt.savefig('god_rho')

plt.figure(9)
plt.plot(case_a_exact[0], case_b_exact[2])
plt.plot(case_a_exact[0], god[nt_god-1][1], 'x')
plt.ylabel('Velocity (m/s)')
plt.xlabel('Distance (m)')
plt.savefig('god_u')

plt.figure(10)
plt.plot(case_a_exact[0], case_b_exact[3]/1e3)
plt.plot(case_a_exact[0], god[nt_god-1][2]/1e3, 'x')
plt.ylabel('Pressure (kPa)')
plt.xlabel('Distance (m)')
plt.savefig('god_p')

plt.figure(11)
plt.plot(case_a_exact[0], case_b_exact[4])
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.savefig('god_T')

# plt.show()
