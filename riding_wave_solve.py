# rho: the density of the fluid
# u   : the velocity of the fluid
# p   : the pressure of the fluid
# e_T : the specific total energy

# matplotlib inline
import numpy as np

# initial condition on the left side
ICL = [1.0, 0.0, 100000.0]
rho_L = ICL[0]
u_L = ICL[1]
p_L = ICL[2]

# initial condition on the right side
ICR = [0.125, 0.0, 10000.0]
rho_R = ICR[0]
u_R = ICR[1]
p_R = ICR[2]

nx = 81
dx = 0.25
dt = 0.0002
gamma = 1.4
T = 0.01  # final time in question "calculate at time t = 0.01 s
nt = int(T / dt) + 1


def search_initial(nx, initial_left, initial_right):
    initial_condition = initial_right * np.ones(nx)
    initial_condition[:int((nx - 1) * 10. / 20.)] = initial_left
    return initial_condition


def computeU(rho, u, p):
    U = np.empty([nx, 3])

    for i in range(nx):
        e = p[i] / ((gamma - 1) * rho[i])
        e_T = e + (u[i] ** 2) / 2
        U[i] = [rho[i], rho[i] * u[i], rho[i] * e_T]
        return U


def computeF(U):
    F = np.empty([nx, 3])
    for i in range(nx):
        U_dum = U[i]

        U1 = U_dum[0]
        U2 = U_dum[1]
        U3 = U_dum[2]

        F[i] = [U2, (U2 ** 2) / U1 + (gamma - 1) * (U3 - 0.5 * (U2 ** 2) / U1), (U3 + (gamma - 1) * (U3 - 0.5 * (U2 ** 2) / U1)) * U2 / U1]
        return F


def Search_U_star(U, F):
    U_star = np.zeros_like(U)

    for i in range(1, nx):
        U_star[i - 1] = 0.5 * (U[i] + U[i - 1]) - dt / (2 * dx) * (F[i] - F[i - 1])
    return U_star


def Search_U_next(U, F_dum):
    U_next = np.empty([nx, 3])
    K2 = np.zeros([nx, 3])
    K1 = np.zeros([nx, 3])

    U_next[0] = U[0]
    U_next[-1] = U[-1]

    for i in range(1, nx - 1):
        U_next[i] = U[i] - dt / dx * (F_dum[i] - F_dum[i - 1])
    return U_next


rho_initial = np.empty(nx)
u_initial = np.empty(nx)
p_initial = np.empty(nx)
U_real = np.empty([nx, 3])

rho_initial = search_initial(nx, rho_L, rho_R)
u_initial = search_initial(nx, u_L, u_R)
p_initial = search_initial(nx, p_L, p_R)
U_real = computeU(rho_initial, u_initial, p_initial)
F_n_prev = np.empty([nx, 3])
U_dummy = np.empty([nx, 3])
F_dummy = np.empty([nx, 3])
U_n = np.empty([nx, 3])
U_49 = np.empty([nx, 3])
# initial condition for U (n=0 -> U_0)
U_n = U_real

# calculate at time n=1 until n=49
for n in range(1, nt - 1):
    U = U_n
    F_n_prev = computeF(U)
    U_dummy = Search_U_star(U, F_n_prev)
    F_dummy = computeF(U_dummy)
    U_n = Search_U_next(U, F_dummy)

U_49 = U_n

# calculate at time n=50 -> t = 0.01 s
F = computeF(U_49)
U_dummy = Search_U_star(U_49, F)
F_dummy = computeF(U_dummy)
U_final = Search_U_next(U_49, F_dummy)

print('Output Final\n')

for i in range(nx):
    Out = U_final[i]

    print(i)
    print(U_final[i])

    if i == 50:
        U1 = Out[0]
        U2 = Out[1]
        U3 = Out[2]

        velocity = U2 / U1
        pressure = (gamma - 1) * (U3 - 0.5 * (U2 ** 2) / U1)
        density = U1

        print('velocity =', velocity)
        print('pressure =', pressure)
        print('density =', density)