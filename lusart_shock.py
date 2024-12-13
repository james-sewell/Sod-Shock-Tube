import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
from matplotlib import animation


def vector_u(array, gamma):
    """Transform the array of float "array" to get the vector u

    Parameters
    ----------
    array : array of float
        array containing the values of (rho,u,p)
    gamma : float
        constant from the exercise


    Returns
    -------
    vector u : array of float
        array containing the equations of the vector u
    """

    rho = array[0]
    u = array[1]
    p = array[2]
    e_t = p / ((gamma - 1) * rho) + .5 * u ** 2
    return np.array([rho, rho * u, rho * e_t])


def vector_f(u, gamma):
    """Transform the array of float "u" to get the vector f

    Parameters
    ----------
    u : array of float
        array containing the equations of the vector u
    gamma : float
        constant from the exercise


    Returns
    -------
    vector f : array of float
        array containing the equations of the vector f
    """

    u_1 = u[0]
    u_2 = u[1]
    u_3 = u[2]

    return np.array([u_2, u_2 ** 2 / u_1 + (gamma - 1) * (u_3 - .5 * u_2 ** 2 / u_1),
                        (u_3 + (gamma - 1) * (u_3 - .5 * u_2 ** 2 / u_1)) * u_2 / u_1])


def basic(u, nt, nx, gamma):
    """Transform the array of float "u" to get the vector A

    Parameters
    ----------
    u : array of float
        array containing the solution of vector u
    nt : float
        number of grid points in t
    nx : float
        number of grid points in x
    gamma : float
        constant from the exercise


    Returns
    -------
    initial : array of float
        array containing the values of (rho,u,p) throught the whole simulation time

    """
    initial = np.zeros((nt, 3, nx))
    for t in range(nt):
        u_1 = u[t][0]
        u_2 = u[t][1]
        u_3 = u[t][2]
        initial[t] = np.array([u_1, u_2 / u_1, (gamma - 1) * (u_3 - .5 * u_2 ** 2 / u_1)])
    return initial


def shocktube(nx, cdt_left, cdt_right):
    """Computes initial condition with shock tube

    Parameters
    ----------
    nx         : int
        Number of grid points in x
    cdt_left   : list of float
        condition on the left side of the tube with elements (density,velocity,pressure)
    cdt_right  : list of float
        condition on the right side of the tube with elements (density,velocity,pressure)

    Returns
    -------
    array      : array of floats
        Array with initial values of density, velocity and pressure
    """
    x = np.ones(nx)

    middle = int((nx - 1) / 2)

    rho = x.copy() * cdt_left[0]
    rho[middle:] = cdt_right[0]

    u = x.copy() * cdt_left[1]
    u[middle:] = cdt_right[1]

    p = x.copy() * cdt_left[2]
    p[middle:] = cdt_right[2]

    array = np.array([rho, u, p])

    return array


#Basic code parameters
nx = 81
dx = 0.25
dt = 0.0002
gamma = 1.4

#These are included so we don't have to calculate the value of nt (we are getting really lazy)
T = 0.01             # time considered for the simulation
nt = int(T/dt)+1     # number of time steps

# Initial boundary conditons
### from -10 to 0 m inside the tube (left side)
rho_left = 1.        # density in kg m^{-3}
u_left = 0           # velocity in m/s
p_left = 100000      # pressure in kg m^{-2}
## from 0 to 10m inside the tube (right side)
rho_right = 0.125    # density in kg m^{-3}
u_right = 0          # velocity in m/s
p_right = 10000      # pressure in kg m^{-2}

# Boundaries list used with the function shocktube()
condition_left = [rho_left,u_left,p_left]
condition_right = [rho_right,u_right,p_right]


A = shocktube(nx,condition_left,condition_right)

# the spatial grid is onle needed for the graphics
x = np.linspace(-10,10,nx)

plt.figure(0)
plt.plot(x, A[0], color='#003366', ls='-', lw=1)
plt.ylabel('Density')
plt.xlabel('Distance')
plt.ylim(0,1.1);
plt.show()
plt.figure(1)
plt.plot(x, A[1], color='#003366', ls='-', lw=1)
plt.ylabel('Velocity')
plt.xlabel('Distance')
plt.show()
plt.figure(2)
plt.plot(x, A[2], color='#003366', ls='-', lw=1)
plt.ylabel('Pressure')
plt.xlabel('Distance')
plt.ylim(5000,105000);
plt.show()

def dothemath(array, nt, nx, dt, dx, gamma):
    u = np.zeros((nt, 3, nx))
    u[0] = vector_u(array, gamma)

    for t in range(nt - 1):
        un = u[t]
        f = vector_f(un, gamma)
        print(f)
        u_star = 0.5 * (un[:, 1:] + un[:, :-1]) - dt / (2 * dx) * (f[:, 1:] - f[:, :-1])
        f_star = vector_f(u_star, gamma)
        u[t + 1][:, 1:-1] = un[:, 1:-1] - dt / dx * (f_star[:, 1:] - f_star[:, :-1])
        u[t + 1][:, 0] = un[:, 0]
        u[t + 1][:, -1] = un[:, -1]

    return basic(u, nt, nx, gamma)


a = dothemath(A,nt,nx,dt,dx,gamma)

# print(a.np.shape)

plt.figure(3)
plt.plot(x, a[50][0], color='#003366', ls='-', lw=1)
plt.ylabel('Density')
plt.xlabel('Distance')
plt.show()
plt.figure(4)
plt.plot(x, a[50][1], color='#003366', ls='-', lw=1)
plt.ylabel('Velocity')
plt.xlabel('Distance')
plt.show()
plt.figure(5)
plt.plot(x, a[50][2], color='#003366', ls='-', lw=1)
plt.ylabel('Pressure')
plt.xlabel('Distance')
plt.show()

# def animate(data):
#     x = np.linspace(-10,10,nx)
#     y = data
#     line.set_data(x,y)
#     return line,


# fig = plt.figure()
# ax = plt.axes(xlim=(-10,10),ylim=(0,1.),xlabel=('Distance'),ylabel=('Density'))
# line, = ax.plot([],[],color='#003366', lw=2)
#
# A = shocktube(nx,condition_left,condition_right)
# A_n = dothemath(A,nt,nx,dt,dx,gamma)
#
# anim = animation.FuncAnimation(fig, animate, frames=A_n[:,0], interval=50)
# # HTML(anim.to_html5_video())
#
#
# fig = plt.figure();
# ax = plt.axes(xlim=(-10,10),ylim=(0,450.),xlabel=('Distance'),ylabel=('Velocity'));
# line, = ax.plot([],[],color='#003366', lw=2);
#
# anim = animation.FuncAnimation(fig, animate, frames=A_n[:,1], interval=50)
# # HTML(anim.to_html5_video())
#
# fig = plt.figure();
# ax = plt.axes(xlim=(-10,10),ylim=(10000,110000.),xlabel=('Distance'),ylabel=('Pressure'));
# line, = ax.plot([],[],color='#003366', lw=2);
#
# anim = animation.FuncAnimation(fig, animate, frames=A_n[:,2], interval=50)
# HTML(anim.to_html5_video())