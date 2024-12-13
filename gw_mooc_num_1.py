import numpy as np
import matplotlib.pyplot as plt

# Set parameters
nx = 41
L = 2.
dx = L / (nx -1)
nt = 25
dt = 0.02
c = 1.0

x = np.linspace(0., L, nx)

u0 = np.ones(nx)
mask = np.where(np.logical_and(x >= 0.5, x <= 1.0))
print(mask)

u0[mask] = 2.0
print(u0)


for n in range(1, nt):
    un = u0.copy()
    for i in range(1, nx):
        un[i] = un[i] - un[i] * dt / dx * (un[i] - un[i-1])

u = u0.copy()
for n in range(1, nt):
    u[1:] = u[1:] - dt / dx * u[1:] * (u[1:] - u[:-1])

print(u)

plt.title('Initial Conditions')
plt.xlabel('x')
plt.ylabel('u')
plt.grid
plt.plot(x, u0, label='Initial', color='C0', linestyle='--')
plt.plot(x, un, label='nt = {}'.format(nt), color='C1', linestyle='-')
plt.plot(x - 0.7, u, label='nt = {}'.format(nt), color='C2', linestyle=':')
plt.legend()
plt.xlim(0., L)
plt.ylim(0., 2.5)
plt.show()
plt.savefig('gw_mooc_num_1', bbox_inches='tight')