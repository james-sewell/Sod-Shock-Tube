import numpy as np

a = 1
b = 2
c = 3

region = [a, b, c]

print(region)
print(region[2])

nx = 81
dx = 0.25
dt = 0.004
gamma = 1.4
T = 0.2
nt = int(T/dt)+1

u = np.zeros((nt, 3, nx))