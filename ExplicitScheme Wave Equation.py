import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, exp, pi


t_end = 10.0
x_end = 10.0
dt = 0.1
dx = 0.1
c  = 1.0                        

t = np.arange(0, t_end + dt, dt)
x = np.arange(0, x_end + dx, dx)

M = len(t)
J = len(x)
r2 = (c * dt / dx)**2

def waveequation(H):
    """
    Explicit 1D wave equation: u_tt = c^2 u_xx
    H(x): initial displacement u(x,0)
    """
    v = np.zeros((M, J))
    v[0, :] = H(x)

    # Boundary conditions
    v[:, 0]  = 0.0
    v[:, -1] = 0.0

    # First step (using Taylor expansion)
    v[1, 1:-1] = v[0, 1:-1] + 0.5 * r2 * (v[0, 0:-2] - 2*v[0, 1:-1] + v[0, 2:])
    v[1, 0]  = 0.0
    v[1, -1] = 0.0

    
    for m in range(1, M-1):
        v[m+1, 1:-1] = (2*v[m, 1:-1] - v[m-1, 1:-1]
                        + r2 * (v[m, 0:-2] - 2*v[m, 1:-1] + v[m, 2:]))
        v[m+1, 0]  = 0.0
        v[m+1, -1] = 0.0

    return v, x

# Example
v, x = waveequation(lambda x: np.sin(pi * x / x_end))

plt.plot(x, v[0], label='t=0')
plt.plot(x, v[M//4], label=f't≈{t[M//4]:.1f}')
plt.plot(x, v[M//2], label=f't≈{t[M//2]:.1f}')
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.title("1D Wave Equation (Explicit Scheme)")
plt.grid(True)
plt.tight_layout()
plt.show()
