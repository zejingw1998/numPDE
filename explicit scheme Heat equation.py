

import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0, 200,150)
dt = 0.001
dx = 0.05 

L = 1.0
kappa = 1.0

x = np.arange(0.0, L + dx, dx)
N = x.size

r = kappa * dt / dx**2
if r > 0.5:
    print(f"⚠️ Explicit scheme may be unstable: r = {r:.3f} > 0.5")


T = 200.0
M = int(T / dt) + 1

def heat_equation(H):
    # v[m, j] ≈ v_j^m  (numerical approximation of v at time m*dt and position j*dx)
    v = np.zeros((M, N))

    # Initial condition (at m = 0): use the function H(x) provided by the user
    # Example: H = lambda x: np.sin(np.pi * x)
    v[0, :] = H(x)

    # Boundary conditions (for all time steps)
    v[:, 0]  = 0.0
    v[:, -1] = 0.0

    # Explicit update: only update interior points j = 1 .. N-2
    for m in range(M - 1):
        v[m+1, 1:-1] = v[m, 1:-1] + r * (v[m, 0:-2] - 2.0*v[m, 1:-1] + v[m, 2:])

    return v, x

v, x = heat_equation(lambda x: np.sin(np.pi * x))
plt.plot(x, v[0], label='t=0')
plt.plot(x, v[ int(0.1/dt) ], label='t=0.1')
plt.plot(x, v[-1], label=f't={T}')
plt.xlabel("x")
plt.ylabel("v")
plt.grid(True)
plt.legend()
plt.show()

    
    