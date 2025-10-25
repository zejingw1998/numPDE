
"""
1D Heat Equation Solver (Backward Euler Method)
------------------------------------------------
This script solves the one-dimensional heat equation:

    u_t = κ * u_xx,   for 0 < x < L, 0 < t <= T

with Dirichlet boundary conditions:
    u(0,t) = u(L,t) = 0

and an initial condition u(x,0) = H(x).

The numerical method used is the implicit Backward Euler scheme,
which is unconditionally stable.
"""

import numpy as np
import matplotlib.pyplot as plt

def heat_equation(H, L=1.0, kappa=1.0, dx=0.05, dt=0.001, T=200.0):
    """
    Solve 1D heat equation u_t = kappa * u_xx using Backward Euler scheme.

    Parameters
    ----------
    H : callable
        Initial condition function, H(x).
    L : float
        Domain length (default 1.0).
    kappa : float
        Diffusion coefficient (default 1.0).
    dx : float
        Spatial step size (default 0.05).
    dt : float
        Time step size (default 0.001).
    T : float
        Final simulation time (default 200.0).

    Returns
    -------
    v : ndarray of shape (M, N)
        Numerical solution. v[m, j] ≈ u(x_j, t_m).
    x : ndarray of shape (N,)
        Spatial grid.
    t : ndarray of shape (M,)
        Time grid.
    """
   
    x = np.arange(0.0, L + dx, dx)
    N = x.size
    M = int(np.round(T / dt)) + 1
    t = np.linspace(0.0, T, M)

    r = kappa * dt / dx**2
    print(f"[Backward Euler] r = {r:.3f} (stable). N={N}, steps={M-1}")

   
    main = (1.0 + 2.0 * r) * np.ones(N)
    off  = (-r) * np.ones(N - 1)
    A = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)

    # Apply Dirichlet BC: set first and last rows to identity
    A[0, :]  = 0.0; A[0, 0]   = 1.0
    A[-1, :] = 0.0; A[-1, -1] = 1.0

    v = np.zeros((M, N))
    v[0, :] = H(x)           # initial condition
    v[0, 0]  = 0.0
    v[0, -1] = 0.0

    for m in range(M - 1):
        b = v[m].copy()
        b[0]  = 0.0
        b[-1] = 0.0
        v[m + 1, :] = np.linalg.solve(A, b)
        v[m + 1, 0]  = 0.0
        v[m + 1, -1] = 0.0

    return v, x, t


print("Running demo example for the heat equation...")

# Define parameters
L = 1.0
kappa = 1.0
dx = 0.05
dt = 0.001
T  = 1.0  # shorter for quick test

# Example initial condition: sin(pi*x)
def H(x):
    return np.sin(np.pi * x)

# Solve the equation
v, x, t = heat_equation(H, L=L, kappa=kappa, dx=dx, dt=dt, T=T)

# Plot results at selected times
def idx_at(timepoint):
    return int(round(timepoint / dt))

plt.figure(figsize=(7,4))
plt.plot(x, v[0],            label='t=0')
plt.plot(x, v[idx_at(0.1)],  label='t=0.1')
plt.plot(x, v[-1],           label=f't={T:g}')
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("1D Heat Equation – Backward Euler Scheme")
plt.grid(True)
plt.legend()
plt.show()
