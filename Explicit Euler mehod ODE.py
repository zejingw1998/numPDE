import numpy as np
import matplotlib.pyplot as plt

k = 100
dt = 0.001  

t = np.arange(0, 0.2, dt)  
y = np.zeros_like(t)
y[0] = 20.0

for n in range(len(t) - 1):
    y[n+1] = y[n] - k * y[n] * dt 

y_exact = 20.0 * np.exp(-k * t)

plt.plot(t, y, 'b-',  label='Explicit Euler (dt=0.001)')
plt.plot(t, y_exact, 'r--', label='Exact')
plt.legend()
plt.xlabel('t'); plt.ylabel('y(t)')
plt.title("Explicit Euler (stable dt) for y' = -k*y, k=100")
plt.grid(True)
plt.show()
