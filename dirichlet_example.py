from fin_diff_pde import *
import numpy as np
import matplotlib.pyplot as plt

p = lambda x,y: x**2 + y**2
q = lambda x,y: 1.0
f = lambda x,y: 0.0
g = lambda x,y: np.cos(x * y + 2 * x + y)

N = 100
a = np.pi
b = np.pi
h = np.pi / N

approx = solve_dirichlet_rectangle(p, q, f, g, a, b, h).reshape((N+1, N+1))
x,y = np.meshgrid(np.linspace(0, a, N+1), np.linspace(0, b, N+1))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, approx)
plt.show()