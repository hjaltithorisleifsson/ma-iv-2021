from fin_diff_pde import *
import numpy as np
import matplotlib.pyplot as plt

p = lambda x,y: 1.0
q = lambda x,y: 0.0
f = lambda x,y: 0.0
g = lambda x,y: y / ((x+1)**2 + y**2)

it = 9
hs = np.zeros(it)
errors = np.zeros(it)
N = 3
for i in range(it):
	h = 1 / N
	approx = solve_dirichlet_rectangle(p, q, f, g, a = 1, b = 1, h = h)
	ref_sol = eval_on_grid(a = 1, b = 1, M = N, N = N, f = g)
	error = np.max(np.fabs(approx - ref_sol))
	hs[i] = h
	errors[i] = error
	N *= 2

ln_hs = np.log(hs)
ln_e = np.log(errors)
(m,b) = np.polyfit(ln_hs, ln_e, deg = 1)

plt.plot(ln_hs, ln_e)
plt.plot(ln_hs, m * ln_hs + b, label = 'm = %3.2f, b = %3.2f' % (m,b))
plt.legend()
plt.title('ln(h) vs. ln(error)')
plt.show()