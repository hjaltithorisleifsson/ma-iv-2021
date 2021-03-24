from fin_diff_ode import *
import numpy as np 
import matplotlib.pyplot as plt

a2 = lambda x: -x
a1 = lambda x: 0.0 * x - 1.0
a0 = lambda x: 1.0 / x
f = lambda x: 0.0 * x - 2.0

ode = SecondDegreeODE(a0, a1, a2, f)
bvo = SecondDegreeBVO(0.0, 1.0, 0.0, 1.0, -2.0, -2.0)
bvp = SecondDegreeBVP(ode, bvo, 1.0, 2.0)

errors = []
Ns = []
for i in range(10):
	N = 3 * 2 ** i
	Ns.append(N)
	x = np.linspace(1,2,N + 1)
	c = bvp.fin_diff(N)
	u = x * np.log(x) - x
	max_error = np.max(np.fabs(u - c))
	errors.append(max_error)

plt.plot(np.log(Ns), np.log(errors))
plt.show()

#N = 3
#x = np.linspace(1,2,N + 1)
#c = bvp.fin_diff(N)
#u = x * np.log(x) - x
#print(c)
#max_error = np.max(np.fabs(u - c))
#print(max_error)