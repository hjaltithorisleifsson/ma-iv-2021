from fin_diff_ode import *
import numpy as np

a2 = lambda x: -1.0 / x
a1 = lambda x: 1 / x**2
a0 = lambda x: x
f = lambda x: 0.0 * x

ode = SecondDegreeODE(a0, a1, a2, f)
bvo = SecondDegreeBVO(1.0, 0.0, 1.0 / np.sqrt(np.e), 1.0, 0.5, 0.0)
bvp = SecondDegreeBVP(ode, bvo, 1.0, 2.0)

x = np.linspace(1,2,3 + 1)
c = bvp.fin_diff(3)
u = np.exp(-0.5 * x**2)
max_error = np.max(np.fabs(u - c))
print(max_error)