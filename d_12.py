import numpy as np
from bvp_ode import *
import matplotlib.pyplot as plt

p = lambda x: 1.0 / (1.0 + x**2)
q = lambda x: -2.0 / (1.0 + x**2)**2

a0 = q
a1 = lambda x: 2*x / (1.0 + x**2)**2
a2 = lambda x: -p(x)

f = lambda x: -1.0 / (1.0 + x**2)**2
alpha1 = 1.0
beta1 = -1.0
gamma1 = 0.0
alpha2 = 1.0
beta2 = 0.0
gamma2 = 1.0

N = 6
approx_fem = fem(N, 0.0, 1.0, p, q, f, alpha1, beta1, gamma1, alpha2, beta2, gamma2)
approx_fdm = fdm(N, 0.0, 1.0, a0, a1, a2, f, alpha1, beta1, gamma1, alpha2, beta2, gamma2)

diff = np.max(np.fabs(approx_fem - approx_fdm))
print(diff)

######################################################################################

u = lambda x: 0.5 * (2.0 + np.exp(2.0 - x) - np.exp(x))

p = lambda x: 0.0 * x + 1.0
q = lambda x: 0.0 * x + 1.0
f = lambda x: 0.0 * x + 1.0

a0 = q
a1 = lambda x: 0.0 * x
a2 = lambda x: 0.0 * x - 1.0

Ns = []
errors_fem = []
errors_fdm = []

for i in range(10):
	approx_fem = fem(N, 0.0, 1.0, p, q, f, alpha1, beta1, gamma1, alpha2, beta2, gamma2)
	approx_fdm = fdm(N, 0.0, 1.0, a0, a1, a2, f, alpha1, beta1, gamma1, alpha2, beta2, gamma2)

	x = np.linspace(0.0, 1.0, N + 1)
	correct = u(x)

	error_fem = np.max(np.fabs(approx_fem - correct))
	error_fdm = np.max(np.fabs(approx_fdm - correct))
	Ns.append(N)
	errors_fem.append(error_fem)
	errors_fdm.append(error_fdm)
	N *= 2

log_Ns = np.log10(Ns)
log_fem = np.log10(errors_fem)
log_fdm = np.log10(errors_fdm)

m_fem,b_fem = np.polyfit(log_Ns, log_fem, deg = 1)
m_fdm,b_fdm = np.polyfit(log_Ns, log_fdm, deg = 1)

xx = np.linspace(log_Ns[0], log_Ns[-1])

print(m_fem)
print(m_fdm)

plt.plot(log_Ns, log_fem, label = 'FEM')
plt.plot(xx, m_fem * xx + b_fem, label = 'FEM trendline')
plt.plot(log_Ns, log_fdm, label = 'FDM')
plt.plot(xx, m_fdm * xx + b_fdm, label = 'FDM trendline')
plt.legend()
plt.show()


