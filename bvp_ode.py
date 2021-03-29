import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def fdm(N, a, b, a0, a1, a2, f, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
	h = (b - a) / N
	x = np.linspace(a + h, b - h, N - 1)
	a0_vec = a0(x)
	a1_vec = a1(x)
	a2_vec = a2(x)
	dm1 = np.zeros(N+1)
	d0 = np.zeros(N+1)
	d1 = np.zeros(N+1)

	dm1[0:N-1] = a2_vec / (h ** 2) - a1_vec / (2 * h)
	dm1[N-1] = -beta2 / h

	d0[0] = alpha1 + beta1 / h
	d0[1:N] = -2 * a2_vec / (h ** 2) + a0_vec
	d0[N] = alpha2 + beta2 / h

	d1[1] = -beta1 / h
	d1[2:N+1] = a2_vec / (h ** 2) + a1_vec / (2 * h)

	A = sp.spdiags([dm1, d0, d1], [-1, 0, 1], N+1, N+1, format = 'csc')

	b = np.zeros(N+1)
	b[0] = gamma1
	b[1:N] = f(x)
	b[N] = gamma2

	c = spsolve(A,b)
	return c

def fem(N, a, b, p, q, f, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
	h = (b - a) / N
	m = np.linspace(a + 0.5 * h, b - 0.5 * h, N)
	ph_vec = p(m) / h
	hq_vec = q(m) * h
	x = np.linspace(a, b, N+1)
	f_vec = f(x)

	dm1 = np.zeros(N+1)
	d0 = np.zeros(N+1)
	d1 = np.zeros(N+1)
	b = np.zeros(N+1)

	dm1[0:N-1] = -ph_vec[0:N-1] + hq_vec[0:N-1] / 6.0

	d0[1:N] = ph_vec[0:N-1] + ph_vec[1:N] + (hq_vec[0:N-1] + hq_vec[1:N]) / 3.0

	d1[2:N+1] = -ph_vec[1:N] + hq_vec[1:N] / 6.0

	b[1:N] = h * (f_vec[0:N-1] + 4 * f_vec[1:N] + f_vec[2:N+1]) / 6.0

	if beta1 == 0:
		d0[0] = 1.0
		b[0] = gamma1 / alpha1
	else:
		pa = p(a)
		d0[0] = ph_vec[0] + hq_vec[0] / 3.0 + pa * alpha1 / beta1
		d1[1] = -ph_vec[0] + hq_vec[0] / 6.0
		b[0] = h * (2 * f_vec[0] + f_vec[1]) / 6.0 + pa * gamma1 / beta1

	if beta2 == 0:
		d0[N] = 1.0
		b[N] = gamma2 / alpha2
	else:
		pb = p(b)
		d0[N] = ph_vec[N-1] + hq_vec[N-1] + pb * alpha2 / beta2
		dm1[N-1] = -ph_vec[N-1] + hq_vec[N-1] / 6.0
		b[N] = h * (f_vec[N-1] + 2 * f_vec[N]) / 6.0 + pb * gamma2 / beta2

	A = sp.spdiags([dm1, d0, d1], [-1, 0, 1], N+1, N+1, format = 'csc')
 
	c = spsolve(A,b) #Ac = b
	return c

