import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

class SecondDegreeBVO:
	def __init__(self, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
		self.alpha1 = alpha1
		self.beta1 = beta1
		self.gamma1 = gamma1
		self.alpha2 = alpha2
		self.beta2 = beta2
		self.gamma2 = gamma2

class SecondDegreeODE: 
	def __init__(self, a0, a1, a2, f):
		self.a0 = a0
		self.a1 = a1
		self.a2 = a2
		self.f = f

class SecondDegreeBVP:
	def __init__(self, ode, bvo, a, b):
		self.ode = ode
		self.bvo = bvo
		self.a = a
		self.b = b

	def fin_diff(self, N):
		h = (self.b - self.a) / N
		x = np.linspace(self.a + h, self.b - h, N - 1)
		a0_vec = self.ode.a0(x)
		a1_vec = self.ode.a1(x)
		a2_vec = self.ode.a2(x)
		dm1 = np.zeros(N+1)
		d0 = np.zeros(N+1)
		d1 = np.zeros(N+1)

		dm1[0:N-1] = a2_vec / (h ** 2) - a1_vec / (2 * h)
		dm1[N-1] = -self.bvo.beta2 / h

		d0[0] = self.bvo.alpha1 + self.bvo.beta1 / h
		d0[1:N] = -2 * a2_vec / (h ** 2) + a0_vec
		d0[N] = self.bvo.alpha2 + self.bvo.beta2 / h

		d1[1] = -self.bvo.beta1 / h
		d1[2:N+1] = a2_vec / (h ** 2) + a1_vec / (2 * h)

		A = sp.spdiags([dm1, d0, d1], [-1, 0, 1], N+1, N+1, format = 'csc')

		b = np.zeros(N+1)
		b[0] = self.bvo.gamma1
		b[1:N] = self.ode.f(x)
		b[N] = self.bvo.gamma2

		c = spsolve(A,b)
		return c

def main():
	a2 = lambda x: -(1.0+x)
	a1 = lambda x: -1.0
	a0 = lambda x: 2.0
	f = lambda x: x

	ode = SecondDegreeODE(a0, a1, a2, f)
	bvo = SecondDegreeBVO(1, 0, 1, 1, 1, 0)
	bvp = SecondDegreeBVP(ode, bvo, 0, 1)
	print(bvp.fin_diff(1000))

#main()
