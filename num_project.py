import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

#Author: Hjalti Thor Isleifsson
#Date: 2.4.2021

def helmholtz_eq(a, b, h, l, w, v):
	M = round(a / h) + 1
	N = round(b / h) + 1
	hm2 = 1 / (h * h)
	l2 = l * l

	n_non_zero = 5 * (M-2) * (N-2) + 2 * M + 8 * (N-2)
	row_ind = [0] * n_non_zero
	col_ind = [0] * n_non_zero
	A_data = [0.0] * n_non_zero
	y = np.zeros(M * N)

	A_idx = 0
	#Handle inner points
	for j in range(1, N-1):
		for i in range(1, M-1):
			k = i + M * j

			row_ind[A_idx] = k
			col_ind[A_idx] = k
			A_data[A_idx] = 4 * hm2 - l2
			A_idx += 1

			row_ind[A_idx] = k
			col_ind[A_idx] = k + 1
			A_data[A_idx] = -hm2
			A_idx += 1

			row_ind[A_idx] = k
			col_ind[A_idx] = k - 1
			A_data[A_idx] = -hm2
			A_idx += 1

			row_ind[A_idx] = k
			col_ind[A_idx] = k - M
			A_data[A_idx] = -hm2
			A_idx += 1

			row_ind[A_idx] = k
			col_ind[A_idx] = k + M
			A_data[A_idx] = -hm2
			A_idx += 1


	#Handle lower boundary
	for k in range(M):
		row_ind[A_idx] = k
		col_ind[A_idx] = k
		A_data[A_idx] = 1.0
		A_idx += 1

	y[0:M] = w(np.linspace(0, a, M))

	#Handle upper boundary
	k = (N - 1) * M
	for i in range(M):
		row_ind[A_idx] = k
		col_ind[A_idx] = k
		A_data[A_idx] = 1.0
		A_idx += 1
		k += 1

	y[-M:] = v(np.linspace(0, a, M)) 

	#Handle left boudnary
	for j in range(1, N-1):
		k = j * M

		row_ind[A_idx] = k
		col_ind[A_idx] = k
		A_data[A_idx] = 4 * hm2 - l2
		A_idx += 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k + 1
		A_data[A_idx] = -2 * hm2
		A_idx += 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k - M
		A_data[A_idx] = -hm2
		A_idx += 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k + M
		A_data[A_idx] = -hm2
		A_idx += 1

	#Handle right boundary
	for j in range(1, N-1):
		k = (j + 1) * M - 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k
		A_data[A_idx] = 4 * hm2 - l2
		A_idx += 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k - 1
		A_data[A_idx] = -2 * hm2
		A_idx += 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k - M
		A_data[A_idx] = -hm2
		A_idx += 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k + M
		A_data[A_idx] = -hm2
		A_idx += 1


	A = csr_matrix((A_data, (row_ind, col_ind)), shape=(M * N, M * N))
	c = spsolve(A, y)
	HZ = c.reshape((N, M))
	return HZ

def laplace(a, b, h, psi1, psi2):
	M = round(a / h) + 1
	N = round(b / h) + 1

	n_non_zero = 5 * (M - 2) * (N - 2) + 8 * (N - 2) + 2 * M
	row_ind = [0] * n_non_zero
	col_ind = [0] * n_non_zero
	A_data = [0.0] * n_non_zero
	
	A_idx = 0

	#Handle inner points
	for i in range(1, M-1):
		for j in range(1, N-1):
			k = i + M * j

			row_ind[A_idx] = k
			col_ind[A_idx] = k
			A_data[A_idx] = 4.0
			A_idx += 1

			row_ind[A_idx] = k
			col_ind[A_idx] = k - 1
			A_data[A_idx] = -1.0
			A_idx += 1

			row_ind[A_idx] = k
			col_ind[A_idx] = k + 1
			A_data[A_idx] = -1.0
			A_idx += 1

			row_ind[A_idx] = k
			col_ind[A_idx] = k - M
			A_data[A_idx] = -1.0
			A_idx += 1

			row_ind[A_idx] = k
			col_ind[A_idx] = k + M
			A_data[A_idx] = -1.0
			A_idx += 1

	#Lower boundary
	for k in range(M):
		row_ind[A_idx] = k
		col_ind[A_idx] = k
		A_data[A_idx] = 1.0
		A_idx += 1

	#Upper boundary
	for k in range(M * (N-1), M*N):
		row_ind[A_idx] = k
		col_ind[A_idx] = k
		A_data[A_idx] = 1.0
		A_idx += 1

	x = np.linspace(0, a, M)
	y = np.zeros(M*N)
	y[0:M] = psi1(x)
	y[M*(N-1):M*N] = psi2(x)

	#Handle left boundary
	for j in range(1, N-1):
		k = j * M

		row_ind[A_idx] = k
		col_ind[A_idx] = k
		A_data[A_idx] = 2.0
		A_idx += 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k + 1
		A_data[A_idx] = -1.0
		A_idx += 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k - M
		A_data[A_idx] = -0.5
		A_idx += 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k + M
		A_data[A_idx] = -0.5
		A_idx += 1

	#Handle right boundary
	for j in range(1, N-1):
		k = M * (j+1) - 1
		row_ind[A_idx] = k
		col_ind[A_idx] = k
		A_data[A_idx] = 2.0
		A_idx += 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k - 1
		A_data[A_idx] = -1.0
		A_idx += 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k - M
		A_data[A_idx] = -0.5
		A_idx += 1

		row_ind[A_idx] = k
		col_ind[A_idx] = k + M
		A_data[A_idx] = -0.5
		A_idx += 1

	A = csr_matrix((A_data, (row_ind, col_ind)), shape=(M * N, M * N))
	c = spsolve(A, y)
	approx = c.reshape((N, M))
	return approx

##########################################################################
# Part 1
##########################################################################

def part1():
	#II
	a = 1.0
	b = 1.0
	h = 0.25
	l = 0.01
	w = lambda x: 0.0 * x + 1.0
	v = lambda x: 0.0 * x
	HZ = helmholtz_eq(a, b, h, l, w, v)
	print(HZ)

	#III
	a = 1.0
	b = 2.0
	h = 0.05
	l = 1
	w = lambda x: 0.0 * x + 1.0
	v = lambda x: 0.0 * x
	HZ = helmholtz_eq(a, b, h, l, w, v)

	u = lambda x,y: np.sin(l * (b - y)) / np.sin(l * b)

	M = round(a / h) + 1
	N = round(b / h) + 1

	(x,y) = np.meshgrid(np.linspace(0, a, M), np.linspace(0, b, N))
	correct = u(x,y)

	error = np.max(np.fabs(correct - HZ))
	print(error)

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	ax.plot_surface(x,y,correct)
	ax.plot_surface(x,y,HZ)
	plt.show()

	#IV
	a = 1.0
	b = 1.0
	h = 0.001
	l = 10
	u0 = 10
	u1 = 1
	w = lambda x: -u0 * x / a * ((x / a - 1)**2) * (1 + x/a)
	v = lambda x: u1 * x / a * (1 - x/a) * (1 + x/a)**2
	HZ = helmholtz_eq(a, b, h, l, w, v)

	M = round(a / h) + 1
	N = round(b / h) + 1

	(x,y) = np.meshgrid(np.linspace(0, a, M), np.linspace(0, b, N))

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	ax.plot_surface(x,y,HZ)
	plt.show()

part1()

########################################################################
# Part 2
########################################################################

def part2():
	#II
	beta1 = 1.0
	beta2 = 0.0
	a = 1.0
	b = 1.0
	h = 0.01

	sqrt2rec = 1.0 / np.sqrt(2.0)

	psi1 = lambda x: np.sin(4 * np.pi * (x - 0.25))
	psi2 = lambda x: np.sin(4 * np.pi * x)

	approx = laplace(a, b, h, psi1, psi2)

	M = round(a / h) + 1
	N = round(b / h) + 1

	(x,y) = np.meshgrid(np.linspace(0, a, M), np.linspace(0, b, N))

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	ax.plot_surface(x,y,approx)
	plt.show()

#part2()

