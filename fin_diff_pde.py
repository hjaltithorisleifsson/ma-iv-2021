import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Computes an approximate solution to the Dirichlet problem
#
#    -div(p * grad(u)) + q * u = f  on interior of D
#                            u = g  on boundary of D   
#
# where D = [0,a] x [0,b].
def solve_dirichlet_rectangle(p, q, f, g, a, b, h):
	M = round(a / h) + 1
	N = round(b / h) + 1
	hm2 = 1 / (h * h)

	n_non_zero = 2 * M + 2 * (N - 2) + 5 * (N-2) * (M-2)
	row_ind = [0] * n_non_zero
	col_ind = [0] * n_non_zero
	A_data = [0.0] * n_non_zero
	y = np.zeros(M * N)

	A_idx = 0
	idx = 0

	for i in range(M):
		row_ind[A_idx] = idx
		col_ind[A_idx] = idx
		A_data[A_idx] = 1.0
		y[idx] = g(h * i, 0)
		idx += 1
		A_idx += 1

	for j in range(1, N-1):
		row_ind[A_idx] = idx
		col_ind[A_idx] = idx
		A_data[A_idx] = 1.0
		y[idx] = g(0, j * h)
		idx += 1
		A_idx += 1

		for i in range(1, M-1):
			p_e = p((i + 0.5) * h, j * h)
			p_n = p(i * h, (j + 0.5) * h)
			p_w = p((i - 0.5) * h, j * h)
			p_s = p(i * h, (j - 0.5) * h)

			row_ind[A_idx] = idx
			col_ind[A_idx] = idx
			A_data[A_idx] = hm2 * (p_e + p_n + p_w + p_s) + q(i*h, j*h)
			A_idx += 1

			row_ind[A_idx] = idx
			col_ind[A_idx] = idx + 1
			A_data[A_idx] = -p_e * hm2
			A_idx += 1

			row_ind[A_idx] = idx
			col_ind[A_idx] = idx - 1
			A_data[A_idx] = -p_w * hm2
			A_idx += 1

			row_ind[A_idx] = idx
			col_ind[A_idx] = idx - M
			A_data[A_idx] = -p_s * hm2
			A_idx += 1

			row_ind[A_idx] = idx
			col_ind[A_idx] = idx + M
			A_data[A_idx] = -p_n * hm2
			A_idx += 1

			y[idx] = f(i*h, j*h)
			idx += 1

		row_ind[A_idx] = idx
		col_ind[A_idx] = idx
		A_data[A_idx] = 1.0
		A_idx += 1
		y[idx] = g(a, j*h)
		idx += 1

	for i in range(M):
		row_ind[A_idx] = idx
		col_ind[A_idx] = idx
		A_data[A_idx] = 1
		A_idx += 1
		y[idx] = g(h*i, b)
		idx += 1

	A = sp.csc_matrix((A_data, (row_ind, col_ind)), shape=(M * N, M * N))
	c = spsolve(A, y)
	return c

def get_grid(a,b,h):
	M = round(a / h) + 1
	N = round(b / h) + 1
	x = np.linspace(0, a, M)
	y = np.linspace(0, b, N)
	return np.meshgrid(x,y)

def eval_on_grid(a, b, M, N, f):
	h = a / M
	k = b / N
	y = np.zeros((M+1) * (N+1))
	for j in range(N+1):
		for i in range(M+1):
			y[i + j * (M+1)] = f(i * h, j * k)

	return y
