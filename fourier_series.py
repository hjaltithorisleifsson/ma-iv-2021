import numpy as np

#Author: Hjalti Thor Isleifsson

#Evaluates the sum of the first n frequencies of a Fourier series of a given function.
#x: The point at which the series should be evaluated
#int_f: Integral of the function over one period.
#a_coef: The cosine coefficients. A function which takes in a
#        positive integer and returns a floating point number.
#b_coef: The sine coefficients. A function which takes in a
#        a positive integer and returns a floating point number.
#T: The period
def eval_fourier(x, int_f, a_coef, b_coef, T, n):
	w = 2 * np.pi / T
	y = 0.0 * x #To handle the case when x is a vector and n = 0
	y += int_f / T
	for k in range(1, n + 1):
		tmp = k * x * w
		y += a_coef(k) * np.cos(tmp) + b_coef(k) * np.sin(tmp)

	return y

#Returns sin(n*pi/2) where n is an integer.
def sin_npi2(n):
	if n & 1 == 0:
		return 0.0
	elif (n >> 1) & 1 == 0:
		return 1.0
	else:
		return -1.0

#Computes cos(n*pi/2) where n is an integer.
def cos_npi2(n):
	if n & 1 == 1:
		return 0.0
	elif (n >> 1) & 1 == 0:
		return 1.0
	else:
		return -1.0