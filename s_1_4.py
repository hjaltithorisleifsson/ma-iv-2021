import fourier_series as fs 
import math
import numpy as np 
import matplotlib.pyplot as plt

f = lambda x: np.fabs(x - 2 * math.pi * np.floor(x / (2 * math.pi)) - math.pi)

int_f = math.pi ** 2
a = lambda n: 0 if n & 1 == 0 else 4 / (math.pi * n * n)
b = lambda n: 0

T = 2 * math.pi

res = 100

x = np.array([T * i / res for i in range(res)])
y = f(x)
y1 = fs.eval_fourier(x, int_f, a, b, T, 1)
y5 = fs.eval_fourier(x, int_f, a, b, T, 5)
y10 = fs.eval_fourier(x, int_f, a, b, T, 10)

plt.plot(x,y)
plt.plot(x,y1)
plt.plot(x,y5)
plt.plot(x,y10)

plt.show()