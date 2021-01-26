import fourier_series as fs 
import math
import numpy as np 
import matplotlib.pyplot as plt

f = lambda x: x - 2 * math.pi * np.floor(x / (2 * math.pi)) < math.pi * 0.5

int_f = math.pi * 0.5

a = lambda n: fs.sin_npi2(n) / (math.pi * n)
b = lambda n: (1.0 - fs.cos_npi2(n)) / (math.pi * n) 

T = 2 * math.pi

res = 1000

x = np.array([T * i / res - math.pi for i in range(res)])
y = f(x)
y0 = fs.eval_fourier(x, int_f, a, b, T, 0)
y1 = fs.eval_fourier(x, int_f, a, b, T, 1)
y10 = fs.eval_fourier(x, int_f, a, b, T, 10)
y1000 = fs.eval_fourier(x, int_f, a, b, T, 1000)

plt.plot(x,y)
plt.plot(x,y0)
plt.plot(x,y1)
plt.plot(x,y10)
plt.plot(x,y1000)

plt.show()