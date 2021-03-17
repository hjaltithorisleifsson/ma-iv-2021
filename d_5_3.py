import matplotlib.pyplot as plt
import numpy as np 
import math

xbase = np.array([i / 1000 for i in range(1, 1000)]) * np.pi

n_min = -1
n_max = 3
for i in range(n_min, n_max):
	xi = xbase + (i - 0.5) * np.pi
	tanxi = np.tan(xi)
	plt.plot(xi, tanxi, color = 'b')

f = lambda x: 2 * x / (1 - x**2)

xm1 = np.array([i / 1000 for i in range(int((n_min - 0.5) * np.pi * 1000), -1000)])
yxm1 = f(xm1)
plt.plot(xm1, yxm1, color = 'r')

xmid = np.array([i / 1000 for i in range(-1000, 1000)])
yxmid = f(xmid)
plt.plot(xmid, yxmid, color = 'r')

xp1 = np.array([i / 1000 for i in range(1000, int((n_max - 0.5) * np.pi * 1000))])
yxp1 = f(xp1)
plt.plot(xp1, yxp1, color = 'r')

plt.ylim([-10,10])
plt.show()

plt.clf()

g = lambda x: 2 * x / (1 + x**2)

xmid = np.array([i / 1000 for i in range(0, 1000)])
yxmid = g(xmid)
plt.plot(xmid, yxmid, color = 'r')

xp1 = np.array([i / 1000 for i in range(1000, 5000)])
yxp1 = g(xp1)
plt.plot(xp1, yxp1, color = 'r')

x = np.array([i / 1000 for i in range(0, 5000)])
y = np.tanh(x)
plt.plot(x,y, color = 'b')

plt.ylim([0,1.5])
plt.show()