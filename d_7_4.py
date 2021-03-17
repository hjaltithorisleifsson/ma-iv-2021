import numpy as np 
import matplotlib.pyplot as plt

x_2 = np.array([i / 1000 for i in range(-2 * 1000, 2 * 1000)])
fx = np.where(np.abs(x_2) < 1, 1 - x_2**2, 0)
plt.plot(x_2, fx)
plt.show()

plt.clf()

x_25 = np.array([i / 1000 for i in range(-25 * 1000, 25 * 1000)])
Ffx = (4 / x_25 ** 3) * (np.sin(x_25) - x_25 * np.cos(x_25))
plt.plot(x_25, Ffx)
plt.show()

plt.close()