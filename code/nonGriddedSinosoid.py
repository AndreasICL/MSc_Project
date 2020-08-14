from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt

def _2Dsin(x, y, omega1, omega2):
    return np.sin(omega1 * x + omega2 * y)

xn = 50
yn = 50

x = np.linspace(-2, 2, xn)
y = np.linspace(-2, 2, yn)

X, Y = np.meshgrid(x, y)

Z = _2Dsin( X, Y, np.pi, np.pi )

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()
