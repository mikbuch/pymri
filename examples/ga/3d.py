import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin

def fitness(x, y, category):
    if category == 1:
        z = x - y
    else:
        z = y - x
    return z

n = 1000
xs = randrange(n, 0, 1)
ys = randrange(n, 0, 1)
category_0 = []
category_1 = []

for x, y in zip(xs, ys):
    category_0.append(fitness(x, y, 0))
    category_1.append(fitness(x, y, 1))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, category_0, c='b', marker='o', s=40)
ax.scatter(xs, ys, category_1, c='g', marker=',', s=40)
ax.scatter(1, 0, 1, c='r', marker='^', s=100)
ax.scatter(0, 1, 1, c='y', marker='s', s=100)
# import pdb
# pdb.set_trace()

ax.set_xlabel('y_1')
ax.set_ylabel('y_2')
ax.set_zlabel('fitness value')

plt.show()
