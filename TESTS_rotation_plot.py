import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R

def _orient_along_x(array, vector):
    '''
    :params array:    array of atomic coordinates arrays: len(array) structures with len(array[i]) atoms
    :params vector:   list of shape (1,3) with anchor vector to align to the x axis
    :return:          array, aligned so that vector is on x
    '''
    rotation_matrix = R.align_vectors(np.array([[1,0,0]]), np.array([vector]))[0].as_matrix()
    return np.array([rotation_matrix @ v for v in array])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

vectors = np.array([[1,1,0],
                    [1,0,1],
                    [0,1,1]])

x = [v[0] for v in vectors]
y = [v[1] for v in vectors]
z = [v[2] for v in vectors]

for i in range(len(x)):
    ax.plot([0, x[i]], [0, y[i]], [0, z[i]], label='original', color='r')

vectors = _orient_along_x(vectors, vectors[0])

x = [v[0] for v in vectors]
y = [v[1] for v in vectors]
z = [v[2] for v in vectors]

for i in range(len(x)):
    ax.plot([0, x[i]], [0, y[i]], [0, z[i]], label='rotated', color='b')

ax.legend()
plt.show()