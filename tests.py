from openbabel import openbabel as ob
import os
import numpy as np
from ensemble_to_object import Density_object

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('Resources')
Density_object('dienamine.xyz')
 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cclib.io import ccread

original = ccread('dienamine.xyz')
aligned = ccread('dienamine_aligned.xyz')

structs = zip(np.array(original.atomcoords), np.array(aligned.atomcoords))
diffs = [np.sum(tup[0] - tup[1]) for tup in structs]
for i, v in enumerate(diffs):
    print(f'Structure {i} - {v} differences')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

original_atoms = np.array([atom for structure in original.atomcoords for atom in structure])
x1 = [atom[0] for atom in original_atoms]
y1 = [atom[1] for atom in original_atoms]
z1 = [atom[2] for atom in original_atoms]

aligned_atoms = np.array([atom for structure in aligned.atomcoords for atom in structure])
x2 = [atom[0] for atom in aligned_atoms]
y2 = [atom[1] for atom in aligned_atoms]
z2 = [atom[2] for atom in aligned_atoms]

plot = ax.scatter(x1, y1, z1, color='r')
plot = ax.scatter(x2, y2, z2, color='b')
plt.show()