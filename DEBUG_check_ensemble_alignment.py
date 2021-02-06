from openbabel import openbabel as ob
import os
import numpy as np
from ensemble_to_object import Density_object

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('Resources')

test = 'funky_single_aligned_rdkit.sdf'
 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cclib.io import ccread

if test.endswith('.sdf'):
    outname = test.split('.')[0] + '.xyz'
    os.system(f'obabel {test} -o xyz -O {outname}')
    aligned = ccread(outname)
    os.remove(outname)
else:
    aligned = ccread(test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

aligned_atoms = np.array([atom for structure in aligned.atomcoords for atom in structure])
x = [atom[0] for atom in aligned_atoms]
y = [atom[1] for atom in aligned_atoms]
z = [atom[2] for atom in aligned_atoms]

col = {
    1:'gray',
    6:'black',
    7:'blue',
    8:'red'
}
col_list = [col[i] for i in aligned.atomnos]
col_list_full = []
for s in aligned.atomcoords:
    for c in col_list:
        col_list_full.append(c)

# x = x[6:8]
# y = y[6:8]
# z = z[6:8]
# col_list_full = col_list_full[6:8]


plot = ax.scatter(x, y, z, c=col_list_full, label='aligned')
ax.legend()
plt.show()