# %%
from optimization_methods import optimize
from cclib.io import ccread
from ase import Atoms
from ase.visualize import view
from hypermolecule_class import graphize
import os
import numpy as np
os.chdir('Resources/SN2')
mol = ccread('TS_out_test.xyz')
# %%
coords = mol.atomcoords[1]
atoms = Atoms(mol.atomnos, positions=coords)
# %%
view(atoms)
# %%
constraints = np.array([[2,13], [7,15]])
graphs = [graphize(coords[0:10], mol.atomnos[0:10]),
          graphize(coords[10:], mol.atomnos[10:])]

# %%
newcoords = optimize(coords, mol.atomnos, constraints, graphs)[0]
# %%
atoms = Atoms(mol.atomnos, positions=newcoords)
view(atoms)

# %%
constraints

# %%
os.environ['PATH'].split(';')
# os.system('mopac2016')

# %%
import ase
ase.__version__

# %% PM7
%%time
for i in range(len(mol.atomcoords)):
    optimize(mol.atomcoords[i], mol.atomnos, constraints, graphs, method='PM7')
# %%
%%time
for i in range(len(mol.atomcoords)):
    optimize(mol.atomcoords[i], mol.atomnos, constraints, graphs, method='RM1')

