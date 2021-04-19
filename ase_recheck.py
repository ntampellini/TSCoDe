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
# view(atoms)
# %%
constraints = np.array([[2,13], [7,15]])
graphs = [graphize(coords[0:10], mol.atomnos[0:10]),
          graphize(coords[10:], mol.atomnos[10:])]

# %%
# newcoords = optimize(coords, mol.atomnos, constraints, graphs)[0]
# %%
# atoms = Atoms(mol.atomnos, positions=newcoords)
# view(atoms)

# # %% PM7
# %%time
# for i in range(len(mol.atomcoords)):
#     optimize(mol.atomcoords[i], mol.atomnos, constraints, graphs, method='PM7')
# # %%
# %%time
# for i in range(len(mol.atomcoords)):
#     optimize(mol.atomcoords[i], mol.atomnos, constraints, graphs, method='PM7 THREADS=8')

# %%
%load_ext cython
#%%
%%cython
from ctypes import c_double
import numpy as np
cimport numpy as np

cdef double* c_array(np.ndarray[np.float64_t, mode='c'] array):
    # cpdef double* carr = array
    return carr
# %%cython
from openbabel import openbabel as ob
# from openbabel cimport openbabel as ob
import numpy as np
# cimport numpy as np
# from ctypes import c_double

def OB_MM_OPT(coords, constrained_indexes, method = 'UFF'):
    '''
    return : name of MM-optimized structure

    '''
    # cdef double* carr = coords
    # carr = c_double * len(coords)
    # carr(*coords)

    mol = ob.OBMol()
    mol.SetCoordinates(c_array(coords))

    # Define constraints

    first_atom = mol.GetAtom(constrained_indexes[0]+1)
    length = first_atom.GetDistance(constrained_indexes[1]+1)

    constraints = ob.OBFFConstraints()
    constraints.AddDistanceConstraint(constrained_indexes[0]+1, constrained_indexes[1]+1, length)       # Angstroms
    # constraints.AddAngleConstraint(1, 2, 3, 120.0)      # Degrees
    # constraints.AddTorsionConstraint(1, 2, 3, 4, 180.0) # Degrees

    # Setup the force field with the constraints
    forcefield = ob.OBForceField.FindForceField(method)
    forcefield.Setup(mol, constraints)
    forcefield.SetConstraints(constraints)

    # Do a 500 steps conjugate gradient minimiazation
    # and save the coordinates to mol.
    forcefield.ConjugateGradients(500)
    forcefield.GetCoordinates(mol)

    # cpdef np.ndarray[np.float64_t, ndim=2] opt_coords = mol.GetCoordinates()

    # return opt_coords

# %%
OB_MM_OPT(coords, constraints)
# %%
vec = c_double * 3

x = np.array([[1,3,4],
              [4,6,2],
              [5,1,4]])
l = []
for v in x:
    l.append(vec(*v))
    
carr = c_double * len(x)
carr(*l)
carr
# %%
