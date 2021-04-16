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

#%%
from openbabel import openbabel as ob


def OB_MM_OPT(coords, constrained_indexes, method):
    '''
    return : name of MM-optimized structure

    '''

    mol = ob.OBMol()
    mol.SetCoordinates(coords)

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

    return mol.coordinates()

# %%
