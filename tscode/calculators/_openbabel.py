# coding=utf-8
'''

TSCODE: Transition State Conformational Docker
Copyright (C) 2021 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''
from cclib.io import ccread
from tscode.settings import FF_OPT_BOOL, FF_CALC
from tscode.utils import scramble_check, write_xyz

if FF_OPT_BOOL and FF_CALC == 'OB':

    from openbabel import openbabel as ob

    def openbabel_opt(structure, atomnos, constrained_indexes, graphs=None, check=False, method='UFF', **kwargs):
        '''
        return : MM-optimized structure (UFF/MMFF)
        '''

        assert not check or graphs is not None, 'Either provide molecular graphs or do not check for scrambling.'

        filename='temp_ob_in.xyz'

        with open(filename, 'w') as f:
            write_xyz(structure, atomnos, f)

        outname = 'temp_ob_out.xyz'

        # Standard openbabel molecule load
        conv = ob.OBConversion()
        conv.SetInAndOutFormats('xyz','xyz')
        mol = ob.OBMol()
        more = conv.ReadFile(mol, filename)
        i = 0

        # Define constraints
        constraints = ob.OBFFConstraints()

        for a, b in constrained_indexes:

            first_atom = mol.GetAtom(int(a+1))
            length = first_atom.GetDistance(int(b+1))

            constraints.AddDistanceConstraint(int(a+1), int(b+1), length)       # Angstroms
            # constraints.AddAngleConstraint(1, 2, 3, 120.0)      # Degrees
            # constraints.AddTorsionConstraint(1, 2, 3, 4, 180.0) # Degrees

        # Setup the force field with the constraints
        forcefield = ob.OBForceField.FindForceField(method)
        forcefield.Setup(mol, constraints)
        forcefield.SetConstraints(constraints)

        # Do a 500 steps conjugate gradient minimization
        # (or less if converges) and save the coordinates to mol.
        forcefield.ConjugateGradients(500)
        forcefield.GetCoordinates(mol)
        energy = forcefield.Energy()

        # Write the mol to a file
        conv.WriteFile(mol,outname)
        conv.CloseOutFile()

        opt_coords = read_xyz(outname).atomcoords[0]

        if check:
            success = scramble_check(opt_coords, atomnos, constrained_indexes, graphs)
        else:
            success = True

        return opt_coords, energy, success