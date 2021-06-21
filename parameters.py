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
import os

OPENBABEL_OPT_BOOL = False
# whether to run Force Field minimization with
# Openbabel prior to the MOPAC semiempirical one.
# (set to False if no Openbabel Python bindings are available)

CALCULATOR = 'MOPAC'
# CALCULATOR = 'ORCA'

# Calculator used to run force field minimizations.
# Possibilites are:
# 'MOPAC' : Semiempirical MOPAC2016 (PM7, PM6-DH3, ...)
# 'ORCA' : All methods supported by ORCA

MOPAC_COMMAND = 'MOPAC2016.exe'
# command with which MOPAC will be called from the command line

ORCA_COMMAND = 'orca.exe'
# command with which ORCA will be called from the command line.
# Inserting the full path to orca.exe is required for parallel runs
# with more than one core! See ORCA documentation.

ORCA_PROCS = 1
# number of processors (cores) to be used by ORCA

orb_dim_dict = {
    'H Single Bond' : 0.85,
    'C Single Bond' : 1,
    'O Single Bond' : 1,
    'N Single Bond' : 1,
    'F Single Bond' : 1,
    'Cl Single Bond' : 1.5,
    'Br Single Bond' : 1.5,
    'I Single Bond' : 2,

    'C sp' : 1,
    'N sp' : 1,

    'B sp2' : 0.8,
    'C sp2' : 1.1,
    'N sp2' : 1,

    'C sp3' : 1,
    'Br sp3' : 1,

    'O Ether' : 1,
    'S Ether' : 1,

    'O Ketone': 0.85,
    'S Ketone': 1,

    'N Imine' : 1,

    'C bent carbene' : 1,

    'Fallback' : 1
                }       
# Half-lenght of the transition state bonding distance involving a given atom

nci_dict={
    # tag in alphabetical order
    'FF':(3.5,'F-F interaction'),
    'HO':(2,'O-H hydrogen bond'),
    'HN':(2,'N-H hydrogen bond'),
    'HPh':(2.8,'H-Ar non-conventional hydrogen bond'), # taken from https://doi.org/10.1039/C1CP20404A
    'PhPh':(3.8, 'pi-stacking interaction'),            # guessed from https://doi.org/10.1039/C2SC20045G
    'FPh':(0.0, 'F-Ar halogen-bonding interaction'),
    'ClPh':(0.0, 'Cl-Ar halogen-bonding interaction'),
    'BrPh':(0.0, 'Br-Ar halogen-bonding interaction'),
    'IPh':(0.0, 'I-Ar halogen-bonding interaction'),

    # TODO - halogen bonds, pi bonds interactions
}
# non covalent interaction threshold and types for atomic pairs