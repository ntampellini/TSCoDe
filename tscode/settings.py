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

# IF YOU MANUALLY EDIT THIS FILE, BE SURE NOT TO
# CHANGE IDENTATION/WHITESPACES/NEWLINES!

OPENBABEL_OPT_BOOL = False
# Whether to run Force Field minimization with
# Openbabel prior to the final one.
# (set to False if no Openbabel Python bindings are available)

CALCULATOR = 'MOPAC'
# Calculator used to run geometry optimization.
# Possibilites are:
# 'MOPAC' : Semiempirical MOPAC2016 (PM7, PM6-DH3, ...)
# 'ORCA' : All methods supported by ORCA
# 'GAUSSIAN' : All methods supported by Gaussian
# 'XTB' : All methods supported by XTB

DEFAULT_LEVELS = {
    'MOPAC':'PM7',
    'ORCA':'PM3',
    'GAUSSIAN':'PM6',
    'XTB':'GFN2-xTB',
}
# Default levels used to run calculations, overridden by LEVEL keyword

COMMANDS = {
    'MOPAC':'MOPAC2016.exe',
    'ORCA':'orca.exe',
    'GAUSSIAN':'g09.exe',
}
# Command with which calculators will be called from the command line

PROCS = 1
# Number of processors (cores) to be used by ORCA and/or Gaussian

MEM_GB = 1
# Memory allocated for each job (Gaussian only). If you experience problems
# in running Gaussian calculation, try setting PROCS to 1 and MEM_GB to 1.
