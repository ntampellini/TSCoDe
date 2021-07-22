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

OPENBABEL_OPT_BOOL = False
# whether to run Force Field minimization with
# Openbabel prior to the final one.
# (set to False if no Openbabel Python bindings are available)

CALCULATOR = 'MOPAC'
# CALCULATOR = 'ORCA'
# CALCULATOR = 'GAUSSIAN'
# CALCULATOR = 'XTB'

# Calculator used to run geometry optimization.
# Possibilites are:
# 'MOPAC' : Semiempirical MOPAC2016 (PM7, PM6-DH3, ...)
# 'ORCA' : All methods supported by ORCA
# 'GAUSSIAN' : All methods supported by Gaussian

DEFAULT_LEVELS = {
    'MOPAC':'PM7',
    'ORCA':'PM3',
    'GAUSSIAN':'PM7',
    'XTB':'GFN2-xTB'
}

COMMANDS = {
    'MOPAC':'MOPAC2016.exe',
    'ORCA':'orca.exe',
    'GAUSSIAN':'g09.exe'
}
# command with which calculators will be called from the command line

PROCS = 1
# number of processors (cores) to be used by ORCA and/or Gaussian

MEM = '1GB'
# memory allocated for each job (Gaussian only). If you experience problems
# in running Gaussian calculation, try setting PROCS to 1 and MEM to '1GB'.
