# coding=utf-8
'''

TSCODE: Transition State Conformational Docker
Copyright (C) 2021-2022 Nicolò Tampellini

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

FF_OPT_BOOL = True
# Whether to run Force Field optimization with
# prior to the final one. Set to False if no
# Openbabel/XTB programs and python bindings
# are installed.

FF_CALC = 'XTB'
# Calculator to perform Force Field optimizations.
# Possibilites are:
# 'OB' : Openbabel UFF and MMFF94 methods
# 'GAUSSIAN' : FF methods supported by Gaussian (UFF, MMFF)
# 'XTB' : GFN-FF method

DEFAULT_FF_LEVELS = {
    ### DO NOT REMOVE
    ### THESE TWO LINES
    'GAUSSIAN':'UFF',
    'XTB':'GFN-FF',
    'OB':'MMFF94',
}
# Default levels used to run calculations, overridden by FFLEVEL keyword

CALCULATOR = 'XTB'
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

PROCS = 8
# Number of processors (cores) to be used by ORCA and/or Gaussian

MEM_GB = 1
# Memory allocated for each job (Gaussian only). If you experience problems
# in running Gaussian calculation, try setting PROCS to 1 and MEM_GB to 0.5.