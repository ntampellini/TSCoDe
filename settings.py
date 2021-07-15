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
# Openbabel prior to the MOPAC semiempirical one.
# (set to False if no Openbabel Python bindings are available)

CALCULATOR = 'MOPAC'
# CALCULATOR = 'ORCA'
# CALCULATOR = 'GAUSSIAN'

# Calculator used to run geometry optimization.
# Possibilites are:
# 'MOPAC' : Semiempirical MOPAC2016 (PM7, PM6-DH3, ...)
# 'ORCA' : All methods supported by ORCA
# 'GAUSSIAN' : All methods supported by Gaussian

MOPAC_DEFAULT_LEVEL = 'PM7'
# Default theory level for MOPAC

ORCA_DEFAULT_LEVEL = 'PM3'
# Default theory level for ORCA

GAUSSIAN_DEFAULT_LEVEL = 'PM6'
# Default theory level for Gaussian

MOPAC_COMMAND = 'MOPAC2016.exe'
# command with which MOPAC will be called from the command line

ORCA_COMMAND = 'orca.exe'
# command with which ORCA will be called from the command line.
# Inserting the full path to orca.exe is required for parallel runs
# with more than one core! See ORCA documentation.

GAUSSIAN_COMMAND = 'g09.exe'
# command with which Gaussian will be called from the command line

PROCS = 1
# number of processors (cores) to be used by ORCA and/or Gaussian

MEM = '1GB'
# memory allocated for each job (Gaussian only). If you experience problems
# in running Gaussian calculation, try setting PROCS to 1 and MEM to '1GB'.
