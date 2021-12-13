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
from subprocess import DEVNULL, STDOUT, check_call

from tscode.utils import read_xyz
from tscode.settings import COMMANDS
from tscode.solvents import get_solvent_line
from tscode.utils import clean_directory, pt


def orca_opt(coords, atomnos, constrained_indexes=None, method='PM3', procs=1, solvent=None, title='temp', read_output=True):
    '''
    This function writes an ORCA .inp file, runs it with the subprocess
    module and reads its output.

    :params coords: array of shape (n,3) with cartesian coordinates for atoms.
    :params atomnos: array of atomic numbers for atoms.
    :params constrained_indexes: array of shape (n,2), with the indexes
                                 of atomic pairs to be constrained.
    :params method: string, specifiyng the first line of keywords for the MOPAC input file.
    :params title: string, used as a file name and job title for the mopac input file.
    :params read_output: Whether to read the output file and return anything.
    '''

    s = '! %s Opt\n\n# ORCA input generated by TSCoDe\n\n' % (method)

    if solvent is not None:
        s += '\n' + get_solvent_line(solvent, 'ORCA', method) + '\n'

    if procs > 1:
        s += f'%pal nprocs {procs} end\n'

    if constrained_indexes is not None:
        s += f'%{""}geom\nConstraints\n'
        # weird f-string to prevent python misinterpreting %

        for a, b in constrained_indexes:
            s += '{B %s %s C}\n' % (a, b)

        s += 'end\nend\n\n'

    s += '*xyz 0 1\n'

    for i, atom in enumerate(coords):
        s += '%s     % .6f % .6f % .6f\n' % (pt[atomnos[i]].symbol, atom[0], atom[1], atom[2])

    s += '*\n'

    s = ''.join(s)
    with open(f'{title}.inp', 'w') as f:
        f.write(s)
    
    try:
        check_call(f'{COMMANDS["ORCA"]} {title}.inp'.split(), stdout=DEVNULL, stderr=STDOUT)

    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    if read_output:

        try:
            opt_coords = read_xyz(f'{title}.xyz').atomcoords[0]
            energy = read_orca_property(f'{title}_property.txt')

            clean_directory()

            return opt_coords, energy, True

        except FileNotFoundError:
            return None, None, False

def read_orca_property(filename):
    '''
    Read energy from ORCA property output file
    '''
    energy = None

    with open(filename, 'r') as f:

        while True:
            line = f.readline()

            if not line:
                break

            if 'SCF Energy:' in line:
                energy = float(line.split()[2])

    return energy