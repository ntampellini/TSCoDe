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
import os
from subprocess import DEVNULL, STDOUT, check_call

import numpy as np
from tscode.algebra import norm_of
from tscode.utils import (clean_directory, read_xyz, write_xyz)


def xtb_opt(coords, atomnos, constrained_indexes=None, method='GFN2-xTB', solvent=None, title='temp', read_output=True, **kwargs):
    '''
    This function writes an XTB .inp file, runs it with the subprocess
    module and reads its output.

    :params coords: array of shape (n,3) with cartesian coordinates for atoms.
    :params atomnos: array of atomic numbers for atoms.
    :params constrained_indexes: array of shape (n,2), with the indexes
                                 of atomic pairs to be constrained.
    :params method: string, specifiyng the theory level to be used.
    :params title: string, used as a file name and job title for the mopac input file.
    :params read_output: Whether to read the output file and return anything.
    '''

    with open(f'{title}.xyz', 'w') as f:
        write_xyz(coords, atomnos, f, title=title)

    s = f'$opt\n   logfile={title}_opt.log\n$end'
         
    if constrained_indexes is not None:
        s += '\n$constrain\n'
        for a, b in constrained_indexes:
            s += '   distance: %s, %s, %s\n' % (a+1, b+1, round(norm_of(coords[a]-coords[b]), 5))
    
    if method.upper() in ('GFN-XTB', 'GFNXTB'):
        s += '\n$gfn\n   method=1\n'

    elif method.upper() in ('GFN2-XTB', 'GFN2XTB'):
        s += '\n$gfn\n   method=2\n'
    
    s += '\n$end'

    s = ''.join(s)
    with open(f'{title}.inp', 'w') as f:
        f.write(s)
    
    flags = '--opt'
    
    if method in ('GFN-FF', 'GFNFF'):
        flags += ' tight'
        # tighter convergence for GFN-FF works better

        flags += ' --gfnff'
        # declaring the use of FF instead of semiempirical


    if solvent is not None:

        if solvent == 'methanol':
            flags += f' --gbsa methanol'

        else:
            flags += f' --alpb {solvent}'

    elif method.upper() in ('GFN-FF', 'GFNFF'):
        flags += f' --alpb thf'

    try:
        check_call(f'xtb --input {title}.inp {title}.xyz {flags} > temp.log 2>&1'.split(), stdout=DEVNULL, stderr=STDOUT)

    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    if read_output:

        try:
            outname = 'xtbopt.xyz'
            opt_coords = read_xyz(outname).atomcoords[0]
            energy = read_xtb_energy(outname)

            clean_directory()
            os.remove(outname)

            for filename in ('gfnff_topo', 'charges', 'wbo', 'xtbrestart', 'xtbtopo.mol', '.xtboptok'):
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    pass

            return opt_coords, energy, True

        except FileNotFoundError:
            return None, None, False

def read_xtb_energy(filename):
    '''
    returns energy in kcal/mol from an XTB
    .xyz result file (xtbotp.xyz)
    '''
    with open(filename, 'r') as f:
        line = f.readline()
        line = f.readline() # second line is where energy is printed
        return float(line.split()[1]) * 627.5096080305927 # Eh to kcal/mol

def xtb_metadyn_augmentation(coords, atomnos, constrained_indexes=None, new_structures:int=5, title=0, debug=False):
    '''
    Runs a metadynamics simulation (MTD) through
    the XTB program to obtain new conformations.
    The GFN-FF force field is used.
    '''
    with open(f'temp.xyz', 'w') as f:
        write_xyz(coords, atomnos, f, title='temp')

    s = (
        '$md\n'
        '   time=%s\n' % (new_structures) +
        '   step=1\n'
        '   temp=300\n'
        '$end\n'
        '$metadyn\n'
        '   save=%s\n' % (new_structures) +
        '$end'
        )
         
    if constrained_indexes is not None:
        s += '\n$constrain\n'
        for a, b in constrained_indexes:
            s += '   distance: %s, %s, %s\n' % (a+1, b+1, round(norm_of(coords[a]-coords[b]), 5))

    s = ''.join(s)
    with open(f'temp.inp', 'w') as f:
        f.write(s)

    try:
        check_call(f'xtb --md --input temp.inp temp.xyz --gfnff > Structure{title}_MTD.log 2>&1'.split(), stdout=DEVNULL, stderr=STDOUT)

    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    structures = [coords]
    for n in range(1,new_structures):
        name = 'scoord.'+str(n)
        structures.append(parse_xtb_out(name))
        os.remove(name)

    for filename in ('gfnff_topo', 'xtbmdoc', 'mdrestart'):
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

    # if debug:
    os.rename('xtb.trj', f'Structure{title}_MTD_traj.xyz')

    # else:
    #     os.remove('xtb.traj')  

    structures = np.array(structures)

    return structures

def parse_xtb_out(filename):
    '''
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()

    coords = np.zeros((len(lines)-3,3))

    for l, line in enumerate(lines[1:-2]):
        coords[l] = line.split()[:-1]

    return coords * 0.529177249 # Bohrs to Angstroms
