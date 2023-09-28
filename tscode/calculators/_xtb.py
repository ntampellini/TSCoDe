# coding=utf-8
'''

TSCODE: Transition State Conformational Docker
Copyright (C) 2021-2023 Nicolò Tampellini

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
from subprocess import DEVNULL, STDOUT, CalledProcessError, check_call

import numpy as np
from tscode.algebra import norm, norm_of
from tscode.utils import clean_directory, read_xyz, write_xyz

def xtb_opt(coords, atomnos, constrained_indices=None,
            constrained_distances=None, method='GFN2-xTB',
            maxiter=None,solvent=None, charge=0, title='temp',
            read_output=True, procs=None, opt=True, conv_thr="tight",
            assert_convergence=False, **kwargs):
    '''
    This function writes an XTB .inp file, runs it with the subprocess
    module and reads its output.

    coords: array of shape (n,3) with cartesian coordinates for atoms.
    atomnos: array of atomic numbers for atoms.
    constrained_indices: array of shape (n,2), with the indices
                         of atomic pairs to be constrained.
    method: string, specifiyng the theory level to be used.
    title: string, used as a file name and job title for the mopac input file.
    read_output: Whether to read the output file and return anything.
    '''
    
    if constrained_indices is not None:
        if len(constrained_indices) == 0:
            constrained_indices = None

    if constrained_distances is not None:
        if len(constrained_distances) == 0:
            constrained_distances = None

    if constrained_distances is not None:

        try:

            for i, (target_d, ci) in enumerate(zip(constrained_distances, constrained_indices)):

                if target_d == None:
                    continue

                if len(ci) == 2:
                    a, b = ci
                else:
                    continue

                d = norm_of(coords[b] - coords[a])
                delta = d - target_d

                if abs(delta) > 0.2:
                    recursive_c_d = constrained_distances.copy()
                    recursive_c_d[i] = target_d + (0.2 * np.sign(d-target_d))
                    # print(f"-------->  d is {round(d, 3)}, target d is {round(target_d, 3)}, delta is {round(delta, 3)}, setting new pretarget at {recursive_c_d}")
                    coords, _, _ = xtb_opt(
                                            coords,
                                            atomnos,
                                            constrained_indices,
                                            constrained_distances=recursive_c_d,
                                            method=method,
                                            title=title,
                                            **kwargs,
                                        )
                
                d = norm_of(coords[b] - coords[a])
                delta = d - target_d
                coords[b] -= norm(coords[b] - coords[a]) * delta
                # print(f"--------> moved atoms from {round(d, 3)} A to {round(norm_of(coords[b] - coords[a]), 3)} A")

        except RecursionError:
            with open(f'{title}_crashed.xyz', 'w') as f:
                write_xyz(coords, atomnos, f, title=title)
            print("Recursion limit reached in constrained optimization - Crashed.")
            quit()

    with open(f'{title}.xyz', 'w') as f:
        write_xyz(coords, atomnos, f, title=title)

    s = f'$opt\n   logfile={title}_opt.log\n$end'
         
    if constrained_indices is not None:
        # s += '\n$constrain\n'
        # for a, b in constrained_indices:
        #     s += '   distance: %s, %s, %s\n' % (a+1, b+1, round(norm_of(coords[a]-coords[b]), 5))
    
        s += '\n$fix\n   atoms: '
        for i in np.unique(np.array(constrained_indices).flatten()):
            s += f"{i+1},"
        s = s[:-1] + "\n"

    if method.upper() in ('GFN-XTB', 'GFNXTB'):
        s += '\n$gfn\n   method=1\n'

    elif method.upper() in ('GFN2-XTB', 'GFN2XTB'):
        s += '\n$gfn\n   method=2\n'
    
    s += '\n$end'

    s = ''.join(s)
    with open(f'{title}.inp', 'w') as f:
        f.write(s)
    
    flags = ''
    
    if opt:
        flags += f'--opt {conv_thr}'
        # specify convergence tightness
    
    if method in ('GFN-FF', 'GFNFF'):

        flags += ' --gfnff'
        # declaring the use of FF instead of semiempirical

    if charge != 0:
        flags += f' --chrg {charge}'

    if procs != None:
        flags += f' -P {procs}'

    if solvent is not None:

        if solvent == 'methanol':
            flags += f' --gbsa methanol'

        else:
            flags += f' --alpb {solvent}'

    elif method.upper() in ('GFN-FF', 'GFNFF'):
        flags += f' --alpb ch2cl2'
        # if using the GFN-FF force field, add CH2Cl2 solvation for increased accuracy

    try:
        with open("temp.log", "w") as f:
            check_call(f'xtb {title}.xyz --input {title}.inp {flags}'.split(), stdout=f, stderr=STDOUT)

    # sometimes the SCC does not converge: only raise the error if specified
    except CalledProcessError:
        if assert_convergence:
            raise CalledProcessError
    
    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    if read_output:
        
        if opt:

            outname = 'xtbopt.xyz'

            if outname in os.listdir():
                coords = read_xyz(outname).atomcoords[0]
                energy = read_xtb_energy(outname)

            else:
                energy = None

            clean_directory((outname, f'{title}.inp', f'{title}.xyz', f'{title}_opt.log'))
        
        else:    
            energy = energy_grepper('temp.log', 'TOTAL ENERGY', 3)
            clean_directory((f'{title}.inp', f'{title}.xyz', f'{title}.log'))

        for filename in ('gfnff_topo', 'charges', 'wbo', 'xtbrestart', 'xtbtopo.mol', '.xtboptok'):
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass

        return coords, energy, True
        
def read_xtb_energy(filename):
    '''
    returns energy in kcal/mol from an XTB
    .xyz result file (xtbopt.xyz)
    '''
    with open(filename, 'r') as f:
        line = f.readline()
        line = f.readline() # second line is where energy is printed
        return float(line.split()[1]) * 627.5096080305927 # Eh to kcal/mol

def xtb_get_free_energy(coords, atomnos, method='GFN2-xTB', solvent=None,
                        charge=0, title='temp', sph=False, **kwargs):
    '''
    '''
    with open(f'{title}.xyz', 'w') as f:
        write_xyz(coords, atomnos, f, title=title)

    s = f'$opt\n   logfile={title}_opt.log\n$end'
          
    if method.upper() in ('GFN-XTB', 'GFNXTB'):
        s += '\n$gfn\n   method=1\n'

    elif method.upper() in ('GFN2-XTB', 'GFN2XTB'):
        s += '\n$gfn\n   method=2\n'
    
    s += '\n$end'

    s = ''.join(s)
    with open(f'{title}.inp', 'w') as f:
        f.write(s)
    
    if sph:
        flags = '--bhess'
    else:
        flags = '--ohess'
    
    if method in ('GFN-FF', 'GFNFF'):
        flags += ' --gfnff'
        # declaring the use of FF instead of semiempirical

    if charge != 0:
        flags += f' --chrg {charge}'

    if solvent is not None:

        if solvent == 'methanol':
            flags += f' --gbsa methanol'

        else:
            flags += f' --alpb {solvent}'

    try:
        with open('temp_hess.log', 'w') as outfile:
            check_call(f'xtb --input {title}.inp {title}.xyz {flags}'.split(), stdout=outfile, stderr=STDOUT)
        
    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    try:
        free_energy = energy_grepper('temp_hess.log', 'TOTAL FREE ENERGY', 4)

        clean_directory()
        for filename in ('gfnff_topo', 'charges', 'wbo', 'xtbrestart', 'xtbtopo.mol', '.xtboptok',
                         'hessian', 'g98.out', 'vibspectrum', 'wbo', 'xtbhess.xyz', 'charges', 'temp_hess.log'):
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass

        return free_energy

    except FileNotFoundError:
        # return np.inf
        print(f'temp_hess.log not present here - we are in', os.getcwd())
        print(os.listdir())
        quit()

def energy_grepper(filename, signal_string, position):
    '''
    returns a kcal/mol energy from a Eh energy in a textfile.
    '''
    with open(filename, 'r') as f:
        line = f.readline()
        while True:
            line = f.readline()
            if signal_string in line:
                return float(line.split()[position]) * 627.5096080305927 # Eh to kcal/mol
            if not line:
                raise Exception()

def xtb_get_free_energy(coords, atomnos, method='GFN2-xTB', solvent=None,
                        charge=0, title='temp', **kwargs):
    '''
    '''
    with open(f'{title}.xyz', 'w') as f:
        write_xyz(coords, atomnos, f, title=title)

    s = f'$opt\n   logfile={title}_opt.log\n$end'
          
    if method.upper() in ('GFN-XTB', 'GFNXTB'):
        s += '\n$gfn\n   method=1\n'

    elif method.upper() in ('GFN2-XTB', 'GFN2XTB'):
        s += '\n$gfn\n   method=2\n'
    
    s += '\n$end'

    s = ''.join(s)
    with open(f'{title}.inp', 'w') as f:
        f.write(s)
    
    flags = '--ohess'
    
    if method in ('GFN-FF', 'GFNFF'):
        flags += ' --gfnff'
        # declaring the use of FF instead of semiempirical

    if charge != 0:
        flags += f' --chrg {charge}'

    if solvent is not None:

        if solvent == 'methanol':
            flags += f' --gbsa methanol'

        else:
            flags += f' --alpb {solvent}'

    try:
        with open('temp_hess.log', 'w') as outfile:
            check_call(f'xtb --input {title}.inp {title}.xyz {flags}'.split(), stdout=outfile, stderr=STDOUT)
        
    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    try:
        free_energy = read_xtb_free_energy('temp_hess.log')

        clean_directory()
        for filename in ('gfnff_topo', 'charges', 'wbo', 'xtbrestart', 'xtbtopo.mol', '.xtboptok',
                         'hessian', 'g98.out', 'vibspectrum', 'wbo', 'xtbhess.xyz', 'charges', 'temp_hess.log'):
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass

        return free_energy

    except FileNotFoundError:
        # return np.inf
        print(f'temp_hess.log not present here - we are in', os.getcwd())
        print(os.listdir())
        quit()

def read_xtb_free_energy(filename):
    '''
    returns free energy in kcal/mol from an XTB
    .xyz result file (xtbopt.xyz)
    '''
    with open(filename, 'r') as f:
        line = f.readline()
        while True:
            line = f.readline()
            if 'TOTAL FREE ENERGY' in line:
                return float(line.split()[4]) * 627.5096080305927 # Eh to kcal/mol
            if not line:
                raise Exception()

def xtb_metadyn_augmentation(coords, atomnos, constrained_indices=None, new_structures:int=5, title=0, debug=False):
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
         
    if constrained_indices is not None:
        s += '\n$constrain\n'
        for a, b in constrained_indices:
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