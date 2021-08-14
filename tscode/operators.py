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
import time
import numpy as np
from copy import deepcopy
from utils import InputError, suppress_stdout_stderr, write_xyz,time_to_string
from subprocess import check_call, DEVNULL, STDOUT
from optimization_methods import optimize
from cclib.io import ccread
from hypermolecule_class import graphize
from networkx import connected_components

def operate(input_string, calculator, theory_level, procs=1, logfunction=None):
    '''
    '''
    
    filename = input_string.split('>')[-1]

    if 'csearch>' in input_string:
        outname = csearch_operator(filename, calculator, theory_level, procs=procs, logfunction=logfunction)

    elif 'opt>' in input_string:
        outname = opt_operator(filename, calculator, theory_level, procs=procs, logfunction=logfunction)

    return outname

def csearch_operator(filename, calculator, theory_level, procs=1, logfunction=None):
    '''
    '''

    if logfunction is not None:
        logfunction(f'--> Performing conformational search and optimization on {filename}')

    t_start = time.time()

    data = ccread(filename)

    if len(data.atomcoords) > 1:
        raise InputError(f'Requested conformational search on file {filename} that already contains more than one structure.')

    if len(tuple(connected_components(graphize(data.atomcoords[0], data.atomnos)))) > 1:
        raise InputError((f'Requested conformational search on a multimolecular file ({filename}). '
                            'This is probably a bad idea, as the OpenBabel conformational search '
                            'algorithm implemented here is quite basic and is not suited for this '
                            'task. A much better idea is to generate conformations for this complex '
                            'by a more sophisticated software.'))

    if len(set(data.atomnos) - {1,6,7,8,9,15,16,17,35,53}) != 0:
        raise InputError(('Requested conformational search on a molecule that contains atoms different '
                            'than the ones for which OpenBabel Force Field is parametrized. Please consider '
                            'performing this conformational search on a different, more sophisticated software.'))
                                
    confname = filename[:-4] + '_confs.xyz'

    with suppress_stdout_stderr():
        check_call(f'obabel {filename} -O {confname} --confab --rcutoff 1 --original'.split(), stdout=DEVNULL, stderr=STDOUT)

    data = ccread(confname)
    conformers = data.atomcoords
    energies = []

    for i, conformer in enumerate(deepcopy(conformers)):
    # optimize the generated conformers
        
        opt_coords, energy, success = optimize(calculator, conformer, data.atomnos, method=theory_level, procs=procs)

        if success:
            conformers[i] = opt_coords
            energies.append(energy)

        else:
            energies.append(np.inf)

    energies = np.array(energies) - np.min(energies)
    energies, conformers = zip(*sorted(zip(energies, conformers), key=lambda x: x[0]))
    # sorting structures based on energy

    os.remove(confname)
    with open(confname, 'w') as f:
        for i, conformer in enumerate(conformers):
            write_xyz(conformer, data.atomnos, f, title=f'Generated conformer {i} - Rel. E. = {round(energies[i], 3)} kcal/mol')

    if logfunction is not None:
        s = 's' if len(conformers) > 1 else ''
        s = f'Completed conformational search and {calculator} {theory_level} optimization - {len(conformers)} conformer{s}. ({time_to_string(time.time()-t_start)}).'
        
        if len(conformers) > 5:
            s += f' Will use the best 5 conformers for TSCoDe embed.'

        logfunction(s+'\n')

    return confname

def opt_operator(filename, calculator, theory_level, procs=1, logfunction=None):
    '''
    '''

    if logfunction is not None:
        logfunction(f'--> Performing {calculator} {theory_level} optimization on {filename} before running TSCoDe')

    from cclib.io import ccread

    t_start = time.time()

    data = ccread(filename)
                                
    conformers = data.atomcoords
    energies = []

    for i, conformer in enumerate(deepcopy(conformers)):

        opt_coords, energy, success = optimize(calculator, conformer, data.atomnos, method=theory_level, procs=procs)

        if success:
            conformers[i] = opt_coords
            energies.append(energy)

        else:
            energies.append(np.inf)
    # optimize the generated conformers

    energies = np.array(energies) - np.min(energies)
    energies, conformers = zip(*sorted(zip(energies, conformers), key=lambda x: x[0]))
    # sorting structures based on energy

    optname = filename[:-4] + '_opt.xyz'
    with open(optname, 'w') as f:
        for i, conformer in enumerate(conformers):
            write_xyz(conformer, data.atomnos, f, title=f'Optimized conformer {i} - Rel. E. = {round(energies[i], 3)} kcal/mol')

    if logfunction is not None:
        s = 's' if len(conformers) > 1 else ''
        s = f'Completed optimization on {len(conformers)} conformer{s}. ({time_to_string(time.time()-t_start)}).'

        if len(conformers) > 5:
            s += f' Will use the best 5 conformers for TSCoDe embed.'

        logfunction(s+'\n')

    return optname