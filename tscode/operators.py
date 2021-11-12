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
from copy import deepcopy
from subprocess import DEVNULL, STDOUT, check_call

import numpy as np
from cclib.io import ccread
from networkx import connected_components
from tscode.ase_manipulations import ase_popt

from tscode.clustered_csearch import clustered_csearch, most_diverse_conformers
from tscode.errors import InputError
from tscode.hypermolecule_class import graphize
from tscode.optimization_methods import optimize
from tscode.settings import (CALCULATOR, DEFAULT_FF_LEVELS, DEFAULT_LEVELS,
                             FF_CALC, FF_OPT_BOOL, PROCS)
from tscode.utils import (loadbar, suppress_stdout_stderr, time_to_string,
                          write_xyz)


def operate(input_string, docker):
    '''
    '''
   
    filename = input_string.split('>')[-1]

    if 'confab>' in input_string:
        outname = confab_operator(filename,
                                    docker.options.calculator,
                                    docker.options.theory_level,
                                    procs=docker.options.procs,
                                    logfunction=docker.log,
                                    let=docker.options.let)

    elif 'csearch>' in input_string:
        outname = csearch_operator(filename, docker)


    elif 'opt>' in input_string:
        outname = opt_operator(filename,
                                    docker.options.calculator,
                                    docker.options.theory_level,
                                    procs=docker.options.procs,
                                    logfunction=docker.log,
                                    let=docker.options.let)

    elif 'neb>' in input_string:
        neb_operator(filename, docker)
        docker.normal_termination()

    return outname

def confab_operator(filename, calculator, theory_level, procs=1, logfunction=None, let=False):
    '''
    '''

    if logfunction is not None:
        logfunction(f'--> Performing conformational search and optimization on {filename}')

    data = ccread(filename)

    if len(data.atomcoords) > 1:
        raise InputError(f'Requested conformational search on file {filename} that already contains more than one structure.')

    if len(tuple(connected_components(graphize(data.atomcoords[0], data.atomnos)))) > 1:
        raise InputError((f'Requested conformational search on a molecular complex (file {filename}). '
                           'Confab is not suited for this task, and using TSCoDe\'s csearch> operator '
                           'is a better idea.'))

    if len(set(data.atomnos) - {1,6,7,8,9,15,16,17,35,53}) != 0:
        raise InputError(('Requested conformational search on a molecule that contains atoms different '
                            'than the ones for which OpenBabel Force Fields are parametrized. Please '
                            'perform this conformational search through the more sophisticated and better '
                            'integrated csearch> operator, part of the TSCoDe program.'))
                                
    confname = filename[:-4] + '_confab.xyz'

    with suppress_stdout_stderr():
        check_call(f'obabel {filename} -O {confname} --confab --rcutoff 0.5 --original'.split(), stdout=DEVNULL, stderr=STDOUT)
        # running Confab through Openbabel

    data = ccread(confname)
    conformers = data.atomcoords
        
    if len(conformers) > 10 and not let:
        conformers = conformers[0:10]
        logfunction(f'Will use only the best 10 conformers for TSCoDe embed.\n')

    os.remove(confname)
    with open(confname, 'w') as f:
        for i, conformer in enumerate(conformers):
            write_xyz(conformer, data.atomnos, f, title=f'Generated conformer {i}')

    return confname

def csearch_operator(filename, docker):
    '''
    '''

    docker.log(f'--> Performing optimization and conformational search on {filename}')

    t_start = time.time()

    data = ccread(filename)

    if len(data.atomcoords) > 1:
        raise InputError(f'Requested conformational search on file {filename} that already contains more than one structure.')
                                
    calc, method, procs = _get_lowest_calc(docker)

    opt_coords = optimize(data.atomcoords[0], data.atomnos, calculator=calc, method=method, procs=procs)[0] if docker.options.optimization else data.atomcoords[0]

    conformers = clustered_csearch(opt_coords, data.atomnos, logfunction=docker.log)

    docker.log(f'Selected the most diverse {len(conformers)} conformers ({time_to_string(time.time()-t_start)})')

    confname = filename[:-4] + '_confs.xyz'
    with open(confname, 'w') as f:
        for i, conformer in enumerate(conformers):
            write_xyz(conformer, data.atomnos, f, title=f'Generated conformer {i}')

    # if len(conformers) > 10 and not docker.options.let:
    #     s += f' Will use only the best 10 conformers for TSCoDe embed.'
    # docker.log(s)

    docker.log('\n')

    return confname

def opt_operator(filename, calculator, theory_level, procs=1, logfunction=None, let=False):
    '''
    '''

    if logfunction is not None:
        logfunction(f'--> Performing {calculator} {theory_level} optimization on {filename} before running TSCoDe')

    t_start = time.time()

    data = ccread(filename)
                                
    conformers = data.atomcoords
    energies = []

    lowest_calc = _get_lowest_calc()
    conformers, energies = _refine_structures(conformers, data.atomnos, *lowest_calc, loadstring='Optimizing conformer')

    loadbar(len(conformers), len(conformers), f'Optimizing conformer {len(conformers)}/{len(conformers)} ')
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

        if len(conformers) > 10 and not let:
            s += f' Will use only the best 10 conformers for TSCoDe embed.'

        logfunction(s+'\n')

    return optname

def neb_operator(filename, docker):
    '''
    '''
    docker.t_start_run = time.time()
    data = ccread(filename)
    assert len(data.atomcoords) == 2, 'NEB calculations need a .xyz input file with two geometries.'

    from tscode.ase_manipulations import ase_neb, ase_popt 

    reagents, products = data.atomcoords
    title = filename[:-4] + '_NEB'

    docker.log(f'--> Performing a NEB TS optimization. Using start and end points from {filename}\n'
               f'Theory level is {docker.options.theory_level} via {docker.options.calculator}')
    _, reag_energy, _ = ase_popt(docker, reagents, data.atomnos, steps=0)
    _, prod_energy, _ = ase_popt(docker, products, data.atomnos, steps=0)

    ts_coords, ts_energy, _ = ase_neb(docker,
                                            reagents,
                                            products,
                                            data.atomnos, 
                                            title=title,
                                            logfile=docker.logfile,
                                            write_plot=True)

    e1 = ts_energy - reag_energy
    e2 = ts_energy - prod_energy

    docker.log(f'NEB completed, relative energy from start/end points (not barrier heights):\n'
               f'  > E(TS)-E(start): {"+" if e1>=0 else "-"}{round(e1, 3)} kcal/mol\n'
               f'  > E(TS)-E(end)  : {"+" if e2>=0 else "-"}{round(e2, 3)} kcal/mol')

    if not (e1 > 0 and e2 > 0):
        docker.log(f'\nNEB failed, TS energy is lower than both the start and end points.\n')

    docker.structures = np.array([ts_coords])
    docker.energies = np.array([ts_energy])
    docker.atomnos = data.atomnos
    docker.write_structures('NEB_TS', indexes=list(range(len(ts_coords))), energies=False, extra=' (see log for energy)')

def _refine_structures(structures, atomnos, calculator, method, procs, loadstring=''):
    '''
    Refine a set of structures.
    '''
    energies = []
    for i, conformer in enumerate(deepcopy(structures)):

        loadbar(i, len(structures), f'{loadstring} {i+1}/{len(structures)} ')

        opt_coords, energy, success = optimize(conformer, atomnos, calculator, method=method, procs=procs)

        if success:
            structures[i] = opt_coords
            energies.append(energy)
        else:
            energies.append(np.inf)

    loadbar(len(structures), len(structures), f'{loadstring} {len(structures)}/{len(structures)} ')
    # optimize the generated conformers

    return structures, energies

def _get_lowest_calc(docker=None):
    '''
    Returns the values for calculator,
    method and processors for the lowest
    theory level available from docker or settings.
    '''
    if docker is None:
        if FF_OPT_BOOL:
            return (FF_CALC, DEFAULT_FF_LEVELS[FF_CALC], None)
        return (CALCULATOR, DEFAULT_LEVELS[CALCULATOR], PROCS)

    if docker.options.ff_opt:
        return (docker.options.ff_calc, docker.options.ff_level, None)
    return (docker.options.calculator, docker.options.theory_level, docker.options.procs)