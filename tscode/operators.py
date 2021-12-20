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
from networkx import connected_components

from tscode.clustered_csearch import clustered_csearch, most_diverse_conformers
from tscode.errors import InputError
from tscode.graph_manipulations import graphize
from tscode.optimization_methods import optimize
from tscode.settings import (CALCULATOR, DEFAULT_FF_LEVELS, DEFAULT_LEVELS,
                             FF_CALC, FF_OPT_BOOL, PROCS)
from tscode.utils import (get_scan_peak_index, loadbar, read_xyz,
                          suppress_stdout_stderr, time_to_string, write_xyz)


def operate(input_string, embedder):
    '''
    Perform the operations according to the chosen
    operator and return the outname of the (new) .xyz
    file to read instead of the input one.
    '''
   
    filename = embedder._extract_filename(input_string)

    if 'confab>' in input_string:
        outname = confab_operator(filename,
                                    embedder.options,
                                    logfunction=embedder.log)

    elif 'csearch_opt>' in input_string:
        conf_name = csearch_operator(filename, embedder)
        outname = opt_operator(conf_name,
                                embedder,
                                logfunction=embedder.log)

    elif 'csearch>' in input_string:
        outname = csearch_operator(filename, embedder)


    elif 'opt>' in input_string:
        outname = opt_operator(filename,
                                embedder,
                                logfunction=embedder.log)

    elif 'scan>' in input_string:
        scan_operator(filename, embedder)
        embedder.normal_termination()

    elif 'neb>' in input_string:
        neb_operator(filename, embedder)
        embedder.normal_termination()

    elif 'prune>' in input_string:
        outname = filename
        # this operator is accounted for in the OptionSetter
        # class of Options, set when the Embedder calls _set_options

    return outname

def confab_operator(filename, options, logfunction=None):
    '''
    '''

    if logfunction is not None:
        logfunction(f'--> Performing conformational search and optimization on {filename}')

    data = read_xyz(filename)

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

    data = read_xyz(confname)
    conformers = data.atomcoords
        
    if len(conformers) > 10 and not options.let:
        conformers = conformers[0:10]
        logfunction(f'Will use only the best 10 conformers for TSCoDe embed.\n')

    os.remove(confname)
    with open(confname, 'w') as f:
        for i, conformer in enumerate(conformers):
            write_xyz(conformer, data.atomnos, f, title=f'Generated conformer {i}')

    return confname

def csearch_operator(filename, embedder):
    '''
    '''

    embedder.log(f'--> Performing conformational search on {filename}')

    t_start = time.perf_counter()

    data = read_xyz(filename)

    if len(data.atomcoords) > 1:
        embedder.log(f'Requested conformational search on multimolecular file - will do\n' +
                      'an individual search from each conformer (might be time-consuming).')
                                
    calc, method, procs = _get_lowest_calc(embedder)
    conformers = []

    for i, coords in enumerate(data.atomcoords):

        opt_coords = optimize(coords, data.atomnos, calculator=calc, method=method, procs=procs)[0] if embedder.options.optimization else coords
        # optimize starting structure before running csearch

        conf_batch = clustered_csearch(opt_coords, data.atomnos, title=f'{filename}, conformer {i+1}', logfunction=embedder.log)
        # generate the most diverse conformers starting from optimized geometry

        conformers.append(conf_batch)

    conformers = np.array(conformers)
    batch_size = conformers.shape[1]

    conformers = conformers.reshape(-1, data.atomnos.shape[0], 3)
    # merging structures from each run in a single array

    if embedder.embed is not None:
        embedder.log(f'\nSelected the most diverse {batch_size} out of {conformers.shape[0]} conformers for {filename} ({time_to_string(time.perf_counter()-t_start)})')
        conformers = most_diverse_conformers(batch_size, conformers, data.atomnos)

    confname = filename[:-4] + '_confs.xyz'
    with open(confname, 'w') as f:
        for i, conformer in enumerate(conformers):
            write_xyz(conformer, data.atomnos, f, title=f'Generated conformer {i}')

    # if len(conformers) > 10 and not embedder.options.let:
    #     s += f' Will use only the best 10 conformers for TSCoDe embed.'
    # embedder.log(s)

    embedder.log('\n')

    return confname

def opt_operator(filename, embedder, logfunction=None):
    '''
    '''

    if logfunction is not None:
        logfunction(f'--> Performing {embedder.options.calculator} {embedder.options.theory_level} optimization on {filename}')

    t_start = time.perf_counter()

    data = read_xyz(filename)
                                
    conformers = data.atomcoords
    energies = []

    lowest_calc = _get_lowest_calc(embedder)
    conformers, energies = _refine_structures(conformers, data.atomnos, *lowest_calc, loadstring='Optimizing conformer')

    energies, conformers = zip(*sorted(zip(energies, conformers), key=lambda x: x[0]))
    energies = np.array(energies) - np.min(energies)
    conformers = np.array(conformers)
    # sorting structures based on energy

    mask = energies < 10
    # getting the structures to reject (Rel Energy > 10 kcal/mol)

    if logfunction is not None:
        s = 's' if len(conformers) > 1 else ''
        s = f'Completed optimization on {len(conformers)} conformer{s}. ({time_to_string(time.perf_counter()-t_start)}).\n'

        if max(energies) > 10:
            s += f'Discarded {len(conformers)-np.count_nonzero(mask)}/{len(conformers)} unstable conformers (Rel. E. > 10 kcal/mol)\n'

    conformers, energies = conformers[mask], energies[mask]
    # applying the mask that rejects high energy confs

    optname = filename[:-4] + '_opt.xyz'
    with open(optname, 'w') as f:
        for i, conformer in enumerate(conformers):
            write_xyz(conformer, data.atomnos, f, title=f'Optimized conformer {i} - Rel. E. = {round(energies[i], 3)} kcal/mol')

        logfunction(s+'\n')

    return optname

def neb_operator(filename, embedder):
    '''
    '''
    embedder.t_start_run = time.perf_counter()
    data = read_xyz(filename)
    assert len(data.atomcoords) == 2, 'NEB calculations need a .xyz input file with two geometries.'

    from tscode.ase_manipulations import ase_neb, ase_popt 

    reagents, products = data.atomcoords
    title = filename[:-4] + '_NEB'

    embedder.log(f'--> Performing a NEB TS optimization. Using start and end points from {filename}\n'
               f'Theory level is {embedder.options.theory_level} via {embedder.options.calculator}')

    print('Getting start point energy...', end='\r')
    _, reag_energy, _ = ase_popt(embedder, reagents, data.atomnos, steps=0)

    print('Getting end point energy...', end='\r')
    _, prod_energy, _ = ase_popt(embedder, products, data.atomnos, steps=0)

    ts_coords, ts_energy, _ = ase_neb(embedder,
                                            reagents,
                                            products,
                                            data.atomnos, 
                                            title=title,
                                            logfunction=embedder.log,
                                            write_plot=True,
                                            verbose_print=True)

    e1 = ts_energy - reag_energy
    e2 = ts_energy - prod_energy

    embedder.log(f'NEB completed, relative energy from start/end points (not barrier heights):\n'
               f'  > E(TS)-E(start): {"+" if e1>=0 else "-"}{round(e1, 3)} kcal/mol\n'
               f'  > E(TS)-E(end)  : {"+" if e2>=0 else "-"}{round(e2, 3)} kcal/mol')

    if not (e1 > 0 and e2 > 0):
        embedder.log(f'\nNEB failed, TS energy is lower than both the start and end points.\n')

    with open('Me_CONMe2_Mal_tetr_int_NEB_NEB_TS.xyz', 'w') as f:
        write_xyz(ts_coords, data.atomnos, f, title='NEB TS - see log for relative energies')

def scan_operator(filename, embedder):
    '''
    '''
    embedder.t_start_run = time.perf_counter()
    mol = embedder.objects[0]
    assert len(mol.atomcoords) == 1, 'The scan> operator works on a single .xyz geometry.'
    assert len(mol.reactive_indexes) == 2, 'The scan> operator needs two reactive indexes ' + (
                                          f'({len(mol.reactive_indexes)} were provided)')

    import matplotlib.pyplot as plt

    from tscode.algebra import norm_of
    from tscode.ase_manipulations import ase_popt
    from tscode.pt import pt

    i1, i2 = mol.reactive_indexes
    coords = mol.atomcoords[0]
    # shorthands for clearer code

    embedder.log(f'--> Performing a distance scan approaching on indexes {i1} ' +
                 f'and {i2}.\nTheory level is {embedder.options.theory_level} ' +
                 f'via {embedder.options.calculator}')

    d = norm_of(coords[i1]-coords[i2])
    # getting the start distance between scan indexes and start energy

    dists, energies, structures = [], [], []
    # creating a dictionary that will hold results
    # and the structure output list

    step = -0.05
    # defining the step magnitude, in Angstroms

    s1, s2 = mol.atomnos[[i1, i2]]
    smallest_d = 0.8*(pt[s1].covalent_radius+
                      pt[s2].covalent_radius)
    max_iterations = round((d-smallest_d) / abs(step))
    # defining the maximum number of iterations,
    # so that atoms are never forced closer than
    # a proportionally small distance between those two atoms.

    for i in range(max_iterations):

        coords, energy, _ = ase_popt(embedder,
                                     coords,
                                     mol.atomnos,
                                     constrained_indexes=np.array([mol.reactive_indexes]),
                                     targets=(d,),
                                     title=f'Step {i+1}/{max_iterations} - d={round(d, 2)} A -',
                                     logfunction=embedder.log,
                                     traj=f'{mol.title}_scanpoint_{i+1}.traj' if embedder.options.debug else None,
                                     )
        # optimizing the structure with a spring constraint


        if i == 0:
            e_0 = energy

        energies.append(energy - e_0)
        dists.append(d)
        structures.append(coords)
        # saving the structure, distance and relative energy

        d += step
        # modify the target distance and reiterate

    ### Start the plotting sequence

    plt.figure()
    plt.plot(
        dists,
        energies,
        color='tab:red',
        label='Scan energy',
        linewidth=3,
    )

    # e_max = max(energies)
    id_max = get_scan_peak_index(energies)
    e_max = energies[id_max]

    # id_max = energies.index(e_max)
    d_opt = dists[id_max]

    plt.plot(
        d_opt,
        e_max,
        color='gold',
        label='Energy maximum (TS guess)',
        marker='o',
        markersize=3,
    )

    title = mol.name + ' distance scan'
    plt.legend()
    plt.title(title)
    plt.xlabel(f'Indexes {i1}-{i2} distance (A)')
    plt.gca().invert_xaxis()
    plt.ylabel('Rel. E. (kcal/mol)')
    plt.savefig(f'{title.replace(" ", "_")}_plt.svg')

    ### Start structure writing 

    with open(f'{mol.name[:-4]}_scan.xyz', 'w') as f:
        for i, (s, d, e) in enumerate(zip(structures, dists, energies)):
            write_xyz(s, mol.atomnos, f, title=f'Scan point {i+1}/{len(structures)} ' +
                      f'- d({i1}-{i2}) = {round(d, 3)} A - Rel. E = {round(e, 3)} kcal/mol')
    # print all scan structures

    with open(f'{mol.name[:-4]}_scan_max.xyz', 'w') as f:
        s = structures[id_max]
        d = dists[id_max]
        write_xyz(s, mol.atomnos, f, title=f'Scan point {id_max+1}/{len(structures)} ' +
                    f'- d({i1}-{i2}) = {round(d, 3)} A - Rel. E = {round(e_max, 3)} kcal/mol')
    # print the maximum on another file for convienience

    embedder.log(f'\n--> Written {len(structures)} structures to {mol.name[:-4]}_scan.xyz')
    embedder.log(f'\n--> Written energy maximum to {mol.name[:-4]}_scan_max.xyz')

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

def _get_lowest_calc(embedder=None):
    '''
    Returns the values for calculator,
    method and processors for the lowest
    theory level available from embedder or settings.
    '''
    if embedder is None:
        if FF_OPT_BOOL:
            return (FF_CALC, DEFAULT_FF_LEVELS[FF_CALC], None)
        return (CALCULATOR, DEFAULT_LEVELS[CALCULATOR], PROCS)

    if embedder.options.ff_opt:
        return (embedder.options.ff_calc, embedder.options.ff_level, None)
    return (embedder.options.calculator, embedder.options.theory_level, embedder.options.procs)