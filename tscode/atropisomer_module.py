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

from time import time

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from ase.constraints import FixInternals
from networkx.algorithms.components.connected import connected_components
from networkx.algorithms.shortest_paths.generic import shortest_path

from hypermolecule_class import graphize, align_structures
from ase_manipulations import get_ase_calc, ase_saddle
from utils import (
    dihedral,
    clean_directory,
    molecule_check,
    loadbar,
    time_to_string,
    write_xyz,
)

def ase_torsion_TSs(coords,
                    atomnos,
                    indexes,
                    calculator,
                    method,
                    procs=1,
                    threshold_kcal=5,
                    title='temp',
                    optimization=True,
                    logfile=None,
                    bernytraj=None,
                    plot=False):
    '''
    '''
    
    assert len(indexes) == 4
    cyclical = False
    
    ts_structures, energies = [], []

    graph = graphize(coords, atomnos)
    i1, i2, i3, i4 = indexes

    if all([len(shortest_path(graph, start, end)) == 2 for start, end in zip(indexes[0:-1], indexes[1:])]):
        graph.remove_edge(i2, i3)
        subgraphs = connected_components(graph)

        for subgraph in subgraphs:
            if i3 in subgraph:
                indexes_to_be_moved = subgraph - {i3}
                break

        if i1 in indexes_to_be_moved:

            cyclical = True
            indexes_to_be_moved = [i4]
            # if molecule is cyclical, just move the fourth atom and
            # let the rest of the structure relax

            s = 'The specified dihedral angle is comprised within a cycle. Two preliminary scans will be conducted.'
            print(s)
            if logfile is not None:
                logfile.write(s+'\n')

    else:
        # if user did not provide four contiguous indexes,
        # just move the fourth atom and
        # let the rest of the structure relax
        indexes_to_be_moved = [i4]
        cyclical = True

        s = 'The specified dihedral angle is made up of non-contiguous atoms.\nThis might cause some unexpected results. Two preliminary scans will be conducted.'
        print(s)
        if logfile is not None:
            logfile.write(s+'\n')


    routine = ((10, 18, '_clockwise'), (-10, 18, '_counterclockwise')) if cyclical else ((10, 36, ''),)

    for degrees, steps, direction in routine:

        print()
        if logfile is not None:
            logfile.write('\n')

        structures, energies = ase_scan(coords,
                                        atomnos,
                                        calculator=calculator,
                                        method=method,
                                        indexes=indexes,
                                        degrees=degrees,
                                        steps=steps,
                                        relaxed=optimization,
                                        indexes_to_be_moved=indexes_to_be_moved,
                                        title='Preliminary scan' + ((' (clockwise)' if direction == '_clockwise' \
                                              else ' (counterclockwise)') if direction != '' else ''),
                                        procs=procs,
                                        logfile=logfile)

        min_e = min(energies)
        output_structures, output_energies = [], []
        
        for structure, energy in zip(structures, energies):
            output_structures.append(structure)
            output_energies.append(energy)

        if plot:
            import matplotlib.pyplot as plt
            import pickle

            fig = plt.figure()

            x1 = [dihedral(structures[i][indexes]) for i in range(steps)]
            y1 = [e-min_e for e in energies]

            x1, y1 = zip(*sorted(zip(x1, y1), key=lambda x: x[0]))

            plt.plot(x1,
                    y1,
                    '-',
                    color='tab:blue',
                    label='Preliminary SCAN'+direction,
                    linewidth=3,
                    alpha=0.50)

        peaks_indexes = peaks(energies, min_thr=min_e+threshold_kcal, max_thr=min_e+75)

        if peaks_indexes:

            s = 's' if len(peaks_indexes) > 1 else ''
            print(f'Found {len(peaks_indexes)} peak{s}. Performing accurate scan{s}.\n')
            if logfile is not None:
                logfile.write(f'Found {len(peaks_indexes)} peak{s}. Performing accurate scan{s}.\n\n')


            for p, peak in enumerate(peaks_indexes):

                sub_structures, sub_energies = ase_scan(structures[peak-1],
                                                            atomnos,
                                                            calculator=calculator,
                                                            method=method,
                                                            indexes=indexes,
                                                            degrees=1,
                                                            steps=20,
                                                            relaxed=optimization,
                                                            ad_libitum=True, # goes on until the hill is crossed
                                                            indexes_to_be_moved=indexes_to_be_moved,
                                                            procs=procs,
                                                            title=f'Accurate scan {p+1}/{len(peaks_indexes)}',
                                                            logfile=logfile)

                if logfile is not None:
                    logfile.write('\n')

                # for sub_structure, sub_energy in zip(sub_structures, sub_energies):
                #     output_structures.append(sub_structure)
                #     output_energies.append(sub_energy)

                # This was a tentative to include accurate scan structures into the
                # final scan file, but yields rough results and i do not like it

                if plot:
                    x2 = [dihedral(structure[indexes]) for structure in sub_structures]
                    y2 = [e-min_e for e in sub_energies]

                    x2, y2 = zip(*sorted(zip(x2, y2), key=lambda x: x[0]))

                    plt.plot(x2, 
                            y2,
                            '-',
                            color='tab:red',
                            label='Accurate SCAN' if p == 0 else None,
                            linewidth=2,
                            alpha=0.75)

                sub_peaks_indexes = peaks(sub_energies, min_thr=threshold_kcal+min_e, max_thr=min_e+75)

                if sub_peaks_indexes:

                    s = 's' if len(sub_peaks_indexes) > 1 else ''
                    print(f'Found {len(sub_peaks_indexes)} sub-peak{s}. Performing Saddle opt optimization{s}.')
                    if logfile is not None:
                        logfile.write(f'Found {len(sub_peaks_indexes)} sub-peak{s}. Performing Saddle opt optimization{s}.\n')

                    for s, sub_peak in enumerate(sub_peaks_indexes):

                        if plot:
                            x = dihedral(sub_structures[sub_peak][indexes])
                            y = sub_energies[sub_peak]-min_e
                            plt.plot(x, y, color='gold', marker='o', label='Maxima' if p == 0 else None, markersize=3)

                        if optimization:

                            loadbar_title = f'  > Saddle opt on sub-peak {s+1}/{len(sub_peaks_indexes)}'
                            # loadbar(s+1, len(sub_peaks_indexes), loadbar_title+' '*(29-len(loadbar_title)))
                            print(loadbar_title)
                        
                            optimized_geom, energy, _ = ase_saddle(sub_structures[sub_peak],
                                                                    atomnos,
                                                                    calculator=calculator,
                                                                    method=method,
                                                                    procs=procs,
                                                                    title=f'Saddle opt - peak {p+1}, sub-peak {s+1}',
                                                                    logfile=logfile,
                                                                    traj=bernytraj+f'_{p+1}_{s+1}.traj' if bernytraj is not None else None)

                            if molecule_check(coords, optimized_geom, atomnos, max_newbonds=3):
                                ts_structures.append(optimized_geom)
                                energies.append(energy)

                        else:
                            ts_structures.append(sub_structures[sub_peak])
                            energies.append(sub_energies[sub_peak])

                    print()
            
                else:
                    print('No suitable sub-peaks found.\n')
                    if logfile is not None:
                        logfile.write('No suitable sub-peaks found.\n\n')
        else:
            print('No suitable peaks found.\n')
            if logfile is not None:
                logfile.write('No suitable peaks found.\n\n')

        # output_structures, output_energies = zip(*sorted(zip(output_structures, output_energies), key=lambda x: dihedral(x[0][indexes])))
        # Sorting structures and energies based on scanned dihedral
        # angle value, to get a continuous scan in the .xyz file

        rel_energies = [e-min_e for e in output_energies]

        tag = '_relaxed' if optimization else '_rigid'

        with open(title + tag + direction + '_scan.xyz', 'w') as outfile:
            for s, structure in enumerate(align_structures(np.array(output_structures), indexes[:-1])):
                write_xyz(structure, atomnos, outfile, title=f'Scan point {s+1}/{len(output_structures)} - Rel. E = {round(rel_energies[s], 3)} kcal/mol')

        if plot:
            plt.legend()
            plt.xlabel(f'Dihedral Angle {tuple(indexes)}')
            plt.ylabel('Energy (kcal/mol)')
            pickle.dump(fig, open(f'{title}{direction}_plt.pickle', 'wb'))
            plt.savefig(f'{title}{direction}_plt.svg')

    ts_structures = np.array(ts_structures)

    clean_directory()

    return ts_structures, energies

def peaks(data, min_thr, max_thr):
    '''
    data: iterable
    threshold: peaks must be greater than threshold
    cap: peaks must be less than cap
    return: list of peak indexes
    '''
    l = len(data)
    peaks = [i for i in range(l) if (

        data[i-1] < data[i] >= data[(i+1)%l] and
        max_thr > data[i] > min_thr # discard peaks that are too small or too big
    )]

    return peaks
    
def ase_scan(coords,
             atomnos,
             calculator,
             method,
             indexes,
             degrees=10,
             steps=36,
             relaxed=True,
             ad_libitum=False,
             indexes_to_be_moved=None,
             procs=1,
             title='temp scan',
             logfile=None):
    '''
    if ad libitum, steps is the minimum number of performed steps
    '''
    assert len(indexes) == 4

    if ad_libitum:
        if not relaxed:
            raise Exception(f'The ad_libitum keyword is only available for relaxed scans.')

    atoms = Atoms(atomnos, positions=coords)
    structures, energies = [], []

    atoms.calc = get_ase_calc(calculator, procs, method)

    if indexes_to_be_moved is None:
        indexes_to_be_moved = range(len(atomnos))

    mask = np.array([i in indexes_to_be_moved for i, _ in enumerate(atomnos)], dtype=bool)

    t_start = time()

    if logfile is not None:
        logfile.write(f'  > {title}\n')

    for scan_step in range(1000):

        loadbar_title = f'{title} - step {scan_step+1}'
        if ad_libitum:
            print(loadbar_title, end='\r')
        else:
            loadbar_title += '/'+str(steps)
            loadbar(scan_step+1, steps, loadbar_title+' '*(29-len(loadbar_title)))

        if logfile is not None:
            t_start_step = time()

        if relaxed:
            atoms.set_constraint(FixInternals(dihedrals_deg=[[atoms.get_dihedral(*indexes), indexes]]))
            
            with LBFGS(atoms, maxstep=0.2, logfile=None, trajectory=None) as opt:
                
                try:
                    opt.run(fmax=0.05, steps=500)
                    exit_str = 'converged'

                except ValueError: # Shake did not converge
                    exit_str = 'crashed'

                iterations = opt.nsteps


            energies.append(atoms.get_total_energy() * 23.06054194532933) # eV to kcal/mol

        if logfile is not None:
            elapsed = time() - t_start_step
            s = '/' + str(steps) if not ad_libitum else ''
            logfile.write(f'        Step {scan_step+1}{s} - {exit_str} - {iterations} iterations ({time_to_string(elapsed)})\n')

        structures.append(atoms.get_positions())

        atoms.rotate_dihedral(*indexes, angle=degrees, mask=mask)

        if exit_str == 'crashed':
            break

        elif scan_step+1 >= steps:
            if ad_libitum:
                if any((
                    (max(energies) - energies[-1]) > 1,
                    (max(energies) - energies[-1]) > max(energies)-energies[0],
                    (energies[-1] - min(energies)) > 50
                )):

                    # ad_libitum stops when one of these conditions is met:
                    # - we surpassed and are below the maximum of at least 1 kcal/mol
                    # - we surpassed maximum and are below starting point
                    # - current step energy is more than 50 kcal/mol above starting point

                    print(loadbar_title)
                    break
            else:
                break

    structures = np.array(structures)

    clean_directory()

    if logfile is not None:
        elapsed = time() - t_start
        logfile.write(f'{title} - completed ({time_to_string(elapsed)})\n')

    return align_structures(structures, indexes), energies
