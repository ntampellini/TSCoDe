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

import time
from copy import deepcopy

import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation as R

from ase_manipulations import ase_neb, ase_popt
from calculators._gaussian import gaussian_opt
from calculators._mopac import mopac_opt
from calculators._orca import orca_opt
from calculators._xtb import xtb_opt
from utils import (center_of_mass, diagonalize, kronecker_delta,
                   molecule_check, norm, pt, scramble_check, time_to_string,
                   write_xyz)

opt_funcs_dict = {
    'MOPAC':mopac_opt,
    'ORCA':orca_opt,
    'GAUSSIAN':gaussian_opt,    
    'XTB':xtb_opt,
}

def optimize(calculator, coords, atomnos,  method, constrained_indexes=None, mols_graphs=None, procs=1, solvent=None, max_newbonds=0, title='temp', check=True):
    '''
    Performs a geometry [partial] optimization (OPT/POPT) with MOPAC, ORCA, Gaussian or XTB at $method level, 
    constraining the distance between the specified atom pairs, if any. Moreover, if $check, performs a check on atomic
    pairs distances to ensure that the optimization has preserved molecular identities and no atom scrambling occurred.

    :params calculator: Calculator to be used. ('MOPAC', 'ORCA', 'GAUSSIAN', 'XTB')
    :params coords: list of coordinates for each atom in the TS
    :params atomnos: list of atomic numbers for each atom in the TS
    :params mols_graphs: list of molecule.graph objects, containing connectivity information for each molecule
    :params constrained_indexes: indexes of constrained atoms in the TS geometry, if this is one
    :params method: Level of theory to be used in geometry optimization. Default if UFF.

    :return opt_coords: optimized structure
    :return energy: absolute energy of structure, in kcal/mol
    :return not_scrambled: bool, indicating if the optimization shifted up some bonds (except the constrained ones)
    '''
    if mols_graphs is not None:
        assert len(coords) == sum([len(graph.nodes) for graph in mols_graphs])

    constrained_indexes = np.array(()) if constrained_indexes is None else constrained_indexes

    opt_func = opt_funcs_dict[calculator]

    opt_coords, energy, success = opt_func(coords,
                                            atomnos,
                                            constrained_indexes,
                                            method=method,
                                            procs=procs,
                                            solvent=solvent,
                                            title=title)
    # success checks that calculation had a normal termination

    if success and check:
        # check boolean ensures that no scrambling occurred during the optimization
        if mols_graphs is not None:
            success = scramble_check(opt_coords, atomnos, constrained_indexes, mols_graphs, max_newbonds=max_newbonds)
        else:
            success = molecule_check(coords, opt_coords, atomnos, max_newbonds=max_newbonds)

    return opt_coords, energy, success

def hyperNEB(docker, coords, atomnos, ids, constrained_indexes, title='temp'):
    '''
    Turn a geometry close to TS to a proper TS by getting
    reagents and products and running a climbing image NEB calculation through ASE.
    '''

    reagents = get_reagent(docker, coords, atomnos, ids, constrained_indexes, method=docker.options.theory_level)
    products = get_product(docker, coords, atomnos, ids, constrained_indexes, method=docker.options.theory_level)
    # get reagents and products for this reaction

    reagents -= np.mean(reagents, axis=0)
    products -= np.mean(products, axis=0)
    # centering both structures on the centroid of reactive atoms

    aligment_rotation = R.align_vectors(reagents, products)
    products = np.array([aligment_rotation @ v for v in products])
    # rotating the two structures to minimize differences

    ts_coords, ts_energy, success = ase_neb(docker, reagents, products, atomnos, title=title)
    # Use these structures plus the TS guess to run a NEB calculation through ASE

    return ts_coords, ts_energy, success

def get_product(docker, coords, atomnos, ids, constrained_indexes, method='PM7'):
    '''
    Part of the automatic NEB implementation.
    Returns a structure that presumably is the association reaction product
    ([cyclo]additions reactions in mind)
    '''

    opt_func = opt_funcs_dict[docker.options.calculator]

    bond_factor = 1.2
    # multiple of sum of covalent radii for two atoms.
    # If two atoms are closer than this times their sum
    # of c_radii, they are considered to converge to
    # products when their geometry is optimized. 

    step_size = 0.1
    # in Angstroms

    if len(ids) == 2:

        mol1_center = np.mean([coords[a] for a, _ in constrained_indexes], axis=0)
        mol2_center = np.mean([coords[b] for _, b in constrained_indexes], axis=0)
        motion = norm(mol2_center - mol1_center)
        # norm of the motion that, when applied to mol1,
        # superimposes its reactive atoms to the ones of mol2

        threshold_dists = [bond_factor*(pt[atomnos[a]].covalent_radius +
                                        pt[atomnos[b]].covalent_radius) for a, b in constrained_indexes]

        reactive_dists = [np.linalg.norm(coords[a] - coords[b]) for a, b in constrained_indexes]
        # distances between reactive atoms

        while not np.all([reactive_dists[i] < threshold_dists[i] for i, _ in enumerate(constrained_indexes)]):
            # print('Reactive distances are', reactive_dists)

            coords[:ids[0]] += motion*step_size

            coords, _, _ = opt_func(coords, atomnos, constrained_indexes, method=method)

            reactive_dists = [np.linalg.norm(coords[a] - coords[b]) for a, b in constrained_indexes]

        newcoords, _, _ = opt_func(coords, atomnos, method=method)
        # finally, when structures are close enough, do a free optimization to get the reaction product

        new_reactive_dists = [np.linalg.norm(newcoords[a] - newcoords[b]) for a, b in constrained_indexes]

        if np.all([new_reactive_dists[i] < threshold_dists[i] for i, _ in enumerate(constrained_indexes)]):
        # return the freely optimized structure only if the reagents did not repel each other
        # during the optimization, otherwise return the last coords, where partners were close
            return newcoords

        return coords

    # trimolecular TSs: the approach is to bring the first pair of reactive
    # atoms closer until optimization bounds the molecules together

    index_to_be_moved = constrained_indexes[0,0]
    reference = constrained_indexes[0,1]
    moving_molecule_index = next(i for i,n in enumerate(np.cumsum(ids)) if index_to_be_moved < n)
    bounds = [0] + [n+1 for n in np.cumsum(ids)]
    moving_molecule_slice = slice(bounds[moving_molecule_index], bounds[moving_molecule_index+1])
    threshold_dist = bond_factor*(pt[atomnos[constrained_indexes[0,0]]].covalent_radius +
                                    pt[atomnos[constrained_indexes[0,1]]].covalent_radius)

    motion = (coords[reference] - coords[index_to_be_moved])
    # vector from the atom to be moved to the target reactive atom

    while np.linalg.norm(motion) > threshold_dist:
    # check if the reactive atoms are sufficiently close to converge to products

        for i, atom in enumerate(coords[moving_molecule_slice]):
            dist = np.linalg.norm(atom - coords[index_to_be_moved])
            # for any atom in the molecule, distance from the reactive atom

            atom_step = step_size*np.exp(-0.5*dist)
            coords[moving_molecule_slice][i] += norm(motion)*atom_step
            # the more they are close, the more they are moved

        # print('Reactive dist -', np.linalg.norm(motion))
        coords, _, _ = opt_func(coords, atomnos, constrained_indexes, method=method)
        # when all atoms are moved, optimize the geometry with the previous constraints

        motion = (coords[reference] - coords[index_to_be_moved])

    newcoords, _, _ = opt_func(coords, atomnos, method=method)
    # finally, when structures are close enough, do a free optimization to get the reaction product

    new_reactive_dist = np.linalg.norm(newcoords[constrained_indexes[0,0]] - newcoords[constrained_indexes[0,0]])

    if new_reactive_dist < threshold_dist:
    # return the freely optimized structure only if the reagents did not repel each other
    # during the optimization, otherwise return the last coords, where partners were close
        return newcoords

    return coords

def get_reagent(docker, coords, atomnos, ids, constrained_indexes, method='PM7'):
    '''
    Part of the automatic NEB implementation.
    Returns a structure that presumably is the association reaction reagent.
    ([cyclo]additions reactions in mind)
    '''

    opt_func = opt_funcs_dict[docker.options.calculator]

    bond_factor = 1.5
    # multiple of sum of covalent radii for two atoms.
    # Putting reactive atoms at this times their bonding
    # distance and performing a constrained optimization
    # is the way to get a good guess for reagents structure. 

    if len(ids) == 2:

        mol1_center = np.mean([coords[a] for a, _ in constrained_indexes], axis=0)
        mol2_center = np.mean([coords[b] for _, b in constrained_indexes], axis=0)
        motion = norm(mol2_center - mol1_center)
        # norm of the motion that, when applied to mol1,
        # superimposes its reactive centers to the ones of mol2

        threshold_dists = [bond_factor*(pt[atomnos[a]].covalent_radius + pt[atomnos[b]].covalent_radius) for a, b in constrained_indexes]

        reactive_dists = [np.linalg.norm(coords[a] - coords[b]) for a, b in constrained_indexes]
        # distances between reactive atoms

        coords[:ids[0]] -= norm(motion)*(np.mean(threshold_dists) - np.mean(reactive_dists))
        # move reactive atoms away from each other just enough

        coords, _, _ = opt_func(coords, atomnos, constrained_indexes=constrained_indexes, method=method)
        # optimize the structure but keeping the reactive atoms distanced

        return coords

    # trimolecular TSs: the approach is to bring the first pair of reactive
    # atoms apart just enough to get a good approximation for reagents

    index_to_be_moved = constrained_indexes[0,0]
    reference = constrained_indexes[0,1]
    moving_molecule_index = next(i for i,n in enumerate(np.cumsum(ids)) if index_to_be_moved < n)
    bounds = [0] + [n+1 for n in np.cumsum(ids)]
    moving_molecule_slice = slice(bounds[moving_molecule_index], bounds[moving_molecule_index+1])
    threshold_dist = bond_factor*(pt[atomnos[constrained_indexes[0,0]]].covalent_radius +
                                    pt[atomnos[constrained_indexes[0,1]]].covalent_radius)

    motion = (coords[reference] - coords[index_to_be_moved])
    # vector from the atom to be moved to the target reactive atom

    displacement = norm(motion)*(threshold_dist-np.linalg.norm(motion))
    # vector to be applied to the reactive atom to push it far just enough

    for i, atom in enumerate(coords[moving_molecule_slice]):
        dist = np.linalg.norm(atom - coords[index_to_be_moved])
        # for any atom in the molecule, distance from the reactive atom

        coords[moving_molecule_slice][i] -= displacement*np.exp(-0.5*dist)
        # the closer they are to the reactive atom, the further they are moved

    coords, _, _ = opt_func(coords, atomnos, constrained_indexes=np.array([constrained_indexes[0]]), method=method)
    # when all atoms are moved, optimize the geometry with only the first of the previous constraints

    newcoords, _, _ = opt_func(coords, atomnos, method=method)
    # finally, when structures are close enough, do a free optimization to get the reaction product

    new_reactive_dist = np.linalg.norm(newcoords[constrained_indexes[0,0]] - newcoords[constrained_indexes[0,0]])

    if new_reactive_dist > threshold_dist:
    # return the freely optimized structure only if the reagents did not approached back each other
    # during the optimization, otherwise return the last coords, where partners were further away
        return newcoords
    
    return coords

def get_inertia_moments(coords, atomnos):
    '''
    Returns the diagonal of the diagonalized inertia tensor, that is
    a shape (3,) array with the moments of inertia along the main axes.
    (I_x, I_y and largest I_z last)
    '''

    coords -= center_of_mass(coords, atomnos)
    inertia_moment_matrix = np.zeros((3,3))

    for i in range(3):
        for j in range(3):
            k = kronecker_delta(i,j)
            inertia_moment_matrix[i][j] = np.sum([pt[atomnos[n]].mass*((np.linalg.norm(coords[n])**2)*k - coords[n][i]*coords[n][j]) for n, _ in enumerate(atomnos)])

    inertia_moment_matrix = diagonalize(inertia_moment_matrix)

    return np.diag(inertia_moment_matrix)

def prune_enantiomers(structures, atomnos, max_delta=10):
    '''
    Remove duplicate (enantiomeric) structures based on the
    moments of inertia on principal axes. If all three MOI
    are within max_delta from another structure, they are
    classified as enantiomers and therefore only one of them
    is kept.
    '''

    l = len(structures)
    mat = np.zeros((l,l), dtype=int)
    for i in range(l):
        for j in range(i+1,l):
            im_i = get_inertia_moments(structures[i], atomnos)
            im_j = get_inertia_moments(structures[j], atomnos)
            delta = np.abs(im_i - im_j)
            mat[i,j] = 1 if np.all(delta < max_delta) else 0

    where = np.where(mat == 1)
    matches = [(i,j) for i,j in zip(where[0], where[1])]

    g = nx.Graph(matches)

    subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
    groups = [tuple(graph.nodes) for graph in subgraphs]

    best_of_cluster = [group[0] for group in groups]

    rejects_sets = [set(a) - {b} for a, b in zip(groups, best_of_cluster)]
    rejects = []
    for s in rejects_sets:
        for i in s:
            rejects.append(i)

    mask = np.array([True for _ in range(l)], dtype=bool)
    for i in rejects:
        mask[i] = False

    return structures[mask], mask

def opt_iscans(docker, coords, atomnos, title='temp', logfile=None, xyztraj=None):
    '''
    Runs one or more independent scans along the constrained indexes
    specified, one at a time, through the ASE package. Each scan starts
    from the previous maximum in energy. This is done as a low-dimensional
    but effective approach of exploring the PES trying to maximize the
    energy. The highest energy structure is returned.
    '''

    overall_success = False

    scan_active_indexes = [indexes for letter, indexes in docker.pairings_table.items() if letter not in ('x', 'y', 'z')]
    for i, indexes in enumerate(scan_active_indexes):
        new_coords, energy, success = opt_linear_scan(docker,
                                                    coords,
                                                    atomnos,
                                                    indexes,
                                                    docker.constrained_indexes[0],
                                                    # safe=True,
                                                    title=title+f' scan {i}',
                                                    logfile=logfile,
                                                    xyztraj=xyztraj,
                                                    )

    if success:
        overall_success = True
        coords = new_coords

    else: # Re-try with safe keyword to prevent scrambling

        for i, indexes in enumerate(scan_active_indexes):
            new_coords, energy, success = opt_linear_scan(docker,
                                                        coords,
                                                        atomnos,
                                                        indexes,
                                                        docker.constrained_indexes[0],
                                                        safe=True,
                                                        title=title+f' scan {i}',
                                                        logfile=logfile,
                                                        xyztraj=xyztraj,
                                                        )
                                                        
        if success:
            overall_success = True
            coords = new_coords

    return coords, energy, overall_success

def opt_linear_scan(docker, coords, atomnos, scan_indexes, constrained_indexes, step_size=0.02, safe=False, title='temp', logfile=None, xyztraj=None):
    '''
    Runs a linear scan along the specified linear coordinate.
    The highest energy structure that passes sanity checks is returned.

    docker
    coords
    atomnos
    scan_indexes
    constrained_indexes
    step_size
    safe
    title
    logfile
    xyztraj
    '''
    assert [i in constrained_indexes.ravel() for i in scan_indexes]

    i1, i2 = scan_indexes
    far_thr = 2 * sum([pt[atomnos[i]].covalent_radius for i in scan_indexes])
    t_start = time.time()
    total_iter = 0

    _, energy, _ = optimize(docker.options.calculator,
                            coords,
                            atomnos,
                            docker.options.theory_level,
                            constrained_indexes=constrained_indexes,
                            mols_graphs=docker.graphs,
                            procs=docker.options.procs,
                            max_newbonds=docker.options.max_newbonds,
                            )

    direction = coords[i1] - coords[i2]
    base_dist = np.linalg.norm(direction)
    energies, geometries = [energy], [coords]

    for sign in (1, -1):
    # getting closer for sign == 1, further apart for -1
        active_coords = deepcopy(coords)
        dist = base_dist

        if scan_peak_present(energies):
            break

        for iterations in range(75):
            
            if safe: # use ASE optimization function - more reliable, but locks all interatomic dists

                targets = [np.linalg.norm(active_coords[a]-active_coords[b]) - step_size
                           if (a in scan_indexes and b in scan_indexes)
                           else np.linalg.norm(active_coords[a]-active_coords[b])
                           for a, b in constrained_indexes]

                active_coords, energy, success = ase_popt(docker,
                                                            active_coords,
                                                            atomnos,
                                                            constrained_indexes,
                                                            targets=targets,
                                                            safe=True,
                                                            )

            else: # use faster raw optimization function, might scramble more often than the ASE one

                active_coords[i2] += sign * norm(direction) * step_size
                active_coords, energy, success = optimize(docker.options.calculator,
                                                            active_coords,
                                                            atomnos,
                                                            docker.options.theory_level,
                                                            constrained_indexes=constrained_indexes,
                                                            mols_graphs=docker.graphs,
                                                            procs=docker.options.procs,
                                                            max_newbonds=docker.options.max_newbonds,
                                                            )

            if not success:
                if logfile is not None and iterations == 0:
                    logfile.write(f'    - {title} CRASHED at first step\n')

                if docker.options.debug:
                    with open(title+'_SCRAMBLED.xyz', 'a') as f:
                        write_xyz(active_coords, atomnos, f, title=title+(
                            f' d({i1}-{i2}) = {round(dist, 3)} A, Rel. E = {round(energy-energies[0], 3)} kcal/mol'))

                break
            
            direction = active_coords[i1] - active_coords[i2]
            dist = np.linalg.norm(direction)

            total_iter += 1
            geometries.append(active_coords)
            energies.append(energy)

            if xyztraj is not None:
                with open(xyztraj, 'a') as f:
                    write_xyz(active_coords, atomnos, f, title=title+(
                        f' d({i1}-{i2}) = {round(dist, 3)} A, Rel. E = {round(energy-energies[0], 3)} kcal/mol'))

            if (dist < 1.2 and sign == 1) or (
                dist > far_thr and sign == -1) or (
                scan_peak_present(energies)
                ):
                break
            
    distances = [np.linalg.norm(g[i1]-g[i2]) for g in geometries]
    best_distance = distances[energies.index(max(energies))]

    distances_delta = [abs(d-best_distance) for d in distances]
    closest_geom = geometries[distances_delta.index(min(distances_delta))]
    closest_dist = distances[distances_delta.index(min(distances_delta))]

    direction = closest_geom[i1] - closest_geom[i2]
    closest_geom[i1] += norm(direction) * (best_distance-closest_dist)

    final_geom, final_energy, _ = optimize(docker.options.calculator,
                                            closest_geom,
                                            atomnos,
                                            docker.options.theory_level,
                                            constrained_indexes=constrained_indexes,
                                            mols_graphs=docker.graphs,
                                            procs=docker.options.procs,
                                            max_newbonds=docker.options.max_newbonds,
                                            check=False,
                                            )

    if docker.options.debug:

        if docker.options.debug:
            with open(xyztraj, 'a') as f:
                write_xyz(active_coords, atomnos, f, title=title+(
                    f' FINAL - d({i1}-{i2}) = {round(np.linalg.norm(final_geom[i1]-final_geom[i2]), 3)} A,'
                    f' Rel. E = {round(final_energy-energies[0], 3)} kcal/mol'))

        import matplotlib.pyplot as plt

        plt.figure()

        distances = [np.linalg.norm(geom[i1]-geom[i2]) for geom in geometries]
        distances, sorted_energies = zip(*sorted(zip(distances, energies), key=lambda x: x[0]))

        plt.plot(distances,
                [s-energies[0] for s in sorted_energies],
                '-o',
                color='tab:red',
                label=f'Linear SCAN ({i1}-{i2})',
                linewidth=3,
                alpha=0.5)

        plt.plot(np.linalg.norm(coords[i1]-coords[i2]),
                0,
                marker='o',
                color='tab:blue',
                label='Starting point (0 kcal/mol)',
                markersize=5,
                )

        plt.plot(best_distance,
                final_energy-energies[0],
                marker='o',
                color='black',
                label='Interpolated best distance, actual energy',
                markersize=5)

        plt.legend()
        plt.title(title)
        plt.xlabel(f'Interatomic distance {tuple(scan_indexes)}')
        plt.ylabel('Energy Rel. to starting point (kcal/mol)')
        plt.savefig(f'{title.replace(" ", "_")}_plt.svg')

    if logfile is not None:
        logfile.write(f'    - {title} COMPLETED {total_iter} steps ({time_to_string(time.time()-t_start)})\n')

    return final_geom, final_energy, True

def scan_peak_present(energies) -> bool:
    '''
    Returns True if the maximum value of the list
    occurs in the middle of it, that is not in first,
    second, second to last or last positions
    '''
    if energies.index(max(energies)) in range(2,len(energies)-1):
        return True
    return False

def fitness_check(docker, coords) -> bool:
    '''
    Returns True if the strucure respects
    the imposed pairings.
    '''
    if hasattr(docker, 'pairings_dists'):
        for letter, pairing in docker.pairings_table.items():
            
            if letter in ('a', 'b', 'c'):
                i1, i2 = pairing
                dist = np.linalg.norm(coords[i1]-coords[i2])
                target = docker.pairings_dists.get(letter)

                if target is not None and abs(target-dist) > 0.05:
                    return False

            else:
                if dist < 2.5:
                    return False
                    
    return True