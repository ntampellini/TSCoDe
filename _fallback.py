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

# This library was intentionally meant to be cythonized, but 
# for the moment it is kept in pure Python.

import numpy as np
from math import sqrt
import networkx as nx
from rmsd import kabsch_rmsd
from spyrmsd.rmsd import symmrmsd

def norm_of(v):
    s = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    norm = sqrt(s)
    return norm

def compenetration_check(coords, ids, thresh=1.5, max_clashes=2):

    clashes = 0
    # max_clashes clashes is good, max_clashes + 1 is not

    if len(ids) == 2:
        m1 = coords[0:ids[0]]
        m2 = coords[ids[0]:]
        for v1 in m1:
            for v2 in m2:
                dist = norm_of(v1-v2)
                if dist < thresh:
                    clashes += 1
                if clashes > max_clashes:
                    return False
        return True

    else:
        m1 = coords[0:ids[0]]
        m2 = coords[ids[0]:ids[0]+ids[1]]
        m3 = coords[ids[0]+ids[1]:]

        for v1 in m1:
            for v2 in m2:
                dist = norm_of(v1-v2)
                if dist < thresh:
                    clashes += 1
                if clashes > max_clashes:
                    return False

        for v2 in m2:
            for v3 in m3:
                dist = norm_of(v2-v3)
                if dist < thresh:
                    clashes += 1
                if clashes > max_clashes:
                    return False

        for v3 in m3:
            for v1 in m1:
                dist = norm_of(v3-v1)
                if dist < thresh:
                    clashes += 1
                if clashes > max_clashes:
                    return False

        return True

def scramble_mask(array, sequence):
    return np.array([array[s] for s in sequence])

def scramble(array, sequence):
    return np.array([array[s] for s in sequence])

def prune_conformers(structures, k = 1, max_rmsd = 1.):

    if k != 1:

        r = np.arange(structures.shape[0])
        sequence = np.random.permutation(r)
        inv_sequence = np.array([np.where(sequence == i)[0][0] for i in r], dtype=int)

        structures = scramble(structures, sequence)
        # energies = scramble_mask(energies, sequence)
        # scrambling array before splitting, so to improve efficiency when doing
        # multiple runs of group pruning

    mask_out = []
    d = len(structures) // k

    for step in range(k):
        if step == k-1:
            structures_subset = structures[d*step:]
            # energies_subset = energies[d*step:]
        else:
            structures_subset = structures[d*step:d*(step+1)]
            # energies_subset = energies[d*step:d*(step+1)]

        l = structures_subset.shape[0]
        rmsd_mat = np.zeros((l, l))
        rmsd_mat[:] = max_rmsd

        # t0 = time()

        for i in range(l):
            for j in range(i+1,l):
                # val = rmsd_py(structures_subset[i], structures_subset[j])
                val = kabsch_rmsd(structures_subset[i], structures_subset[j], translate=True)
                rmsd_mat[i, j] = val
                if val < max_rmsd:
                    break


        # t1 = time()

        where = np.where(rmsd_mat < max_rmsd)
        matches = [(i,j) for i,j in zip(where[0], where[1])]

        g = nx.Graph(matches)

        subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
        groups = [tuple(graph.nodes) for graph in subgraphs]

        best_of_cluster = [group[0] for group in groups]
        # re-do with energies?

        rejects_sets = [set(a) - {b} for a, b in zip(groups, best_of_cluster)]
        rejects = []
        for s in rejects_sets:
            for i in s:
                rejects.append(i)

        mask = np.array([True for _ in range(l)], dtype=bool)
        for i in rejects:
            mask[i] = False

        mask_out.append(mask)
    
    mask = np.concatenate(mask_out)

    if k != 1:
        mask = scramble_mask(mask, inv_sequence)
        structures = scramble(structures, inv_sequence)
        # undoing the previous shuffling, therefore preserving the input order

    return structures[mask], mask

def prune_conformers_symmetry(structures, atomnos, adjacency_matrix, k = 1, max_rmsd = 1.):

    if k != 1:

        r = np.arange(structures.shape[0])
        sequence = np.random.permutation(r)
        inv_sequence = np.array([np.where(sequence == i)[0][0] for i in r], dtype=int)

        structures = scramble(structures, sequence)
        # energies = scramble_mask(energies, sequence)
        # scrambling array before splitting, so to improve efficiency when doing
        # multiple runs of group pruning

    mask_out = []
    d = len(structures) // k

    for step in range(k):
        if step == k-1:
            structures_subset = structures[d*step:]
            # energies_subset = energies[d*step:]
        else:
            structures_subset = structures[d*step:d*(step+1)]
            # energies_subset = energies[d*step:d*(step+1)]

        l = structures_subset.shape[0]
        rmsd_mat = np.zeros((l, l))
        rmsd_mat[:] = max_rmsd

        # t0 = time()

        for i in range(l):
            for j in range(i+1,l):
                val = symmrmsd(structures_subset[i],
                               structures_subset[j],
                               atomnos, atomnos,
                               adjacency_matrix,
                               adjacency_matrix,
                               center=True, minimize=True)
                
                rmsd_mat[i, j] = val
                if val < max_rmsd:
                    break


        # t1 = time()

        where = np.where(rmsd_mat < max_rmsd)
        matches = [(i,j) for i,j in zip(where[0], where[1])]

        g = nx.Graph(matches)

        subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
        groups = [tuple(graph.nodes) for graph in subgraphs]

        best_of_cluster = [group[0] for group in groups]
        # re-do with energies?

        rejects_sets = [set(a) - {b} for a, b in zip(groups, best_of_cluster)]
        rejects = []
        for s in rejects_sets:
            for i in s:
                rejects.append(i)

        mask = np.array([True for _ in range(l)], dtype=bool)
        for i in rejects:
            mask[i] = False

        mask_out.append(mask)
    
    mask = np.concatenate(mask_out)

    if k != 1:
        mask = scramble_mask(mask, inv_sequence)
        structures = scramble(structures, inv_sequence)
        # undoing the previous shuffling, therefore preserving the input order

    # t2 = time()
    # print(f'First step: {round(t1-t0, 2)} s\nSecond step: {round(t2-t1, 2)} s')
    # sys.stdout.flush()

    return structures[mask], mask
