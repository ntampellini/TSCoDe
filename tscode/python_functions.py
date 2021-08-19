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

# This library was intentionally meant to be cythonized, but 
# for the moment it is kept in pure Python.

import numpy as np
import networkx as nx
from rmsd import kabsch_rmsd
from scipy.spatial.distance import cdist
# from spyrmsd.rmsd import symmrmsd

def compenetration_check(coords, ids, thresh=1.3, max_clashes=0):

    clashes = 0
    # max_clashes clashes is good, max_clashes + 1 is not

    if len(ids) == 2:
        m1 = coords[0:ids[0]]
        m2 = coords[ids[0]:]
        return False if np.count_nonzero(cdist(m2,m1) < thresh) > max_clashes else True

    # if len(ids) == 3:

    m1 = coords[0:ids[0]]
    m2 = coords[ids[0]:ids[0]+ids[1]]
    m3 = coords[ids[0]+ids[1]:]

    clashes += np.count_nonzero(cdist(m2,m1) < thresh)
    if clashes > max_clashes:
        return False

    clashes += np.count_nonzero(cdist(m3,m2) < thresh)
    if clashes > max_clashes:
        return False

    clashes += np.count_nonzero(cdist(m1,m3) < thresh)
    if clashes > max_clashes:
        return False

    return True

def scramble(array, sequence):
    return np.array([array[s] for s in sequence])

def prune_conformers(structures, atomnos, k = 1, max_rmsd = 1.):

    heavy_atoms = (atomnos != 1)
    heavy_structures = np.array([structure[heavy_atoms] for structure in structures])

    if k != 1:

        r = np.arange(structures.shape[0])
        sequence = np.random.permutation(r)
        inv_sequence = np.array([np.where(sequence == i)[0][0] for i in r], dtype=int)

        heavy_structures = scramble(heavy_structures, sequence)
        # scrambling array before splitting, to improve efficiency when doing
        # multiple runs of group pruning

    mask_out = []
    d = len(structures) // k

    for step in range(k):
        if step == k-1:
            structures_subset = heavy_structures[d*step:]
            # energies_subset = energies[d*step:]
        else:
            structures_subset = heavy_structures[d*step:d*(step+1)]
            # energies_subset = energies[d*step:d*(step+1)]

        l = structures_subset.shape[0]
        rmsd_mat = np.zeros((l, l))
        rmsd_mat[:] = max_rmsd

        # t0 = time()

        for i in range(l):
            for j in range(i+1,l):
                val = kabsch_rmsd(structures_subset[i], structures_subset[j], translate=True)
                rmsd_mat[i, j] = val
                if val < max_rmsd:
                    break


        where = np.where(rmsd_mat < max_rmsd)
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

        mask_out.append(mask)
    
    mask = np.concatenate(mask_out)

    if k != 1:
        mask = scramble(mask, inv_sequence)
        # undoing the previous shuffling, therefore preserving the input order

    return structures[mask], mask