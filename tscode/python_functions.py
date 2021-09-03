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

import networkx as nx
import numpy as np
from rmsd import kabsch_rotate
from scipy.spatial.distance import cdist

# These functions are here to facilitate eventual porting to
# faster precompiled versions of themselves (Cython/C++/Julia/...)
# if the necessity ever occurs

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

def rmsd_and_max(P, Q):
    '''
    ** ADAPTED FROM THE PYTHON RMSD LIBRARY **

    Rotate matrix P unto Q using Kabsch algorithm and calculate the RMSD.
    Returns RMSD and max deviation.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rmsd : float
        root-mean squared deviation
    max_delta : float
        maximum deviation value
    '''

    Q = Q - Q.mean(axis=0)
    P = P - P.mean(axis=0)
    P = kabsch_rotate(P, Q)

    diff = Q - P
    rmsd = np.sqrt((diff * diff).sum() / len(diff))
    max_delta = np.linalg.norm(diff, axis=1).max()

    return rmsd, max_delta

def fast_score(coords, close=1.3, far=3):
    '''
    return a fast to compute score
    used as a metric to evaluate
    the best structure between
    similar conformers. The higher,
    the least the structure is stable.
    '''
    dist_mat = cdist(coords, coords)
    close_contacts = dist_mat[dist_mat < far]
    return np.sum(close_contacts/(close-far) - far/(close-far))

def prune_conformers(structures, atomnos, k=1, max_rmsd=1, max_delta=None):
    '''
    Group structures into k subgroups and remove the similar ones.
    Similarity occurs for structures with both RMSD < max_rmsd and
    maximum deviation < max_delta.
    '''

    max_delta = max_rmsd * 2 if max_delta is None else max_delta

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

        similarity_mat = np.zeros((l, l))

        for i in range(l):
            for j in range(i+1,l):
                rmsd, max_dev = rmsd_and_max(structures_subset[i], structures_subset[j])
                if rmsd < max_rmsd and max_dev < max_delta:
                    similarity_mat[i,j] = 1
                    break

        matches = [(i,j) for i,j in zip(*np.where(similarity_mat))]

        g = nx.Graph(matches)

        subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
        groups = [tuple(graph.nodes) for graph in subgraphs]

        best_of_cluster = [sorted(group, key=lambda i: fast_score(structures[i]))[0] for group in groups]
        # of each cluster, keep the structure that looks the best

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