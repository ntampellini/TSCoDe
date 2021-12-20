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
from numba import njit
from rmsd import kabsch_rotate

from tscode.algebra import all_dists

# These functions are here to facilitate eventual porting to
# faster precompiled versions of themselves (Cython/C++/Julia/...)
# if the necessity ever occurs

@njit
def compenetration_check(coords, ids=None, thresh=1.3, max_clashes=0):

    clashes = 0
    # max_clashes clashes is good, max_clashes + 1 is not

    if ids is None:
        return 0 if np.count_nonzero(
                                     (all_dists(coords,coords) < 0.95) & (
                                      all_dists(coords,coords) > 0)
                                    ) > max_clashes else 1

    if len(ids) == 2:
        m1 = coords[0:ids[0]]
        m2 = coords[ids[0]:]
        return 0 if np.count_nonzero(all_dists(m2,m1) < thresh) > max_clashes else 1

    # if len(ids) == 3:

    m1 = coords[0:ids[0]]
    m2 = coords[ids[0]:ids[0]+ids[1]]
    m3 = coords[ids[0]+ids[1]:]

    clashes += np.count_nonzero(all_dists(m2,m1) < thresh)
    if clashes > max_clashes:
        return 0

    clashes += np.count_nonzero(all_dists(m3,m2) < thresh)
    if clashes > max_clashes:
        return 0

    clashes += np.count_nonzero(all_dists(m1,m3) < thresh)
    if clashes > max_clashes:
        return 0

    return 1

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
    dist_mat = all_dists(coords, coords)
    close_contacts = dist_mat[dist_mat < far]
    return np.sum(close_contacts/(close-far) - far/(close-far))

def prune_conformers(structures, atomnos, max_rmsd=0.5, max_delta=None):
    '''
    Removes similar structures by repeatedly grouping them into k
    subgroups and removing similar ones. A cache is present to avoid
    repeating RMSD computations.
    
    Similarity occurs for structures with both RMSD < max_rmsd and
    maximum deviation < max_delta.
    '''

    max_delta = max_rmsd * 2 if max_delta is None else max_delta

    heavy_atoms = (atomnos != 1)
    heavy_structures = np.array([structure[heavy_atoms] for structure in structures])

    cache_set = set()
    final_mask = np.ones(structures.shape[0], dtype=bool)
    
    for k in (5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1):
        num_active_str = np.count_nonzero(final_mask)
        
        if k == 1 or 5*k < num_active_str:
        # proceed only of there are at least five structures per group

            d = len(structures) // k

            for step in range(k):
            # operating on each of the k subdivisions of the array
                if step == k-1:
                    l = len(range(d*step, num_active_str))
                else:
                    l = len(range(d*step, d*(step+1)))

                similarity_mat = np.zeros((l, l))

                for i_rel in range(l):
                    for j_rel in range(i_rel+1,l):

                        i_abs = i_rel+(d*step)
                        j_abs = j_rel+(d*step)

                        if (i_abs, j_abs) not in cache_set:
                        # if we have already performed the comparison,
                        # structures were not similar and we can skip them

                            rmsd, max_dev = rmsd_and_max(heavy_structures[i_abs],
                                                         heavy_structures[j_abs])

                            if rmsd < max_rmsd and max_dev < max_delta:
                                similarity_mat[i_rel,j_rel] = 1
                                break

                for i_rel, j_rel in zip(*np.where(similarity_mat == False)):
                    i_abs = i_rel+(d*step)
                    j_abs = j_rel+(d*step)
                    cache_set.add((i_abs, j_abs))
                    # adding indexes of structures that are considered equal,
                    # so as not to repeat computing their RMSD
                    # Their index accounts for their position in the initial
                    # array (absolute index)

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

                for i in rejects:
                    abs_index = i + d*step
                    final_mask[abs_index] = 0

    return structures[final_mask], final_mask