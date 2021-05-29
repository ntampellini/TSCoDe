import numpy as np
cimport numpy as np
import networkx as nx
cimport cython

# from libc.stdio cimport stdout, fprintf
# import sys
# from time import time
# # after print statements, use sys.stdout.flush()


DTYPE = np.float
ctypedef np.float_t DTYPE_t


# cdef int en(tup, energies):
#     ens = [energies[t] for t in tup]
#     return tup[ens.index(min(ens))]
        

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline np.ndarray[np.int_t, ndim=3, mode='c'] scramble_mask(np.ndarray array,
                                                np.ndarray[np.int_t, ndim=1, mode='c'] sequence):
    return np.array([array[s] for s in sequence])

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline np.ndarray[np.int_t, ndim=3, mode='c'] scramble(np.ndarray[DTYPE_t, ndim=3, mode='c'] array,
                                                            np.ndarray[np.int_t, ndim=1, mode='c'] sequence):
    return np.array([array[s] for s in sequence])

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef tuple prune_conformers(np.ndarray[DTYPE_t, ndim=3, mode='c'] structures,
                             int k = 1,
                             double max_rmsd = 1.):

    cdef np.ndarray[np.int_t, ndim=1, mode='c'] r, sequence, inv_sequence
    cdef np.ndarray mask

    if k != 1:

        r = np.arange(structures.shape[0])
        sequence = np.random.permutation(r)
        inv_sequence = np.array([np.where(sequence == i)[0][0] for i in r], dtype=int)

        structures = scramble(structures, sequence)
        # energies = scramble_mask(energies, sequence)
        # scrambling array before splitting, so to improve efficiency when doing
        # multiple runs of group pruning

    cdef list mask_out = []
    cdef unsigned int step, d = len(structures) // k
    cdef unsigned int l
    cdef unsigned int i, j
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] rmsd_mat
    cdef double[:,:] rmsd_mat_view
    cdef int[:] energies_subset
    cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] structures_subset
    cdef tuple where
    cdef list matches, subgraphs, groups, best_of_cluster, rejects_sets, rejects
    cdef object g
    cdef double val

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
        rmsd_mat_view = rmsd_mat

        # t0 = time()

        for i in range(l):
            for j in range(i+1,l):
                val = rmsd_c(structures_subset[i], structures_subset[j])
                rmsd_mat_view[i, j] = val
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


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline double rmsd_c(np.ndarray[DTYPE_t, ndim=2, mode='c'] coords1, np.ndarray[DTYPE_t, ndim=2, mode='c'] coords2):

    cdef float atol = 1e-9
    cdef double c0, c1, c2
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] A = coords1 - np.mean(coords1, axis=0)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] B = coords2 - np.mean(coords2, axis=0)


    cdef int N = A.shape[0]

    cdef double Ga = np.trace(A.T @ A)
    cdef double Gb = np.trace(B.T @ B)

    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] M = M_mtx(A, B)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] K = K_mtx(M)

    cdef double l_max = _lambda_max_eig(K)

    cdef double s = Ga + Gb - 2 * l_max
    cdef double rmsd

    if abs(s) < atol:  # Avoid numerical errors when Ga + Gb = 2 * l_max
        rmsd = 0.0
    else:
        rmsd = np.sqrt(s / N)

    return rmsd


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline np.ndarray M_mtx(np.ndarray[DTYPE_t, ndim=2, mode='c'] A, np.ndarray[DTYPE_t, ndim=2, mode='c'] B):
    return B.T @ A

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline np.ndarray K_mtx(np.ndarray[DTYPE_t, ndim=2, mode='c'] M):

    S_xx = M[0, 0]
    S_xy = M[0, 1]
    S_xz = M[0, 2]
    S_yx = M[1, 0]
    S_yy = M[1, 1]
    S_yz = M[1, 2]
    S_zx = M[2, 0]
    S_zy = M[2, 1]
    S_zz = M[2, 2]

    # p = plus, m = minus
    S_xx_yy_zz_ppp = S_xx + S_yy + S_zz
    S_yz_zy_pm = S_yz - S_zy
    S_zx_xz_pm = S_zx - S_xz
    S_xy_yx_pm = S_xy - S_yx
    S_xx_yy_zz_pmm = S_xx - S_yy - S_zz
    S_xy_yx_pp = S_xy + S_yx
    S_zx_xz_pp = S_zx + S_xz
    S_xx_yy_zz_mpm = -S_xx + S_yy - S_zz
    S_yz_zy_pp = S_yz + S_zy
    S_xx_yy_zz_mmp = -S_xx - S_yy + S_zz

    return np.array(
        [
            [S_xx_yy_zz_ppp, S_yz_zy_pm, S_zx_xz_pm, S_xy_yx_pm],
            [S_yz_zy_pm, S_xx_yy_zz_pmm, S_xy_yx_pp, S_zx_xz_pp],
            [S_zx_xz_pm, S_xy_yx_pp, S_xx_yy_zz_mpm, S_yz_zy_pp],
            [S_xy_yx_pm, S_zx_xz_pp, S_yz_zy_pp, S_xx_yy_zz_mmp],
        ]
    )

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef inline double _lambda_max_eig(np.ndarray[DTYPE_t, ndim=2, mode='c'] K):
    return np.max(np.linalg.eig(K)[0])

