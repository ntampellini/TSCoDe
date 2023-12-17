import numpy as np
import numba as nb
from numba.typed import List
from tscode.algebra import norm_of

@nb.njit
def rmsd_and_max_numba(p, q):
    '''
    Returns a tuple with the RMSD between p and q
    and the maximum deviation of their positions

    '''
    
    # calculate the covariance matrix
    cov_mat = np.ascontiguousarray(p.T) @ q
    # cov_mat = np.transpose(p) * q
    
    # Compute the SVD
    v, _, w = np.linalg.svd(cov_mat)
    d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0

    if d:
        v[:, -1] = -v[:, -1]

    # Create Rotation matrix u
    rot_mat = np.dot(v, w)

    # Apply it to p
    p = np.ascontiguousarray(p) @ rot_mat

    # Calculate deviations
    diff = p - q

    # Calculate RMSD
    rmsd = np.sqrt((diff * diff).sum() / len(diff))

    # # Calculate max deviation
    # max_delta = np.linalg.norm(diff, axis=1).max()
    max_delta = max([norm_of(v) for v in diff])

    return rmsd, max_delta

@nb.njit
def _rmsd_similarity_cache(ref, structures, in_mask, cache, first_abs_index, rmsd_thr, maxdev_thr):
    '''
    Returns True (as the int 1) if ref is similar to any
    structure in structures, returning at the first instance of a match.
    Ignores structures that are False (0) in in_mask and saves pairs
    that evaluate to False (0) by returning them in computed_pairs.

    '''

    # initialize results container
    computed_pairs = List()

    # iterate over target structures
    for i in range(len(structures)):

        # only compare active structures
        if in_mask[i]:

            # check if we have performed this computation already,
            # and in that case we know the structures were not similar,
            # since the in_mask attribute is not False for ref nor i
            hash_value = (first_abs_index, first_abs_index+1+i)
            if hash_value in cache:
                return 0, computed_pairs
            
            # if we have not computed the RMSD before do it
            rmsd_value, maxdev_value = rmsd_and_max_numba(ref, structures[i])

            # if structures are not similar, add the result to the
            # cache List, because they will potentially return here,
            # while similar structures would be discarded
            if rmsd_value < rmsd_thr and maxdev_value < maxdev_thr:
                computed_pairs.append(hash_value)
                return 1, computed_pairs
            
    return 0, computed_pairs

@nb.njit
def _similarity_mask_rmsd_cache(structures, in_mask, cache, first_abs_index, rmsd_thr):
    '''
    For a given set of structures, check if each is similar
    to any other after itself. Returns a boolean mask to slice
    the array, only retaining the structures that are dissimilar.
    Also returns the non-similar pairs that were computed for caching purposes.

    '''
    #initialize results containers
    computed_pairs = List()
    out_mask = np.empty(shape=in_mask.shape, dtype=np.bool_)

    # set the threshold for the maximum deviation
    maxdev_thr = 2 * rmsd_thr

    # loop over the structures
    for i, ref in enumerate(structures):

        # only check for similarity if the structure is active
        if in_mask[i]:

            # reject structure i if it is similar to any other after itself
            similar, temp_computed_pairs = _rmsd_similarity_cache(
                                                                    ref,
                                                                    structures[i+1:],
                                                                    in_mask[i+1:],
                                                                    cache=cache,
                                                                    first_abs_index=first_abs_index,
                                                                    rmsd_thr=rmsd_thr,
                                                                    maxdev_thr=maxdev_thr,
                                                                )
            out_mask[i] = not similar

            # save the computed non-similar pairs
            computed_pairs.extend(temp_computed_pairs)

        else:
            out_mask[i] = 0

    return out_mask, computed_pairs

@nb.njit(parallel=True)
def _similarity_mask_rmsd_group(structures, in_mask, cache, k, rmsd_thr):
    '''
    Acts on chunks of the structures array in parallel,
    returning the updated mask and the non-similar pairs computed.

    '''
    # initialize the temporary and the final result containers
    computed_pairs = List([List([(-1,-1)]) for _ in range(k)])
    final_computed_pairs = List()
    out_mask = np.empty(shape=structures.shape[0], dtype=np.bool_)

    # calculate the size of each chunk
    chunksize = int(len(structures) // k)

    # iterate in parallel over chunks
    for chunk in nb.prange(int(k)):
        first = chunk*chunksize
        if chunk == k-1:
            last = len(structures)
        else:
            last = chunksize*(chunk+1)

        # get the structure chunk
        structures_chunk = structures[first:last]

        # compare structures within that chunk and save results to the out_mask and the
        # temporary computed_pairs, so that we have no race in appending to any list
        out_mask[first:last], computed_pairs[chunk] = _similarity_mask_rmsd_cache(
                                                                                    structures_chunk,
                                                                                    in_mask[first:last],
                                                                                    cache=cache,
                                                                                    first_abs_index=first,
                                                                                    rmsd_thr=rmsd_thr,
                                                                                )
    # unroll the temporary results in the final container
    for L in computed_pairs:
        final_computed_pairs.extend(L)

    return out_mask, final_computed_pairs

def prune_conformers_rmsd(structures, atomnos, rmsd_thr=0.5):
    '''
    Removes similar structures by repeatedly grouping them into k
    subgroups and removing similar ones. A cache is present to avoid
    repeating RMSD computations.
    
    Similarity occurs for structures with both rmsd < rmsd_thr and
    maximum absolute atomic deviation < 2 * rmsd_thr.

    Returns the pruned structures and the corresponding boolean mask.

    '''

    # only consider non-hydrogen atoms
    heavy_atoms = (atomnos != 1)
    heavy_structures = np.array([structure[heavy_atoms] for structure in structures])

    # initialize the output mask
    out_mask = np.ones(shape=structures.shape[0], dtype=np.bool_)
    cache = List([(-1,-1)])

    # split the structure array in subgroups and prune them internally
    for k in (5e5, 2e5, 1e5, 5e4, 2e4, 1e4,
              5000, 2000, 1000, 500, 200, 100,
              50, 20, 10, 5, 2, 1):
        
        # choose only k values such that every subgroup
        # has on average at least twenty active structures in it
        if k == 1 or 20*k < np.count_nonzero(out_mask):

            # compute similarities and get back the out_mask
            # and the pairings to be added to cache
            out_mask, computed_pairs = _similarity_mask_rmsd_group(
                                                                    heavy_structures,
                                                                    out_mask,
                                                                    cache=cache,
                                                                    k=k,
                                                                    rmsd_thr=rmsd_thr,
                                                                )
            # extend the cache set with the new pairings we computed
            cache.extend(computed_pairs)

    return structures[out_mask], out_mask

def _rmsd_similarity(ref, structures, rmsd_thr=0.5):
    '''
    Simple, non-cached, non-jitted version
    of the homonym function above.

    '''

    # iterate over target structures
    for structure in structures:
        
        # compute RMSD and max deviation
        rmsd_value, maxdev_value = rmsd_and_max_numba(ref, structure)

        if rmsd_value < rmsd_thr and maxdev_value < 2 * rmsd_thr:
            return True
            
    return False