import numpy as np
cimport numpy as np
from math import sqrt
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef float s
cdef float norm
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef float norm_of(np.ndarray[DTYPE_t, ndim=1] v):
    s = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    norm = sqrt(s)
    return norm

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef int compenetration_check(np.ndarray[DTYPE_t, ndim=2] coords, list ids):

    cdef float thresh = 1.2
    cdef int clashes = 0
    cdef int max_clashes = 2
    # max_clashes clashes is good, max_clashes + 1 is not
    cdef np.ndarray[DTYPE_t, ndim=2] m1, m2, m3
    cdef np.ndarray[DTYPE_t, ndim=1] v1, v2, v3
    cdef float dist

    if len(ids) == 2:
        m1 = coords[0:ids[0]]
        m2 = coords[ids[0]:]
        for v1 in m1:
            for v2 in m2:
                dist = norm_of(v1-v2)
                if dist < thresh:
                    clashes += 1
                if clashes > max_clashes:
                    return 0
        return 1

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
                    return 0

        for v2 in m2:
            for v3 in m3:
                dist = norm_of(v2-v3)
                if dist < thresh:
                    clashes += 1
                if clashes > max_clashes:
                    return 0

        for v3 in m3:
            for v1 in m1:
                dist = norm_of(v3-v1)
                if dist < thresh:
                    clashes += 1
                if clashes > max_clashes:
                    return 0

        return 1


