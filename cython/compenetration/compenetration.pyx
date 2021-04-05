import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef bint compenetration_check(np.ndarray[np.float64_t, ndim=3] coords, list ids):
    cdef float thresh = 1.2
    cdef int clashes = 0
    cdef np.ndarray[np.float64_t, ndim=2] m1, m2, m3
    cdef np.ndarray[np.float64_t, ndim=1] v1, v2, v3, delta
    cdef np.float64_t dist
    if len(ids) == 2:
        m1 = coords[0:ids[0]]
        m2 = coords[ids[0]:]
        for v1 in m1:
            for v2 in m2:
                delta = v1-v2
                dist = np.linalg.norm(delta)
                if delta < thresh:
                    clashes += 1
                if clashes > 2:
                    return False
        return True

    else:
        m1 = coords[0:ids[0]]
        m2 = coords[ids[0]:ids[0]+ids[1]]
        m3 = coords[ids[0]+ids[1]:]

        for v1 in m1:
            for v2 in m2:
                delta = v1-v2
                dist = np.linalg.norm(delta)
                if dist < thresh:
                    clashes += 1
                if clashes > 2:
                    return False

        for v2 in m2:
            for v3 in m3:
                delta = v2-v3
                dist = np.linalg.norm(delta)
                if dist < thresh:
                    clashes += 1
                if clashes > 2:
                    return False

        for v3 in m3:
            for v1 in m1:
                delta = v3-v1
                dist = np.linalg.norm(delta)
                if dist < 1.2:
                    clashes += 1
                if clashes > 2:
                    return False

        return True

