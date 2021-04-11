import numpy as np
cimport numpy as np
cimport cython
from math import sqrt

cdef dict radii = {
    0: 0.2,
    1: 0.31,
    2: 0.28,
    3: 1.28,
    4: 0.96,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    10: 0.58,
    11: 1.66,
    12: 1.41,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    18: 1.06,
    19: 2.03,
    20: 1.76,
    21: 1.7,
    22: 1.6,
    23: 1.53,
    24: 1.39,
    25: 1.39,
    26: 1.32,
    27: 1.26,
    28: 1.24,
    29: 1.32,
    30: 1.22,
    31: 1.22,
    32: 1.2,
    33: 1.19,
    34: 1.2,
    35: 1.2,
    36: 1.16,
    37: 2.2,
    38: 1.95,
    39: 1.9,
    40: 1.75,
    41: 1.64,
    42: 1.54,
    43: 1.47,
    44: 1.46,
    45: 1.42,
    46: 1.39,
    47: 1.45,
    48: 1.44,
    49: 1.42,
    50: 1.39,
    51: 1.39,
    52: 1.38,
    53: 1.39,
    54: 1.4,
    55: 2.44,
    56: 2.15,
    57: 2.07,
    58: 2.04,
    59: 2.03,
    60: 2.01,
    61: 1.99,
    62: 1.98,
    63: 1.98,
    64: 1.96,
    65: 1.94,
    66: 1.92,
    67: 1.92,
    68: 1.89,
    69: 1.9,
    70: 1.87,
    71: 1.87,
    72: 1.75,
    73: 1.7,
    74: 1.62,
    75: 1.51,
    76: 1.44,
    77: 1.41,
    78: 1.36,
    79: 1.36,
    80: 1.32,
    81: 1.45,
    82: 1.46,
    83: 1.48,
    84: 1.4,
    85: 1.5,
    86: 1.5,
    87: 2.6,
    88: 2.21,
    89: 2.15,
    90: 2.06,
    91: 2.0,
    92: 1.96,
    93: 1.9,
    94: 1.87,
    95: 1.8,
    96: 1.69
    }


cdef float x = 1.2
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef float d_min(int e1, int e2):
    return x * (radii[e1] + radii[e2])
    # return 0.2 + (radii[e1] + radii[e2])
# if this is somewhat prone to bugs, this might help https://cccbdb.nist.gov/calcbondcomp1x.asp

cdef float s
cdef float norm
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef float norm_of(np.ndarray[np.float64_t, ndim=1] v):
    s = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    norm = sqrt(s)
    return norm


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef set get_bonds(np.ndarray[np.float64_t, ndim=2] coords,
                   np.ndarray[np.int_t, ndim=1] atomnos):
    cdef int l = coords.shape[0]
    cdef set bonds = set()
    cdef int i, j
    cdef np.ndarray[np.float64_t, ndim=1] delta
    cdef float dist, thresh
    for i in range(l):
        for j in range(i+1,l):
            delta = coords[i]-coords[j]
            dist = norm_of(delta)
            thresh = d_min(atomnos[i], atomnos[j])
            if dist < thresh:
                bonds.add((i,j))

    return bonds


cpdef bint sanity_check(np.ndarray[np.float64_t, ndim=2] TS_structure, np.ndarray[np.int_t, ndim=1] TS_atomnos, np.ndarray[np.int_t, ndim=2] constrained_indexes, set expected_bonds, int max_new_bonds=3):
    cdef int a, b
    cdef set new_bonds = get_bonds(TS_structure, TS_atomnos)
    cdef set delta_bonds = new_bonds - expected_bonds

    cdef set c_bonds = set()
    for a, b in constrained_indexes:
        c_bonds.add((a,b))

    delta_bonds -= c_bonds

    cdef int delta_bonds_num = len(delta_bonds)

    if delta_bonds_num > max_new_bonds:
        return False
    else:
        return True
