# python setup.py build_ext --inplace

# %%
import sanity as s
import numpy as np
import networkx as nx
import os
from time import time
os.chdir(r'C:\Users\Nik\Desktop\Coding\TSCoDe')
from optimization_methods import sanity_check
n = 100
coords = np.random.rand(n,3)
atomnos = np.random.choice([1,6,7,8], n)

ids = np.array([[3,6]])
edges_list = [[tuple(np.random.choice(n, 2)) for _ in range(n)] for _ in range(3)]
graphs = [nx.Graph(edges) for edges in edges_list]
# update with real case scenario dataset

# %% 
n = 100
t1 = time()
for _ in range(n):
    sanity_check(coords, atomnos, ids, graphs)
t2 = time()
molecules_bonds = [list(g.edges) for g in graphs]
atoms = [len(g.nodes) for g in graphs]
molecules_atomnumbers = [6,7,8]
expected_bonds = set()
for i in range(3):
    pos = 0
    while i != 0:
        pos += molecules_atomnumbers[i-1]
        i -= 1

    for a, b in molecules_bonds[i]:
        if a != b:
            expected_bonds.add((a+pos, b+pos))

t3 = time()
for _ in range(n):
    s.sanity_check(coords, atomnos, ids, expected_bonds)
t4 = time()
print(f'Cython is {round((t2-t1)/(t4-t3), 2)} times faster')
# %% internal benchmark
molecules_bonds = [list(g.edges) for g in graphs]
atoms = [len(g.nodes) for g in graphs]
molecules_atomnumbers = [6,7,8]
expected_bonds = set()
for i in range(3):
    pos = 0
    while i != 0:
        pos += molecules_atomnumbers[i-1]
        i -= 1

    for a, b in molecules_bonds[i]:
        if a != b:
            expected_bonds.add((a+pos, b+pos))


s.sanity_check(coords, atomnos, ids, expected_bonds)

# %%
from math import sqrt
from numba import njit

radii = {
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

@njit
def d_min(e1, e2):
    return 1.2 * (radii[e1] + radii[e2])
    # return 0.2 + (radii[e1] + radii[e2])
# if this is somewhat prone to bugs, this might help https://cccbdb.nist.gov/calcbondcomp1x.asp

@njit
def norm_of(v):
    s = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    norm = sqrt(s)
    return norm

@njit
def get_bonds(coords, atomnos):
    '''
    :params coords: atomic coordinates as 3D vectors
    :params atomnos: atomic numbers as a list
    :return connectivity graph
    '''   
    l = coords.shape[0]
    bonds = set()
    for i in range(l):
        for j in range(i+1,l):
            delta = coords[i]-coords[j]
            dist = norm_of(delta)
            thresh = d_min(atomnos[i], atomnos[j])
            if dist < thresh:
                bonds.add((i,j))

    return bonds

@njit
def sanity_check_v2(TS_structure, TS_atomnos, constrained_indexes, expected_bonds, max_new_bonds=3):
    '''
    :params TS_structure: list of coordinates for each atom in the TS
    :params TS_atomnos: list of atomic numbers for each atom in the TS
    :params constrained_indexes: indexes of constrained atoms in the TS geometry
    :params mols_graphs: list of molecule.graph objects, containing connectivity information
    :params max_new_bonds: maximum number of apperent new bonds in TS geometry to accept the
                           structure as a valid one. Too high values may cause ugly results,
                           too low might discard structures that would have led to good results.

    :return result: bool, indicating structure sanity
    '''
    new_bonds = get_bonds(TS_structure, TS_atomnos)
    delta_bonds = new_bonds - expected_bonds

    c_bonds = set()
    for a, b in constrained_indexes:
        c_bonds.add((a,b))

    delta_bonds -= c_bonds

    delta_bonds_num = len(delta_bonds)
    if delta_bonds_num > max_new_bonds:
        return False
    else:
        return True

# %%
# from numba import njit
# jitted = njit(sanity_check_v2, parallel=True)

molecules_bonds = [list(g.edges) for g in graphs]
atoms = [len(g.nodes) for g in graphs]
molecules_atomnumbers = [6,7,8]
expected_bonds = set()
for i in range(3):
    pos = 0
    while i != 0:
        pos += molecules_atomnumbers[i-1]
        i -= 1

    for a, b in molecules_bonds[i]:
        if a != b:
            expected_bonds.add((a+pos, b+pos))
# %%
sanity_check_v2(coords, atomnos, ids, expected_bonds)
# %%
np.linalg.norm(coords, axis=(0,1))