# python setup.py build_ext --inplace

# %%
import sanity as s
import numpy as np
import networkx as nx
import os
from time import time
# os.chdir(r'C:\Users\Nik\Desktop\Coding\TSCoDe')
os.chdir(r'C:\Users\ehrma\Desktop\Coding\TSCoDe')
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
import numpy as np
n = 100
coords = np.random.rand(n,3)
atomnos = np.random.choice([1,6,7,8], n)

a = 13
b = 34
c = 100 - a - b
ids = [a, b, c]
# ids = [30, 70]
def compenetration_check(coords, ids):
    thresh = 1.2
    clashes = 0
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

import compenetration as c
# %% Python
%%time
for _ in range(int(1e5)):
    compenetration_check(coords, ids)

# %% Cython
%%time
for _ in range(int(1e5)):
    c.compenetration_check(coords, ids)

# %% Cython 2
%%time
for _ in range(int(1e5)):
    c.compenetration_check2(coords, ids)
# %%
