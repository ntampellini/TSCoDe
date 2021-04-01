from time import time
import numpy as np
from spyrmsd.rmsd import rmsd
import networkx as nx

def vanilla(structures, atomnos, energies, max_rmsd=1, debug=False):
    '''
    Remove conformations that are too similar (have a small RMSD value).
    When removing structures, only the lowest energy one is kept.

    :params structures: numpy array of conformations
    :params energies: list of energies for each conformation
    :params max_rmsd: maximum rmsd value to consider two structures identical, in Angstroms
    '''
    rmsd_mat = np.zeros((len(structures), len(structures)))
    rmsd_mat[:] = np.nan
    for i, tgt in enumerate(structures):
        for j, ref in enumerate(structures[i+1:]):
            rmsd_mat[i][i+j+1] = rmsd(tgt, ref, atomnos, atomnos, center=True, minimize=True)

    matches = np.where(rmsd_mat < max_rmsd)
    matches = [(i,j) for i,j in zip(matches[0], matches[1])]

    g = nx.Graph(matches)

    if debug:
        g.add_nodes_from(range(len(structures)))
        pos = nx.spring_layout(g)
        nx.draw(g, pos=pos, labels={i:i for i in range(len(g))})
        import matplotlib.pyplot as plt
        plt.show()

    subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
    groups = [tuple(graph.nodes) for graph in subgraphs]

    def en(tup):
        ens = [energies[t] for t in tup]
        return tup[ens.index(min(ens))]

    best_of_cluster = [en(group) for group in groups]
    rejects_sets = [set(a) - {b} for a, b in zip(groups, best_of_cluster)]

    rejects = []
    def flatten(seq):
        for s in seq:
            if type(s) in (tuple, list, set):
                flatten(s)
            else:
                rejects.append(s)

    flatten(rejects_sets)

    mask = np.array([True for _ in range(len(structures))], dtype=bool)
    for i in rejects:
        mask[i] = False

    return structures[mask], mask

def test(structures, atomnos, energies, max_rmsd=1, debug=False):
    '''
    Remove conformations that are too similar (have a small RMSD value).
    When removing structures, only the lowest energy one is kept.

    :params structures: numpy array of conformations
    :params energies: list of energies for each conformation
    :params max_rmsd: maximum rmsd value to consider two structures identical, in Angstroms
    '''
    t_i = time()
    rmsd_mat = np.zeros((len(structures), len(structures)))
    rmsd_mat[:] = np.nan
    for i, tgt in enumerate(structures):
        for j, ref in enumerate(structures[i+1:]):
            rmsd_mat[i][i+j+1] = rmsd(tgt, ref, atomnos, atomnos, center=True, minimize=True)
    t_f = time()
    print(f'matrix building - {round(t_f-t_i,2)} s')


    t_i = time()
    matches = np.where(rmsd_mat < max_rmsd)
    matches = [(i,j) for i,j in zip(matches[0], matches[1])]
    t_f = time()
    print(f'matrix feature extraction - {round(t_f-t_i,2)} s')

    t_i = time()
    g = nx.Graph(matches)
    t_f = time()
    print(f'graph building - {round(t_f-t_i,2)} s')


    if debug:
        g.add_nodes_from(range(len(structures)))
        pos = nx.spring_layout(g)
        nx.draw(g, pos=pos, labels={i:i for i in range(len(g))})
        import matplotlib.pyplot as plt
        plt.show()

    t_i = time()
    subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
    groups = [tuple(graph.nodes) for graph in subgraphs]
    t_f = time()
    print(f'subgraphs extraction - {round(t_f-t_i,2)} s')

    def en(tup):
        ens = [energies[t] for t in tup]
        return tup[ens.index(min(ens))]

    t_i = time()
    best_of_cluster = [en(group) for group in groups]
    rejects_sets = [set(a) - {b} for a, b in zip(groups, best_of_cluster)]
    t_f = time()
    print(f'best of cluster extraction - {round(t_f-t_i,2)} s')

    rejects = []
    def flatten(seq):
        for s in seq:
            if type(s) in (tuple, list, set):
                flatten(s)
            else:
                rejects.append(s)

    t_i = time()
    flatten(rejects_sets)
    t_f = time()
    print(f'matrix feature extraction - {round(t_f-t_i,2)} s')

    t_i = time()
    mask = np.array([True for _ in range(len(structures))], dtype=bool)
    for i in rejects:
        mask[i] = False
    t_f = time()
    print(f'masking - {round(t_f-t_i,2)} s')

    return structures[mask], mask

def skipped(structures, atomnos, energies, max_rmsd=1, debug=False):
    '''
    Remove conformations that are too similar (have a small RMSD value).
    When removing structures, only the lowest energy one is kept.

    :params structures: numpy array of conformations
    :params energies: list of energies for each conformation
    :params max_rmsd: maximum rmsd value to consider two structures identical, in Angstroms
    '''
    rmsd_mat = np.zeros((len(structures), len(structures)))
    rmsd_mat[:] = np.nan
    for i, tgt in enumerate(structures):
        for j, ref in enumerate(structures[i+1:]):
            value = rmsd(tgt, ref, atomnos, atomnos, center=True, minimize=True)
            rmsd_mat[i][i+j+1] = value
            if value < max_rmsd:
                break

    matches = np.where(rmsd_mat < max_rmsd)
    matches = [(i,j) for i,j in zip(matches[0], matches[1])]

    g = nx.Graph(matches)

    if debug:
        g.add_nodes_from(range(len(structures)))
        pos = nx.spring_layout(g)
        nx.draw(g, pos=pos, labels={i:i for i in range(len(g))})
        import matplotlib.pyplot as plt
        plt.show()

    subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
    groups = [tuple(graph.nodes) for graph in subgraphs]

    def en(tup):
        ens = [energies[t] for t in tup]
        return tup[ens.index(min(ens))]

    best_of_cluster = [en(group) for group in groups]
    rejects_sets = [set(a) - {b} for a, b in zip(groups, best_of_cluster)]

    rejects = []
    def flatten(seq):
        for s in seq:
            if type(s) in (tuple, list, set):
                flatten(s)
            else:
                rejects.append(s)

    flatten(rejects_sets)

    mask = np.array([True for _ in range(len(structures))], dtype=bool)
    for i in rejects:
        mask[i] = False

    return structures[mask], mask


n = 100
i = np.array([1,6,7,8])

vecs = np.random.rand(n,50,3)
atomnos = np.random.choice(i, size=50)
energies = np.repeat(0, n)

# t_i = time()
# # vanilla(vecs, atomnos, energies)
# test(vecs, atomnos, energies)
# t_f = time()

# print(f'total - {round(t_f-t_i,2)} s')

t_i = time()
skipped(vecs, atomnos, energies)
t_f = time()
print(f'skipped - {round(t_f-t_i,2)} s')

t_i = time()
vanilla(vecs, atomnos, energies)
t_f = time()
print(f'vanilla - {round(t_f-t_i,2)} s')