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
import time
from copy import deepcopy

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

from tscode.hypermolecule_class import align_structures, graphize
from tscode.optimization_methods import optimize
from tscode.parameters import nci_dict
from tscode.python_functions import compenetration_check, prune_conformers
from tscode.settings import DEFAULT_FF_LEVELS, FF_CALC, FF_OPT_BOOL
from tscode.utils import (cartesian_product, findPaths, loadbar, neighbors, pt,
                          rotate_dihedral, time_to_string)


class Torsion:
    def __init__(self, i1, i2, i3, i4):
        
        self.i1 = i1
        self.i2 = i2
        self.i3 = i3
        self.i4 = i4
        self.torsion = (i1, i2, i3 ,i4)

    def in_cycle(self, graph):
        '''
        Returns True if the torsion is part of a cycle
        '''
        graph.remove_edge(self.i2, self.i3)
        cyclical = nx.has_path(graph, self.i1, self.i4)
        graph.add_edge(self.i2, self.i3)
        return cyclical

    def is_rotable(self, graph, hydrogen_bonds) -> bool:
        '''
        hydrogen bonds: iterable with pairs of sorted atomic indexes
        '''

        if tuple(sorted((self.i2, self.i3))) in hydrogen_bonds:
            self.n_fold = 6
            return True

        if _is_free(self.i2, graph) or (
           _is_free(self.i3, graph)):

            if _is_nondummy(self.i2, self.i3, graph) and (
               _is_nondummy(self.i3, self.i2, graph)):

                self.n_fold = self.get_n_fold(graph)
                return True

        return False

    def get_n_fold(self, graph) -> int:

        nums = (graph.nodes[self.i2]['atomnos'],
                graph.nodes[self.i3]['atomnos'])

        # if 1 in nums:
        #     return 6 # H-N, H-O

        if (6 in nums) or (7 in nums) or (16 in nums): # if C, N or S atoms

            spn2 = next((i for i in (1,2,3) if _is_sp_n(self.i2, graph, i)), None)
            spn3 = next((i for i in (1,2,3) if _is_sp_n(self.i3, graph, i)), None)

            if 3 in (spn2, spn3): # Csp3-X, Nsp3-X, Ssulfone-X
                return 3

            # if 2 in (spn2, spn3):
            #     return 2 # Nsp2-O, S-O, P-O, sp2-sp2, sp2-sp, sp2-O, sp2-N

        # raise SystemExit(f'Torsion not defined:\nnum {nums[0]} - {nums[1]}\nnum {spn2} - {spn3}')
        return 2 #O-O

    def get_angles(self):
        return {
                2:(0, 180),
                3:(0, 120, 240),
                6:(0, 60, 120, 180, 240, 300),
                }[self.n_fold]

    def sort_torsion(self, graph, constrained_indexes) -> None:
        '''
        Acts on the self.torsion tuple leaving it as it is or
        reversing it, so that the first index of it (from which
        rotation will act) is external to the molecule constrained
        indexes. That is we make sure to rotate external groups
        and not the whole transition state.
        '''
        graph.remove_edge(self.i2, self.i3)
        for d in constrained_indexes.flatten():
            if nx.has_path(graph, self.i2, d):
                self.torsion = tuple(reversed(self.torsion))
        graph.add_edge(self.i2, self.i3)


def _is_sp_n(index, graph, n):
    '''
    Returns True if the index has:
    - n=1 : 2 bonds
    - n=2 : 3 bonds
    - n=3 : 4 bonds
    '''
    if len(neighbors(graph, index)) != (2,3,4)[n-1]:
        return False
    return True

def _is_amide_n(index, graph):
    '''
    Returns true if the nitrogen atom at the given
    index is a nitrogen and is part of an amide.
    Carbamates and ureas return True.
    '''
    if graph.nodes[index]['atomnos'] == 7:
        nb = neighbors(graph, index)
        for n in nb:
            if graph.nodes[n]['atomnos'] == 6:
                nb_nb = neighbors(graph, n)
                if len(nb_nb) == 3:
                    nb_nb_sym = [graph.nodes[i]['atomnos'] for i in nb_nb]
                    if 8 in nb_nb_sym:
                        return True
    return False

def _is_ester_o(index, graph):
    '''
    Returns true if the atom at the given
    index is an oxygen and is part of an ester.
    Carbamates and carbonates return True,
    Carboxylic acids return False.
    '''
    if graph.nodes[index]['atomnos'] == 8:
        nb = neighbors(graph, index)
        if not 1 in nb:
            for n in nb:
                if graph.nodes[n]['atomnos'] == 6:
                    nb_nb = neighbors(graph, n)
                    if len(nb_nb) == 3:
                        nb_nb_sym = [graph.nodes[i]['atomnos'] for i in nb_nb]
                        if nb_nb_sym.count(8) > 1:
                            return True
    return False
    
def _is_free(index, graph):
    '''
    Return True if the index specified
    satisfies all of the following:
    - Is not a sp2 carbonyl carbon atom
    - Is not the oxygen atom of an ester
    - Is not the nitrogen atom of an amide

    '''
    if all((
            graph.nodes[index]['atomnos'] == 6,
            _is_sp_n(index, graph, 2),
            8 in (graph.nodes[n]['atomnos'] for n in neighbors(graph, index))
          )):
        return False

    if _is_amide_n(index, graph):
        return False

    if _is_ester_o(index, graph):
        return False

    return True

def _is_nondummy(i, root, graph) -> bool:
    '''
    Checks that a molecular rotation along the dihedral
    angle (*, root, i, *) is non-dummy, that is the atom
    at index i, in the direction opposite to the one leading
    to root, has different substituents. i.e. methyl and tBu
    rotations return False.

    Thought to eliminate methyl/tert-butyl rotations.
    '''
    G = deepcopy(graph)
    nb = neighbors(G, i)
    nb.remove(root)

    if len(nb) == 1:
        if len(neighbors(G, nb[0])) == 2:
            return False
    # if one end has two bonds only (one with root)
    # and the other atom has two bonds only (one with i neighbor)
    # the rotation is considered dummy: some other rotation
    # will account for its freedom (i.e. alkynes, hydrogen bonds)

    for n in nb:
        G.remove_edge(i, n)

    subgraphs_nodes = [_set for _set in nx.connected_components(G)
                       if not root in _set]

    if len(subgraphs_nodes) == 1:
        return True
    # if i is part of a cycle and root is exocyclic, 
    # the rotation is considered non-dummy.

    subgraphs = [nx.subgraph(G, s) for s in subgraphs_nodes]
    for sub in subgraphs[1:]:
        if not nx.is_isomorphic(subgraphs[0], sub,
                                node_match=lambda n1, n2: n1['atomnos'] == n2['atomnos']):
            return True

    return False

def _get_hydrogen_bonds(coords, atomnos, graph):
    '''
    '''
    output = []
    for i1, (coord1, num1) in enumerate(zip(coords, atomnos)):
        for i2, (coord2, num2) in enumerate(zip(coords[i1+1:], atomnos[i1+1:])):
            i2 += (i1+1)

            if (i1, i2) not in graph.edges and num1 in (1,7,8) and num2 in (1,7,8):
                tag = ''.join(sorted([pt[num1].symbol, pt[num2].symbol]))
            
                if tag in nci_dict:
                    thresh, _ = nci_dict[tag]
                    dist = np.linalg.norm(coord1-coord2)

                    if dist < thresh:
                        output.append((i1, i2))
    return output

def _get_rotation_mask(graph, torsion):
    '''
    '''
    i1, i2, i3, _ = torsion

    # temp_graph = deepcopy(graph)
    # temp_graph.remove_edge(i2, i3)
    # reachable_indexes = nx.shortest_path(temp_graph, i1).keys()    # 11.6 s

    graph.remove_edge(i2, i3)
    reachable_indexes = nx.shortest_path(graph, i1).keys()
    graph.add_edge(i2, i3)                                           # 1.6 s

    return np.array([i in reachable_indexes for i in graph.nodes])

def _get_torsions(graph, hydrogen_bonds):
    '''
    Returns list of Torsion objects
    '''
    allpaths = []
    # allpaths = [*findPaths(graph, node, 3) for node in graph]
    for node in graph:
        allpaths.extend(findPaths(graph, node, 3))
    # get all possible continuous indexes quadruplets

    torsions, bond_torsions = [], []
    for path in allpaths:
        _, i2, i3, _ = path
        bt = sorted((i2, i3))

        if bt not in bond_torsions:
            t = Torsion(*path)

            if not t.in_cycle(graph) and t.is_rotable(graph, hydrogen_bonds):

                bond_torsions.append(bt)
                torsions.append(t)
    # Create non-redundant torsion objects
    # Rejects (4,3,2,1) if (1,2,3,4) is present
    # Rejects torsions that do not represent a rotable bond

    return torsions

def _group_torsions(coords, torsions, max_size=3):
    '''
    '''
    torsions_indexes = [t.torsion for t in torsions]
    # get torsion indexes

    torsions_centers = np.array([np.mean((coords[i2],coords[i3]), axis=0) for _, i2, i3, _ in torsions_indexes])
    # compute spatial distance

    l = len(torsions)
    for n in range ((l//max_size)+1, l):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(torsions_centers)

        output = [[] for _ in range(n)]
        for torsion, cluster in zip(torsions, kmeans.labels_):
            output[cluster].append(torsion)

        if max([len(group) for group in output]) <= max_size:
            break

    output = sorted(output, key=lambda x: len(x))
    # largest groups last
    
    return output

def clustered_csearch(coords,
                        atomnos,
                        constrained_indexes=None,
                        ff_opt=False,
                        n=10,
                        mode=1,
                        calc=None,
                        method=None,
                        logfunction=print):
    '''
    n: number of structures to keep from each torsion cluster
    mode: 0 - keep the n lowest energy conformers
          1 - keep the n most diverse conformers
    '''

    assert mode == 1 or ff_opt, 'Either leave mode=1 or turn on force field optimization'
    assert mode in (0,1), 'The mode keyword can only be 0 or 1'

    calc = FF_CALC if calc is None else calc
    method = DEFAULT_FF_LEVELS[calc] if method is None else method
    # Set default calculator attributes if user did not specify them

    constrained_indexes = np.array([]) if constrained_indexes is None else constrained_indexes
    t_start_run = time.time()

    graph = graphize(coords, atomnos)
    for i1, i2 in constrained_indexes:
        graph.add_edge(i1, i2)
    # build a molecular graph of the TS
    # that includes constrained indexes pairs
    
    if len(list(nx.connected_components(graph))) > 1:
        hydrogen_bonds = _get_hydrogen_bonds(coords, atomnos, graph)
        for hb in hydrogen_bonds:
            graph.add_edge(*hb)
    else:
        hydrogen_bonds = ()

    torsions = _get_torsions(graph, hydrogen_bonds)
    # get all torsions that we should explore

    for t in torsions:
        t.sort_torsion(graph, constrained_indexes)
    # sort torsion indexes so that first index of each torsion
    # is the half that will move and is external to the TS

    grouped_torsions = _group_torsions(coords,
                                       torsions,
                                       max_size=3 if ff_opt else 8)

    logfunction(f'--> Clustered CSearch - mode {mode} ({"stability" if mode == 0 else "diversity"}) - ' +
                f'{len(torsions)} torsions in {len(grouped_torsions)} group{"s" if len(grouped_torsions) != 1 else ""} - ' +
                f'{[len(t) for t in grouped_torsions]}')
    
    ############################################## DEBUG
    with open('n_fold_log.txt', 'w') as f:
        for t in torsions:
            f.write(f'{t.torsion} - {t.n_fold}-fold\n')
    _write_torsion_vmd(coords, atomnos, constrained_indexes, grouped_torsions)
    # quit()
    ############################################## DEBUG

    output_structures = []
    starting_points = [coords]
    for tg, torsions_group in enumerate(grouped_torsions):

        logfunction()

        angles  = cartesian_product(*[t.get_angles() for t in torsions_group])
        candidates = len(angles)*len(starting_points)

        new_structures = []
        for s, sp in enumerate(starting_points):
            for a, angle_set in enumerate(angles):

                new_coords = deepcopy(sp)

                for t, torsion in enumerate(torsions_group):
                    angle = angle_set[t]
                    if angle != 0:
                        mask = _get_rotation_mask(graph, torsion.torsion)
                        new_coords = rotate_dihedral(new_coords, torsion.torsion, angle, mask=mask)

                new_structures.append(new_coords)

        mask = np.zeros(len(new_structures), dtype=bool)
        new_structures = np.array(new_structures)
        for s, structure in enumerate(new_structures):
            mask[s] = compenetration_check(structure)

        new_structures = new_structures[mask]
        for_comp = np.count_nonzero(mask == False)

        logfunction(f'-> Group {tg+1}/{len(grouped_torsions)} - {len(torsions_group)} bonds, ' +
                    f'{[t.n_fold for t in torsions_group]} n-folds, {len(starting_points)} ' + 
                    f'starting point{"s" if len(starting_points) > 1 else ""} = {candidates} conformers')
        logfunction(f'   {candidates} generated, {for_comp} removed for compenetration ({len(new_structures)} left)')

        energies = None
        if ff_opt:

            t_start = time.time()

            energies = np.zeros(new_structures.shape[0])
            for c, coords in enumerate(deepcopy(new_structures)):

                opt_coords, energy, success = optimize(coords,
                                                        atomnos,
                                                        calc,
                                                        method=method,
                                                        constrained_indexes=constrained_indexes)

                if success:
                    new_structures[c] = opt_coords
                    energies[c] = energy

                else:
                    energies[c] = np.inf

            logfunction(f'Optimized {len(new_structures)} structures at {method} level ({time_to_string(time.time()-t_start)})')

        if tg+1 != len(grouped_torsions):
            if n is not None and len(new_structures) > n:
                if mode == 0:
                    new_structures, energies = zip(*sorted(zip(new_structures, energies), key=lambda x: x[1]))
                    new_structures = new_structures[0:n]
                if mode == 1:
                    new_structures = most_diverse_conformers(n, new_structures, energies)
            logfunction(f'Kept the best {len(new_structures)} starting points for next rotation cluster')

        output_structures.extend(new_structures)
        starting_points = new_structures

    output_structures = np.array(output_structures)

    n_out = sum([t.n_fold for t in torsions])

    if len(new_structures) > n_out:
        if mode == 0:
            output_structures, energies = zip(*sorted(zip(output_structures, energies), key=lambda x: x[1]))
            output_structures = output_structures[0:n_out]
        if mode == 1:
            output_structures = most_diverse_conformers(n_out, output_structures, energies)
    logfunction(f'Selected the {"best" if mode == 0 else "most diverse"} {len(output_structures)} new structures ({time_to_string(time.time()-t_start_run)})')

    return output_structures

def most_diverse_conformers(n, structures, energies=None):
    '''
    Return the n most diverse structures from the set.
    Divides the structures in n subsets and:
    - If the enrgy list is given, chooses the
      one with the lowest energy from each.
    _ If it is not, picks the most diverse structures.
    '''
    structures = align_structures(structures)
    features = structures.reshape((structures.shape[0], structures.shape[1]*structures.shape[2]))

    kmeans = KMeans(n_clusters=n)
    kmeans.fit(features)

    if energies is not None:
        clusters = [[] for _ in range(n)]
        for coords, energy, c in zip(structures, energies, kmeans.labels_):
            clusters[c].append((coords, energy))

        output = []
        for group in clusters:
            sorted_s, _ = zip(*sorted(group, key=lambda x: x[1]))
            output.append(sorted_s[0])

    else: # afford the most distant structure from each cluster
        centers = kmeans.cluster_centers_.reshape((n, *structures.shape[1:3]))

        clusters = [[] for _ in range(n)]
        for coords, c in zip(structures, kmeans.labels_):
            clusters[c].append(coords)

        r = np.arange(len(clusters))
        output = []
        for cluster in clusters:
            cumdists = [np.sum(np.linalg.norm(centers[r!=c]-ref, axis=2)) for c, ref in enumerate(cluster)]
            furthest = cluster[cumdists.index(max(cumdists))]
            output.append(furthest)

    return np.array(output)

def _write_torsion_vmd(coords, atomnos, constrained_indexes, grouped_torsions):

    import os

    from tscode.utils import write_xyz

    with open('torsion_test.xyz', 'w') as f:
        write_xyz(coords, atomnos, f)

    path = os.path.join(os.getcwd(), 'torsion_test.vmd')
    with open(path, 'w') as f:
        s = ('display resetview\n' +
            'mol new {%s}\n' % (os.path.join(os.getcwd() + r'\torsion_test.xyz')) +
            'mol representation Lines 2\n' +
            'mol color ColorID 16\n'
            )

        for group, color in zip(grouped_torsions, (7,9,10,11,29,16)):
            for torsion in group:
                s += ('mol selection index %s\n' % (' '.join([str(i) for i in torsion.torsion[1:-1]])) +
                    'mol representation CPK 0.7 0.5 50 50\n' +
                f'mol color ColorID {color}\n' +
                    'mol material Transparent\n' +
                    'mol addrep top\n')

        for a, b in constrained_indexes:
            s += f'label add Bonds 0/{a} 0/{b}\n'


        f.write(s)

if __name__ == '__main__':

    import sys
    import time

    from cclib.io import ccread

    from hypermolecule_class import align_structures
    from tscode.utils import time_to_string, write_xyz


    def main(filename):
        # data = ccread(r'C:\Users\Nik\Desktop\complete\ts1.xyz')
        # constrained_indexes = np.array([[133, 168], [23, 151], [61, 130]])

        # data = ccread(r'C:\Users\Nik\Desktop\miller\pep\pep.xyz')
        constrained_indexes = np.array([])

        # data = ccread(r'C:\Users\Nik\Desktop\complete\acid_ensemble.xyz')
        # constrained_indexes = np.array([])

        # data = ccread(r'C:\Users\Nik\Desktop\Coding\TSCoDe\old\Resources\maleimide.xyz')
        # constrained_indexes = np.array([])    

        data = ccread(filename)
        assert data is not None, 'ops'
        # constrained_indexes = np.array([])

        t_start = time.time()
        new_structs = clustered_csearch(data.atomcoords[0],
                                        data.atomnos,
                                        constrained_indexes,
                                        # ff_opt=True,
                                        mode=1,
                                        n=20,
                                        )

        print(f'Run took {time_to_string(time.time()-t_start)}')

        with open('conf_test.xyz', 'w') as f:
            for s in align_structures(new_structs, indexes=constrained_indexes.ravel()):
                write_xyz(s, data.atomnos, f)

    import cProfile
    import os
    from pstats import Stats

    cProfile.run("main(sys.argv[1])", "output.dat")

    with open("output_time.txt", "w") as f:
        p = Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("output_cumtime.txt", "w") as f:
        p = Stats("output.dat", stream=f)
        p.sort_stats("cumtime").print_stats()

    # os.system("notepad output_time.txt")
    os.system("notepad output_cumtime.txt")