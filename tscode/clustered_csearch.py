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

from tscode.algebra import norm_of
from tscode.graph_manipulations import (findPaths, get_sp_n, is_amide_n,
                                        is_ester_o, is_sp_n, neighbors)
from tscode.hypermolecule_class import align_structures, graphize
from tscode.optimization_methods import optimize, prune_enantiomers
from tscode.parameters import nci_dict
from tscode.pt import pt
from tscode.python_functions import compenetration_check, prune_conformers
from tscode.settings import DEFAULT_FF_LEVELS, FF_CALC
from tscode.utils import (cartesian_product, get_double_bonds_indexes,
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
            # This has to be an intermolecular HB: rotate it
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

        if 1 in nums:
            return 6 # H-N, H-O
        
        if is_amide_n(self.i2, graph, mode=2) or (
           is_amide_n(self.i3, graph, mode=2)):
           # tertiary amides rotations are 2-fold
           return 2

        if (6 in nums) or (7 in nums) or (16 in nums): # if C, N or S atoms

            sp_n_i2 = get_sp_n(self.i2, graph)
            sp_n_i3 = get_sp_n(self.i3, graph)

            if 3 in (sp_n_i2, sp_n_i3): # Csp3-X, Nsp3-X, Ssulfone-X
                return 3

            # if 2 in (sp_n_i2, sp_n_i3):
            #     return 2 # Nsp2-O, S-O, P-O, sp2-sp2, sp2-sp, sp2-O, sp2-N

        # raise SystemExit(f'Torsion not defined:\nnum {nums[0]} - {nums[1]}\nnum {sp_n_i2} - {sp_n_i3}')
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


def _is_free(index, graph):
    '''
    Return True if the index specified
    satisfies all of the following:
    - Is not a sp2 carbonyl carbon atom
    - Is not the oxygen atom of an ester
    - Is not the nitrogen atom of a secondary amide (CONHR)

    '''
    if all((
            graph.nodes[index]['atomnos'] == 6,
            is_sp_n(index, graph, 2),
            8 in (graph.nodes[n]['atomnos'] for n in neighbors(graph, index))
          )):
        return False

    if is_amide_n(index, graph, mode=1):
        return False

    if is_ester_o(index, graph):
        return False

    return True

def _is_nondummy(i, root, graph) -> bool:
    '''
    Checks that a molecular rotation along the dihedral
    angle (*, root, i, *) is non-dummy, that is the atom
    at index i, in the direction opposite to the one leading
    to root, has different substituents. i.e. methyl and tBu
    rotations return False.

    Thought to eliminate methyl/tert-butyl-type rotations.
    '''

    if graph.nodes[i]['atomnos'] not in (6,7):
        return True
    # for now, we only discard rotations around carbon
    # and nitrogen atoms, like methyl/tert-butyl/triphenyl
    # and flat symmetrical rings like phenyl, N-pyrrolyl...

    G = deepcopy(graph)
    nb = neighbors(G, i)
    nb.remove(root)

    if len(nb) == 1:
        if len(neighbors(G, nb[0])) == 2:
            return False
    # if node i has two bonds only (one with root and one with a)
    # and the other atom (a) has two bonds only (one with i)
    # the rotation is considered dummy: some other rotation
    # will account for its freedom (i.e. alkynes, hydrogen bonds)

    for n in nb:
        G.remove_edge(i, n)

    subgraphs_nodes = [_set for _set in nx.connected_components(G)
                       if not root in _set]

    if len(subgraphs_nodes) == 1:

        for n in nb:
            G.add_edge(i, n)
        # restore i-neighbor bonds removed previously

        subgraph = G.subgraph(list(subgraphs_nodes[0])+[i])
        cycles = [l for l in nx.cycle_basis(subgraph, root=i) if len(l) != 1]
        # get all cycles that involve i, if any

        if cycles:
            # in this case, i is part of a cycle and root is exocyclic
            # and we will take care of this in a separate function
            return _is_cyclical_nondummy(G, cycles[0])   

        return True
        # if not, the torsion is likely to be rotable
        # (tetramethylguanidyl alanine C(β)-N bond)

    subgraphs = [nx.subgraph(G, s) for s in subgraphs_nodes]
    for sub in subgraphs[1:]:
        if not nx.is_isomorphic(subgraphs[0], sub,
                                node_match=lambda n1, n2: n1['atomnos'] == n2['atomnos']):
            return True
    # Care should be taken because chiral centers are not taken into account: a rotation 
    # involving an index where substituents only differ by stereochemistry, and where a 
    # rotation is not an element of symmetry of the subsystem, the rotation is discarded
    # even if it would be meaningful to keep it.

    return False

def _is_cyclical_nondummy(
                            # i, root, graph
                            G, cycle) -> bool:
    '''
    Extension of _is_nondummy for situations where
    root is exocyclic and i is endocyclic.
    
    Thought to reject symmetric phenyl ring rotations
    and similar.
    '''
    # G = deepcopy(graph)
    # G.remove_edge(i, root)

    # for subgraph_nodes in connected_components(G):
    #     if root in subgraph_nodes:
    #         break

    # G.remove_nodes_from(subgraph_nodes)
    # G.add_edge(i, root)
    # # getting a graph with root but without atoms
    # # connected to it, keeping the ring only

    # cycles = [l for l in nx.cycle_basis(G, root=i) if len(l) != 1]
    # # get all cycles that involve i

    # if len(cycles) > 1:
    #     return True
    # # if there is more than one, we have a bi/tricyclic
    # # compound and since rotation was not discarded
    # # earlier, we should rotate it.

    # cycle = cycles[0]
    if len(cycle) == 6:
    # if we have a six-membered ring

        symbols = [G.nodes[j]['atomnos'] for j in cycle]

        if all([s in (6,7) for s in symbols]):
        # made out only of carbon or nitrogen atoms

            if all([is_sp_n(j, G, 2) for j in cycle]):
            # and they are all sp2 (C w/3 neighbors or
            # N w/ 2 neighbors)

                j0, j1, j2, j3, j4, j5 = cycle

                if _is_isomorphic_fragment(G, index1=j0, forbidden_directions1=(j1,j5),
                                              index2=j4, forbidden_directions2=(j3,j5)) and (

                   _is_isomorphic_fragment(G, index1=j1, forbidden_directions1=(j0,j2),
                                              index2=j3, forbidden_directions2=(j2,j4))):

                    return False
                    # if ortho and meta substituents have identical graphs, the rotation is dummy
            
    return True
    # Anything else is considered non-dummy, and will rotate.

def _is_isomorphic_fragment(graph, index1, forbidden_directions1, 
                                  index2, forbidden_directions2):
    '''
    Returns true of two subgraphs built from the indices 
    provided are isomorphic, that is have the same bonding
    and the same atomic numbers (ignores stereochemistry)
    '''
    G = deepcopy(graph)

    for f1 in forbidden_directions1:
        G.remove_edge(index1, f1)

    for f2 in forbidden_directions2:
        G.remove_edge(index2, f2)
    # removing edges towards forbidden directions

    fragments = [nx.Graph(G.subgraph(s).edges) for s in nx.connected_components(G)]
    frag1 = [f for f in fragments if index1 in f][0]
    frag2 = [f for f in fragments if index2 in f][0]
    # using new graphs instead of subgraphs since we have to modify them

    atomnos_dict = nx.get_node_attributes(G, 'atomnos')
    nx.set_node_attributes(frag1, atomnos_dict, 'atomnos')
    nx.set_node_attributes(frag2, atomnos_dict, 'atomnos')
    # updating the old atomnos into new fragment graphs

    frag1.add_edge(index1, -1)
    frag2.add_edge(index2, -1)
    frag1.nodes[-1]['atomnos'] = -1
    frag2.nodes[-1]['atomnos'] = -1
    # adding a new labeled dummy node, to avoid
    # isomerism issues when comparing subgraphs

    if nx.is_isomorphic(frag1, frag2, node_match=lambda n1, n2: n1['atomnos'] == n2['atomnos']):
        return True
    return False
    # Care should be taken because chiral centers are not taken into account: a rotation 
    # involving an index where substituents only differ by stereochemistry, and where a 
    # rotation is not an element of symmetry of the subsystem, the rotation is discarded
    # even if it would be meaningful to keep it.

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
                    dist = norm_of(coord1-coord2)

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

def _get_torsions(graph, hydrogen_bonds, double_bonds):
    '''
    Returns list of Torsion objects
    '''
    allpaths = []
    for node in graph:
        allpaths.extend(findPaths(graph, node, 3))
    # get all possible continuous indexes quadruplets

    torsions, bond_torsions = [], []
    for path in allpaths:
        _, i2, i3, _ = path
        bt = tuple(sorted((i2, i3)))

        if (bt not in bond_torsions) and (bt not in double_bonds):
            t = Torsion(*path)

            if (not t.in_cycle(graph)) and t.is_rotable(graph, hydrogen_bonds):

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
    for n in range((l//max_size)+1, l+1):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(torsions_centers)

        output = [[] for _ in range(n)]
        for torsion, cluster in zip(torsions, kmeans.labels_):
            output[cluster].append(torsion)

        if max([len(group) for group in output]) <= max_size:
            break

    output = sorted(output, key=len)
    # largest groups last
    
    return output

def clustered_csearch(coords,
                        atomnos,
                        constrained_indexes=None,
                        keep_hb=False,
                        ff_opt=False,
                        n=10,
                        mode=1,
                        calc=None,
                        method=None,
                        title='test',
                        logfunction=print):
    '''
    n: number of structures to keep from each torsion cluster
    mode: 0 - keep the n lowest energy conformers
          1 - keep the n most diverse conformers

    keep_hb: whether to preserve the presence of current hydrogen bonds or not
    '''

    assert mode == 1 or ff_opt, 'Either leave mode=1 or turn on force field optimization'
    assert mode in (0,1), 'The mode keyword can only be 0 or 1'

    calc = FF_CALC if calc is None else calc
    method = DEFAULT_FF_LEVELS[calc] if method is None else method
    # Set default calculator attributes if user did not specify them

    constrained_indexes = np.array([]) if constrained_indexes is None else constrained_indexes
    t_start_run = time.perf_counter()

    graph = graphize(coords, atomnos)
    for i1, i2 in constrained_indexes:
        graph.add_edge(i1, i2)
    # build a molecular graph of the TS
    # that includes constrained indexes pairs
    
    if keep_hb and len(list(nx.connected_components(graph))) > 1:
        hydrogen_bonds = _get_hydrogen_bonds(coords, atomnos, graph)
        for hb in hydrogen_bonds:
            graph.add_edge(*hb)
    else:
        hydrogen_bonds = ()
    # get informations on the intra/intermolecular hydrogen
    # bonds that we should avoid disrupting

    double_bonds = get_double_bonds_indexes(coords, atomnos)
    # get all double bonds - do not rotate these
    
    torsions = _get_torsions(graph, hydrogen_bonds, double_bonds)
    # get all torsions that we should explore

    for t in torsions:
        t.sort_torsion(graph, constrained_indexes)
    # sort torsion indexes so that first index of each torsion
    # is the half that will move and is external to the TS

    if not torsions:
        logfunction(f'No rotable bonds found for {title}.')
        return np.array([coords])

    grouped_torsions = _group_torsions(coords,
                                       torsions,
                                       max_size=3 if ff_opt else 8)

    ############################################## LOG TORSIONS

    # with open('n_fold_log.txt', 'w') as f:
    #     for t in torsions:
    #         f.write(f'{t.torsion} - {t.n_fold}-fold\n')
    logfunction(f'Torsion list: (indexes : n-fold)')
    for t in torsions:
        logfunction('  - {:21s} : {}-fold'.format(str(t.torsion), t.n_fold))

    _write_torsion_vmd(coords, atomnos, constrained_indexes, grouped_torsions)

    ############################################## LOG TORSIONS

    logfunction(f'\n--> Clustered CSearch - mode {mode} ({"stability" if mode == 0 else "diversity"}) - ' +
                f'{len(torsions)} torsions in {len(grouped_torsions)} group{"s" if len(grouped_torsions) != 1 else ""} - ' +
                f'{[len(t) for t in grouped_torsions]}')
    
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
        for_comp = np.count_nonzero(~mask)

        logfunction(f'> Group {tg+1}/{len(grouped_torsions)} - {len(torsions_group)} bonds, ' +
                    f'{[t.n_fold for t in torsions_group]} n-folds, {len(starting_points)} ' + 
                    f'starting point{"s" if len(starting_points) > 1 else ""} = {candidates} conformers')
        logfunction(f'  {candidates} generated, {for_comp} removed for compenetration ({len(new_structures)} left)')

        energies = None
        if ff_opt:

            t_start = time.perf_counter()

            energies = np.zeros(new_structures.shape[0])
            for c, new_coords in enumerate(deepcopy(new_structures)):

                opt_coords, energy, success = optimize(new_coords,
                                                        atomnos,
                                                        calc,
                                                        method=method,
                                                        constrained_indexes=constrained_indexes)

                if success:
                    new_structures[c] = opt_coords
                    energies[c] = energy

                else:
                    energies[c] = np.inf

            logfunction(f'Optimized {len(new_structures)} structures at {method} level ({time_to_string(time.perf_counter()-t_start)})')

        if tg+1 != len(grouped_torsions):
            if n is not None and len(new_structures) > n:
                if mode == 0:
                    new_structures, energies = zip(*sorted(zip(new_structures, energies), key=lambda x: x[1]))
                    new_structures = new_structures[0:n]
                    tag = 'stable'
                if mode == 1:
                    new_structures = most_diverse_conformers(n, new_structures, atomnos, energies)
                    tag = 'diverse'
            logfunction(f'  Kept the most {tag} {len(new_structures)} starting points for next rotation cluster')

        output_structures.extend(new_structures)
        starting_points = new_structures

    output_structures = np.array(output_structures)

    n_out = sum([t.n_fold for t in torsions])

    if len(new_structures) > n_out:
        if mode == 0:
            output_structures, energies = zip(*sorted(zip(output_structures, energies), key=lambda x: x[1]))
            output_structures = output_structures[0:n_out]
            output_structures = np.array(output_structures)
        if mode == 1:
            output_structures = most_diverse_conformers(n_out, output_structures, atomnos, energies)
    logfunction(f'  Selected the {"best" if mode == 0 else "most diverse"} {len(output_structures)} new structures ({time_to_string(time.perf_counter()-t_start_run)})')

    return output_structures

def most_diverse_conformers(n, structures, atomnos, energies=None, force_enantiomer_pruning=False):
    '''
    Return the n most diverse structures from the set.
    First removes similar structures, then divides them in n subsets and:
    - If the enrgy list is given, chooses the
      one with the lowest energy from each.
    - If it is not, picks the most diverse structures.
    '''

    # print('Removing similar structures...', end='\r')
    # for k in (5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1):
    #     if 5*k < len(structures):
    #         structures, mask = prune_conformers(structures, atomnos, max_rmsd=2, max_delta=2, k=k)
    #         if energies is not None:
    #             energies = energies[mask]
    # Remove similar structures based on RMSD and max deviation

    if len(structures) < 3000 or force_enantiomer_pruning:
        print(f'Removing enantiomers...{" "*10}', end='\r')
        structures, mask = prune_enantiomers(structures, atomnos)
        if energies is not None:
            energies = energies[mask]
        # Remove enantiomers or structures similar under reflections
        # Skip if structures are too many (avoids stumping)
        
    if len(structures) <= n:
        return structures
    # if we already pruned enough structures to meet the requirement, return them

    print(f'Aligning structures...{" "*10}', end='\r')
    structures = align_structures(structures)
    features = structures.reshape((structures.shape[0], structures.shape[1]*structures.shape[2]))
    # reduce the dimensionality of the rest of the structure array to cluster them with KMeans

    kmeans = KMeans(n_clusters=n)
    kmeans.fit(features)
    # Generate and train the model

    if energies is not None:
        clusters = [[] for _ in range(n)]
        for coords, energy, c in zip(structures, energies, kmeans.labels_):
            clusters[c].append((coords, energy))

        output = []
        for group in clusters:
            sorted_s, _ = zip(*sorted(group, key=lambda x: x[1]))
            output.append(sorted_s[0])
    # if energies are given, pick the lowest energy structure from each cluster

    else:
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
    # if not, from each cluster yield the structure that is more distant from the other clusters

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