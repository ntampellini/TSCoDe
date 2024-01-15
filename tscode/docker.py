# coding=utf-8
'''

TSCODE: Transition State Conformational Docker
Copyright (C) 2021-2024 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''

import os

import numpy as np
from numba import njit

from python_functions import compenetration_check
from tscode.algebra import (align_vec_pair, internal_mean, norm, norm_of,
                            rot_mat_from_pointer, transform_coords, vec_mean,
                            vector_cartesian_product)
from tscode.graph_manipulations import get_phenyls, neighbors
from tscode.reactive_atoms_classes import get_atom_type
from tscode.utils import cartesian_product, write_xyz


class Docker:

    def __init__(self, embedder):
        self.embedder = embedder

    def get_anchors(self, mol, conf=0, aromatic=False):
        '''
        mol: a Hypermolecule object
        conf: the conformer index to be used

        returns a 3-tuple of arrays
        
        centers - absolute coordinates of points
        vectors - direction of center relative to its atom
        label - 0: electron-poor, 1: electron-rich, 2: aromatic
        '''

        centers, vectors, labels = [], [], []
        # initializing the lists

        for i, atom in enumerate(mol.atomnos):
            if atom in (7,8):
                # N and O atoms
                atom_cls = get_atom_type(mol.graph, i)()
                atom_cls.init(mol, i, update=True, orb_dim=1, conf=conf)
                for c, v in zip(atom_cls.center, atom_cls.orb_vecs):
                    centers.append(c)
                    vectors.append(v)
                    labels.append(1)

            elif atom == 1 and any((mol.graph.nodes[n]['atomnos'] in (7,8) for n in neighbors(mol.graph, i))):
                # protic H atoms
                atom_cls = get_atom_type(mol.graph, i)()
                atom_cls.init(mol, i, update=True, orb_dim=1, conf=conf)
                for c, v in zip(atom_cls.center, atom_cls.orb_vecs):
                    centers.append(c)
                    vectors.append(v)
                    labels.append(0)

        # looking for aromatic rings

        if aromatic and len(mol.atomnos) > 9:

            for coords_ in get_phenyls(mol.atomcoords[conf], mol.atomnos):

                mean = np.mean(coords_, axis=0)
                # getting the center of the ring

                vec = 1.8 * norm(np.cross(coords_[0]-coords_[1],
                                            coords_[1]-coords_[2]))
                # normal vector orthogonal to the ring
                # 1.8 A so that rings will stack at around 3.6 A

                centers.append(mean+vec)
                vectors.append(vec)
                labels.append(2)

                centers.append(mean-vec)
                vectors.append(-vec)
                labels.append(2)

        centers = np.array(centers)
        vectors = np.array(vectors)
        labels = np.array(labels)

        return centers, vectors, labels

    def test_anchors(self):

        lab_dict = {0:3, 1:4, 2:6}

        for i, mol in enumerate(self.embedder.objects):

            with open(f'anchor_test_{i}.xyz', 'w') as f:
                for c, coords in enumerate(mol.atomcoords):
                    centers, _, labels, = docker.get_anchors(mol, conf=c)

                    coords_ = np.concatenate((coords, centers))

                    atomnos_ = np.concatenate((mol.atomnos, [lab_dict[l] for l in labels]))

                    write_xyz(coords_, atomnos_, f)

            self.write_anchor_vmd(mol.atomnos, labels, xyz_name=f'anchor_test_{i}.xyz')

    def write_anchor_vmd(self, atomnos, labels, xyz_name='test.xyz'):
        '''
        Write VMD file with anchors highlighted.
        '''

        indices = [len(atomnos)+i for i in range(len(labels))]

        vmd_name = xyz_name.rstrip('.xyz') + '.vmd'
        # path = os.path.join(os.getcwd() + vmd_name)
        with open(vmd_name, 'w') as f:
            s = ('display resetview\n' +
                'mol new {%s}\n' % (os.path.join(os.getcwd(), xyz_name)) +
                'mol delrep top\n' +
                'mol selection all not index %s\n' % (' '.join([str(i) for i in indices])) +
                'mol addrep top\n' +
                'mol selection index %s\n' % (' '.join([str(i) for i in indices])) +
                'mol representation CPK 0.5 0.0 50 50\n' +
                'mol material Transparent\n' +
                'mol addrep top\n')

            f.write(s)

    def dock_structures(self):
        '''
        Sets the self.structures/self.atomnos attributes, docking the two mols
        '''

        assert len(self.embedder.objects) == 2

        mol1, mol2 = self.embedder.objects

        conf_number = [len(mol.atomcoords) for mol in self.embedder.objects]
        conf_indices = cartesian_product(*[np.array(range(i)) for i in conf_number])
        # get all conformation combinations indices

        self.atomnos = np.concatenate((mol1.atomnos, mol2.atomnos))
        structures = []

        for conf1, conf2 in conf_indices:

            coords1 = mol1.atomcoords[conf1]
            coords2 = mol2.atomcoords[conf2]

            anchors1 = self.get_anchors(mol1, conf=conf1)
            anchors2 = self.get_anchors(mol2, conf=conf2)

            structures.extend(_dock(coords1,
                                    coords2,
                                    anchors1,
                                    anchors2))

        self.structures = np.array(structures)

@njit(parallel=True)
def _dock(coords1, coords2, anchors1, anchors2):
    '''
    Return a (n, d1+d2, 3) shaped structure array where:
    - n is the number of non-compenetrating docked structures
    - d1 and d2 are coords1 and coords2 first dimension
    '''
    a1_centers, a1_vectors, a1_labels = anchors1
    a2_centers, a2_vectors, a2_labels = anchors2

    # getting pivots that connect each pair of anchors in a mol

    pivots1 = vector_cartesian_product(a1_centers, a1_centers)
    pivots2 = vector_cartesian_product(a2_centers, a2_centers)

    directions1 = internal_mean(vector_cartesian_product(a1_vectors,
                                                         a1_vectors))
    directions2 = internal_mean(vector_cartesian_product(a2_vectors,
                                                         a2_vectors))

    pivots1_signatures = vector_cartesian_product(a1_labels, a1_labels)
    pivots2_signatures = vector_cartesian_product(a2_labels, a2_labels)

    # pivots are paired if they respect this pairing table:
    # ep/ep, er/er and ar/er are discarded
    # 0 - electron-poor
    # 1 - electron-rich
    # 2 - aromatic

    signatures_mat = np.array([[0, 1, 1],
                               [1, 0, 0],
                               [1, 0, 1]], dtype=np.int32)

    ids = (len(coords1), len(coords2))
    structures = []
    coords1 = np.ascontiguousarray(coords1)
    coords2 = np.ascontiguousarray(coords2)

    for i1, (p1, s1) in enumerate(zip(pivots1, pivots1_signatures)):
        p1 = np.ascontiguousarray(p1)
        # print(f'pivot {(i1+1)*len(pivots2)}/{len(pivots1)*len(pivots2)}')
        for i2, (p2, s2) in enumerate(zip(pivots2, pivots2_signatures)):

            l1 = norm_of(p1[0]-p1[1])
            l2 = norm_of(p2[0]-p2[1])

            if l1 > 0.1 and l2 > 0.1 and np.abs(l1-l2) < 2:
            # do not pair pivots that are:
            # - starting and ending on the same atom
            # - too different in length (>2 A)
                
                if signatures_mat[s1[0][0]][s2[0][0]]:
                    if signatures_mat[s1[1][0]][s2[1][0]]:
                    # do not pair pivots that do not respect polarity

                        al_mat1 = align_vec_pair((p2[0]-p2[1], -directions2[i2]),
                                                 (p1[0]-p1[1], directions1[i1]))
                        # matrix that applied to coords1, aligns them to coords2
                        # p1 goes to p2
                        # direction1 goes to -direction2

                        step_rot_axis = al_mat1 @ (p1[0]-p1[1])
                        # vector connecting the ends of pivot1 after alignment
                    
                        for angle in np.arange(-90, 90, 20):
                            
                            step_mat1 = rot_mat_from_pointer(step_rot_axis, angle)

                            rot1 = step_mat1 @ al_mat1
                            pos1 = vec_mean(p2) - rot1 @ vec_mean(p1)

                            new_coords1 = transform_coords(coords1, rot1, pos1)
                            embedded_coords = np.concatenate((new_coords1, coords2))

                            if compenetration_check(embedded_coords, ids=ids, thresh=1.5):
                                # structures.append(embedded_coords)

                                ### DEBUG
                                embedded_coords = np.concatenate((embedded_coords,
                                                                  transform_coords(p1, rot1, pos1),
                                                                  p2))
                                structures.append(embedded_coords)

    return structures



if __name__ == '__main__':

    from time import perf_counter as t

    from tscode.embedder import Embedder

    # import os
    # os.chdir(r'C:\\Users\\ehrma\\Desktop\\cool_structs\\csearch')
    embedder = Embedder(r'C:\\Users\\ehrma\\Desktop\\cool_structs\\csearch\\dock_test.txt')
    docker = Docker(embedder)
    # docker.test_anchors()

    import cProfile
    from pstats import Stats

    def profile_run(name):

        datafile = f"TSCoDe_{name}_cProfile.dat"
        cProfile.run("docker.dock_structures()", datafile)

        with open(f"TSCoDe_{name}_cProfile_output_time.txt", "w") as f:
            p = Stats(datafile, stream=f)
            p.sort_stats("time").print_stats()

        with open(f"TSCoDe_{name}_cProfile_output_cumtime.txt", "w") as f:
            p = Stats(datafile, stream=f)
            p.sort_stats("cumtime").print_stats()

    # start = t()
    # docker.dock_structures()
    # print('Took %.3f s' % (t()-start))

    profile_run('dock')

    print(f'Found {len(docker.structures)} structs')

    with open(f'dock_test.xyz', 'w') as f:
        an = np.concatenate((docker.atomnos, [3 for _ in range(4)]))
        for c, coords in enumerate(docker.structures):
            write_xyz(coords, an, f)
