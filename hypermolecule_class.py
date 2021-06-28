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

import os
import warnings
from copy import deepcopy
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import networkx as nx
from rmsd import kabsch
from cclib.io import ccread
from numpy.linalg import LinAlgError

from utils import pt
from reactive_atoms_classes import atom_type_dict

class CCReadError(Exception):
    '''
    Raised when CCRead cannot read
    the provided filename.
    '''

def align_structures(structures, indexes=None):
    '''
    Aligns molecules of a structure array (shape is (n_structures, n_atoms, 3))
    to the first one, based on the indexes. If not provided, all atoms are used
    to get the best alignment. Return is the aligned array.

    '''

    reference = structures[0]
    targets = structures[1:]
    if isinstance(indexes, list):
        indexes = np.array(indexes)
    indexes = indexes.ravel()

    indexes = slice(0,len(reference)) if indexes is None else indexes

    reference -= np.mean(reference[indexes], axis=0)
    for t in range(len(targets)):
        targets[t] -= np.mean(targets[t,indexes], axis=0)

    output = np.zeros(structures.shape)
    output[0] = reference

    for t, target in enumerate(targets):

        try:
            matrix = kabsch(reference[indexes], target[indexes])

        except LinAlgError:
        # it is actually possible for the kabsch alg not to converge
            matrix = np.eye(3)
        
        output[t+1] = np.array([matrix @ vector for vector in target])

    return output

def graphize(coords, atomnos):
    '''
    :params coords: atomic coordinates as 3D vectors
    :params atomnos: atomic numbers as a list
    :return connectivity graph
    '''
    def d_min(e1, e2):
        return 1.2 * (pt[e1].covalent_radius + pt[e2].covalent_radius)
        # return 0.2 + (pt[e1].covalent_radius + pt[e2].covalent_radius)
    # if this is somewhat prone to bugs, this might help https://cccbdb.nist.gov/calcbondcomp1x.asp

    matrix = np.zeros((len(coords),len(coords)))
    for i in range(len(coords)):
        for j in range(i,len(coords)):
            if np.linalg.norm(coords[i]-coords[j]) < d_min(atomnos[i], atomnos[j]):
                matrix[i][j] = 1

    return nx.from_numpy_matrix(matrix)

class Hypermolecule:
    '''
    Molecule class to be used within TSCoDe.
    '''

    def __repr__(self):
        return self.rootname + f' {[str(atom) for atom in self.reactive_atoms_classes_dict.values()]}, ID = {id(self)}'

    def __init__(self, filename, reactive_atoms, debug=False):
        '''
        Initializing class properties: reading conformational ensemble file, aligning
        conformers to first and centering them in origin.

        :params filename:           Input file name. Can be anything, .xyz preferred
        :params reactive_atoms:     Index of atoms that will link during the desired reaction.
                                    May be either int or list of int.
        '''

        if not os.path.isfile(filename):
            if filename.endswith('.xyz'):
                raise SyntaxError((f'Molecule {filename} cannot be read. Please check your syntax.'))

            raise SyntaxError((f'The program is trying to read something that is not a valid molecule input ({filename}). ' +
                                'If this looks like a keyword, it is probably faulted by a syntax error.'))

        self.rootname = filename.split('.')[0]
        self.name = filename
        self.debug = debug

        if reactive_atoms == ():
            reactive_atoms = self._set_reactive_atoms(filename)

        ccread_object = ccread(filename)
        if ccread_object is None:
            raise CCReadError(f'Cannot read file {filename}')

        coordinates = np.array(ccread_object.atomcoords)

        # if coordinates.shape[0] > 5:
        #     coordinates = coordinates[0:5]
        # # Do not keep more than 5 conformations

        self.reactive_indexes = np.array(reactive_atoms)
        # alignment_indexes = self._alignment_indexes(ccread_object.atomcoords[0], ccread_object.atomnos, self.reactive_indexes)
        
        self.atomnos = ccread_object.atomnos
        self.position = np.array([0,0,0], dtype=float)  # used in Docker class
        self.rotation = np.identity(3)                  # used in Docker class - rotation matrix

        assert all([len(coordinates[i])==len(coordinates[0]) for i in range(1, len(coordinates))])     # Checking that ensemble has constant length
        if self.debug:
            print(f'DEBUG--> Initializing object {filename}\nDEBUG--> Found {len(coordinates)} structures with {len(coordinates[0])} atoms')


        self.centroid = np.sum(np.sum(coordinates, axis=0), axis=0) / (len(coordinates) * len(coordinates[0]))

        if self.debug:
            print('DEBUG--> Centroid was', self.centroid)

        self.atomcoords = coordinates - self.centroid
        self.graph = graphize(self.atomcoords[0], self.atomnos)
        # show_graph(self)
        self._inspect_reactive_atoms() # sets reactive atoms properties to rotate the ensemble correctly afterwards

        self.atomcoords = align_structures(self.atomcoords, self.get_alignment_indexes())

        for index, reactive_atom in self.reactive_atoms_classes_dict.items():   
            reactive_atom.init(self, index, update=True)
            # update properties into reactive_atom class

        self.atoms = np.array([atom for structure in self.atomcoords for atom in structure])       # single list with all atomic positions
        
        if self.debug:
            print(f'DEBUG--> Total of {len(self.atoms)} atoms')

        # self._compute_hypermolecule()

    def _set_reactive_atoms(self, filename):
        '''
        Manually set the molecule reactive atoms from the ASE GUI, imposing
        constraints on the desired atoms.

        '''
        from ase import Atoms
        from ase.gui.gui import GUI
        from ase.gui.images import Images

        data = ccread(filename)
        coords = data.atomcoords[0]
        labels = ''.join([pt[i].symbol for i in data.atomnos])

        atoms = Atoms(labels, positions=coords)

        while atoms.constraints == []:
            print(('\nPlease, manually select the reactive atom(s) for molecule %s.'
                    '\nRotate with right click and select atoms by clicking. Multiple selections can be done by Ctrl+Click.'
                    '\nWith desired atom(s) selected, go to Tools -> Constraints -> Constrain, then close the GUI.') % (filename))

            GUI(images=Images([atoms]), show_bonds=True).run()

        return list(atoms.constraints[0].get_indices())
    
    def get_alignment_indexes(self):
        '''
        Return the indexes to align the molecule to, given a list of
        atoms that should be reacting. List is composed by reactive atoms
        plus adjacent atoms.
        :param coords: coordinates of a single molecule
        :param reactive atoms: int or list of ints
        :return: list of indexes
        '''

        indexes = set()

        for atom in self.reactive_indexes:
            indexes |= set(list([(a, b) for a, b in self.graph.adjacency()][atom][1].keys()))
        if self.debug: print('DEBUG--> Alignment indexes are', list(indexes))
        return list(indexes)

    def _inspect_reactive_atoms(self):
        '''
        Control the type of reactive atoms and sets the class attribute self.reactive_atoms_classes_dict
        '''
        self.reactive_atoms_classes_dict = {}

        for index in self.reactive_indexes:
            symbol = pt[self.atomnos[index]].symbol

            neighbors_indexes = list([(a, b) for a, b in self.graph.adjacency()][index][1].keys())
            neighbors_indexes.remove(index)

            atom_type = deepcopy(atom_type_dict[symbol + str(len(neighbors_indexes))])

            atom_type.init(self, index)
            # setting the reactive_atom class type

            self.reactive_atoms_classes_dict[index] = atom_type

            if self.debug: print(f'DEBUG--> Reactive atom {index+1} is a {symbol} atom of {atom_type} type. It is bonded to {len(neighbors_indexes)} atom(s): {atom_type.neighbors_symbols}')
            # understanding the type of reactive atom in order to align the ensemble correctly and build the correct pseudo-orbitals

    def _scale_orbs(self, value):
        '''
        Scale each orbital dimension according to value.
        '''
        for index, atom in self.reactive_atoms_classes_dict.items():
            orb_dim = np.linalg.norm(atom.center[0]-atom.coord)
            atom.init(self, index, update=True, orb_dim=orb_dim*value)

    def _compute_hypermolecule(self):
        '''
        '''

        self.energies = [0 for _ in self.atomcoords]

        self.hypermolecule_atomnos = []
        clusters = {i:{} for i in range(len((self.atomnos)))}  # {atom_index:{cluster_number:[position,times_found]}}
        for i, atom_number in enumerate(self.atomnos):
            atoms_arrangement = [conformer[i] for conformer in self.atomcoords]
            cluster_number = 0
            clusters[i][cluster_number] = [atoms_arrangement[0], 1]  # first structure has rel E = 0 so its weight is surely 1
            self.hypermolecule_atomnos.append(atom_number)
            radii = pt[atom_number].covalent_radius
            for j, atom in enumerate(atoms_arrangement[1:]):

                weight = np.exp(-self.energies[j+1] * 503.2475342795285 / self.T)
                # print(f'Atom {i} in conf {j+1} weight is {weight} - rel. E was {self.energies[j+1]}')

                for cluster_number, reference in deepcopy(clusters[i]).items():
                    if np.linalg.norm(atom - reference[0]) < radii:
                        clusters[i][cluster_number][1] += weight
                    else:
                        clusters[i][max(clusters[i].keys())+1] = [atom, weight]
                        self.hypermolecule_atomnos.append(atom_number)

        self.weights = [[] for _ in range(len(self.atomnos))]
        self.hypermolecule = []

        for i in range(len(self.atomnos)):
            for _, data in clusters[i].items():
                self.weights[i].append(data[1])
                self.hypermolecule.append(data[0])

        def flatten(array):
            out = []
            def rec(l):
                for e in l:
                    if type(e) in [list, np.ndarray]:
                        rec(e)
                    else:
                        out.append(float(e))
            rec(array)
            return out

        self.hypermolecule = np.asarray(self.hypermolecule)
        self.weights = np.array(self.weights).flatten()
        self.weights = np.array([weights / np.sum(weights) for weights in self.weights])
        self.weights = flatten(self.weights)

        self.dimensions = (max([coord[0] for coord in self.hypermolecule]) - min([coord[0] for coord in self.hypermolecule]),
                            max([coord[1] for coord in self.hypermolecule]) - min([coord[1] for coord in self.hypermolecule]),
                            max([coord[2] for coord in self.hypermolecule]) - min([coord[2] for coord in self.hypermolecule]))

    def write_hypermolecule(self):
        '''
        '''

        hyp_name = self.rootname + '_hypermolecule.xyz'
        with open(hyp_name, 'w') as f:
            f.write(str(sum([len(atom.center) for atom in self.reactive_atoms_classes_dict.values()]) + len(self.hypermolecule)))
            f.write(f'\nHypermolecule originated from {self.rootname}\n')
            orbs =np.vstack([atom_type.center for atom_type in self.reactive_atoms_classes_dict.values()]).ravel()
            orbs = orbs.reshape((int(len(orbs)/3), 3))
            for i, atom in enumerate(self.hypermolecule):
                f.write('%-5s %-8s %-8s %-8s\n' % (pt[self.hypermolecule_atomnos[i]].symbol, round(atom[0], 6), round(atom[1], 6), round(atom[2], 6)))
            for orb in orbs:
                f.write('%-5s %-8s %-8s %-8s\n' % ('D', round(orb[0], 6), round(orb[1], 6), round(orb[2], 6)))

        
        print('Written .xyz orbital file -', hyp_name)

if __name__ == '__main__':
    # TESTING PURPOSES

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    test = {

        1 : ('Resources/indole_ensemble.xyz', 6),
        2 : ('Resources/amine_ensemble.xyz', 10),
        3 : ('Resources/dienamine_ensemble.xyz', 6),
        4 : ('Resources/flex_ensemble.xyz', [3, 5]),
        5 : ('Resources/flex_ensemble.xyz', None),

        6 : ('Resources/MeOH_ensemble.xyz', 1),
        7 : ('Resources/CH3Br_ensemble.xyz', (0,)),
        8 : ('Resources/tax.xyz', None),
        9 : ('Resources/diene.xyz', (2,7)),
        10 : ('Resources/dienophile.xyz', (3,5)),
        11 : ('Resources/MeOH_ensemble.xyz', (1,5)),
        12 : ('Resources/ketone_ensemble.xyz', 5),
        13 : ('Resources/NOH.xyz', (0,1)),

        14 : ('Resources/acid_ensemble.xyz', (3,25)),
        15 : ('Resources/dienamine_ensemble.xyz', (6,23)),
        16 : ('Resources/maleimide.xyz', (0,5)),
        17 : ('Resources/C2H4.xyz', (0,3)),
        18 : ('Resources/tropone.xyz', ()),
        19 : ('Resources/C.xyz', (0,4,8))


            }

    for i in (19,):
        t = Hypermolecule(test[i][0], test[i][1])

        # t.reactive_atoms_classes_dict.values()[0].init(t, t.reactive_indexes[0], update=True, orb_dim=1)
        # t.reactive_atoms_classes_dict.values()[1].init(t, t.reactive_indexes[1], update=True, orb_dim=1.5)
        # t._update_orbs()

        t._compute_hypermolecule()
        t.write_hypermolecule()