import numpy as np
from parameters import *
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
# TODO: change with alignment function from linalg_tools
from linalg_tools import *
from periodictable import core, covalent_radius

pt = core.PeriodicTable(table="H=1")
covalent_radius.init(pt)


class Single:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'Single Bond'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.other = mol.atomcoords[atomcoords_index][neighbors_indexes][0]

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + str(self)
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = np.linalg.norm(self.coord - self.other)
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using the bonding distance ({round(orb_dim, 3)} A).')

            self.orb_vecs = np.array([norm(self.coord - self.other)])
            self.center = np.array([orb_dim * self.orb_vecs[0] + self.coord])


class Sp2:
    '''
    '''
    def __init__(self):
        pass

    def __repr__(self):
        return f'sp2'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.others = mol.atomcoords[atomcoords_index][neighbors_indexes]

        self.vectors = self.others - self.coord # vectors connecting reactive atom with neighbors
        self.orb_vec = norm(np.mean(np.array([np.cross(norm(self.vectors[0]), norm(self.vectors[1])),
                                              np.cross(norm(self.vectors[1]), norm(self.vectors[2])),
                                              np.cross(norm(self.vectors[2]), norm(self.vectors[0]))]), axis=0))

        self.alignment_matrix = rotation_matrix_from_vectors(np.array([1,0,0]), self.orb_vec)

        self.orb_vecs = np.vstack((self.orb_vec, -self.orb_vec))

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + str(self)
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = orb_dim_dict['Fallback']
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')
            
            self.center = self.orb_vecs * orb_dim      

            self.center += self.coord


class Sp3:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'sp3'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.others = mol.atomcoords[atomcoords_index][neighbors_indexes]


        if len([atom for atom in self.neighbors_symbols if atom in ['O', 'N', 'Cl', 'Br', 'I']]) == 1: # if we can tell where is the leaving group
            self.leaving_group_coords = self.others[self.neighbors_symbols.index([atom for atom in self.neighbors_symbols if atom in ['O', 'Cl', 'Br', 'I']][0])]
        elif len([atom for atom in self.neighbors_symbols if atom not in ['H']]) == 1: # if no clear leaving group but we only have one atom != H
            self.leaving_group_coords = self.others[self.neighbors_symbols.index([atom for atom in self.neighbors_symbols if atom not in ['H']][0])]
        else: # if we cannot, ask user
            self.leaving_group_coords = self._set_leaving_group(mol.name, neighbors_indexes)

        self.orb_vecs = np.array([self.coord - self.leaving_group_coords])
        self.alignment_matrix = rotation_matrix_from_vectors(np.array([1,0,0]), self.orb_vecs[0])

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + str(self)
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = orb_dim_dict['Fallback']
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')

            self.center = np.array([orb_dim * norm(self.orb_vecs[0]) + self.coord])

    def _set_leaving_group(self, filename, neighbors_indexes):
        '''
        Manually set the molecule leaving group from the ASE GUI, imposing
        a constraint on the desired atom.

        '''
        from ase import Atoms
        from ase.gui.gui import GUI
        from ase.gui.images import Images

        from cclib.io import ccread
        from periodictable import core
        pt_s = core.PeriodicTable(table='s')


        data = ccread(filename)
        coords = data.atomcoords[0]
        labels = ''.join([pt_s[i].symbol for i in data.atomnos])

        atoms = Atoms(labels, positions=coords)

        while True:
            print(('\nPlease, manually select the leaving group atom for molecule %s.'
                '\nRotate with right click and select atoms by clicking.'
                '\nThen go to Tools -> Constraints -> Constrain, and close the GUI.') % (filename))

            GUI(images=Images([atoms]), show_bonds=True).run()
            
            if atoms.constraints != []:
                if len(list(atoms.constraints[0].get_indices())) == 1:
                    if list(atoms.constraints[0].get_indices())[0] in neighbors_indexes:
                        break
                    else:
                        print('\nSeems that the atom you selected is not bonded to the reactive center or is the reactive atom itself.\nThis is probably an error, please try again.')
                        atoms.constraints = []
                else:
                    print('\nPlease only select one leaving group atom.')
                    atoms.constraints = []

        leaving_group_index = list(atoms.constraints[0].get_indices())[0]
        leaving_group_coords = self.others[neighbors_indexes.index(leaving_group_index)]

        # print(f'Check: leaving group index is {leaving_group_index}, d =', np.linalg.norm(leaving_group_coords - self.coord))

        return leaving_group_coords


class Ether:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'Ether'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.others = mol.atomcoords[atomcoords_index][neighbors_indexes]

        self.vectors = self.others - self.coord # vectors connecting center to each of the two substituents
        self.orb_vecs = np.array([norm(v) for v in self.vectors])
        self.alignment_matrix = rotation_matrix_from_vectors(np.array([1,0,0]), -np.mean(self.orb_vecs, axis=0))

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + str(self)
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = orb_dim_dict['Fallback']
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')

            self.vectors = orb_dim * np.array([norm(v) for v in self.vectors]) # making both vectors a fixed, defined length

            orb_mat = rot_mat_from_pointer(np.mean(self.vectors, axis=0), 90) @ rot_mat_from_pointer(norm(np.cross(self.vectors[0], self.vectors[1])), 180)

            self.center = np.array([orb_mat @ v for v in self.vectors])
            
            self.center += self.coord
            # two vectors defining the position of the two orbital lobes centers


class Ketone:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'Ketone'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.other = mol.atomcoords[atomcoords_index][neighbors_indexes][0]

        self.vector = self.other - self.coord # vector connecting center to substituent
        self.alignment_matrix = rotation_matrix_from_vectors(np.array([1,0,0]), -self.vector)

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + str(self)
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = orb_dim_dict['Fallback']
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')

            neighbors_of_neighbor_indexes = list([(a, b) for a, b in mol.graph.adjacency()][neighbors_indexes[0]][1].keys())
            neighbors_of_neighbor_indexes.remove(i)
            neighbors_of_neighbor_indexes.remove(neighbors_indexes[0])

            self.vector = norm(self.vector)*orb_dim

            if len(neighbors_of_neighbor_indexes) == 2:
            # if it is a normal ketone, orbitals must be coplanar with
            # atoms connecting to ketone C atom
                a1 = mol.atomcoords[atomcoords_index][neighbors_of_neighbor_indexes[0]]
                a2 = mol.atomcoords[atomcoords_index][neighbors_of_neighbor_indexes[1]]
                pivot = norm(np.cross(a1 - self.coord, a2 - self.coord))

            else:
            # otherwise, it can be an alkoxide/ketene and they can lie
            # on a random plane.
                r = np.random.rand()
                v0 = self.vector[0]
                v1 = self.vector[1]
                v2 = self.vector[2]
                pivot = np.array([r,r,-r*(v0+v1)/v2])
        
            self.center = np.array([rot_mat_from_pointer(pivot, angle) @ self.vector for angle in (120,240)])

            self.orb_vecs = np.array([norm(center) for center in self.center])
            # unit vectors connecting reactive atom coord with orbital centers

            self.center += self.coord
            # two vectors defining the position of the two orbital lobes centers


class Imine:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'Imine'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.others = mol.atomcoords[atomcoords_index][neighbors_indexes]

        self.vectors = self.others - self.coord # vector connecting center to substituent

        self.alignment_matrix = rotation_matrix_from_vectors(np.array([1,0,0]), -np.mean(self.vectors, axis=0))

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + str(self)
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = orb_dim_dict['Fallback']
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')
        
            self.center = self.center = np.array([-norm(np.mean([norm(v) for v in self.vectors], axis=0))*orb_dim])
            self.orb_vecs = np.array([norm(center) for center in self.center])
            self.center += self.coord
            # two vectors defining the position of the two orbital lobes centers


class Sp_or_carbene:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return self.type

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)
        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]

        self.coord = mol.atomcoords[atomcoords_index][i]
        self.others = mol.atomcoords[atomcoords_index][neighbors_indexes]

        self.vectors = self.others - self.coord # vector connecting center to substituent

        self.alignment_matrix = rotation_matrix_from_vectors(np.array([1,0,0]), -np.mean(self.vectors, axis=0))

        angle = vec_angle(norm(self.others[0] - self.coord),
                          norm(self.others[1] - self.coord))
        
        if np.abs(angle - 180) < 5:
            self.type = 'sp'
        else:
            self.type = 'bent carbene'

        self.allene = False
        if self.type == 'sp' and all([s == 'C' for s in self.neighbors_symbols]):

            neighbors_of_neighbors_indexes = (list([(a, b) for a, b in mol.graph.adjacency()][neighbors_indexes[0]][1].keys()),
                                              list([(a, b) for a, b in mol.graph.adjacency()][neighbors_indexes[1]][1].keys()))

            neighbors_of_neighbors_indexes[0].remove(i)
            neighbors_of_neighbors_indexes[1].remove(i)
            neighbors_of_neighbors_indexes[0].remove(neighbors_indexes[0])
            neighbors_of_neighbors_indexes[1].remove(neighbors_indexes[1])

            if (len(side1) == len(side2) == 2 for side1, side2 in neighbors_of_neighbors_indexes):
                self.allene = True

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + self.type
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = orb_dim_dict['Fallback']
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')
        
            if self.type == 'sp': # TODO would be cool to do better for allenes
                
                r = np.random.rand()
                v0 = self.vectors[0][0]
                v1 = self.vectors[0][1]
                v2 = self.vectors[0][2]
                pivot1 = np.array([r,r,-r*(v0+v1)/v2])

                if self.allene:
                    # if we have an allene, the generated pivot1 is aligned to
                    # one allene substituent so that the resulting positions
                    # for the four orbital centers make chemical sense.

                    allene_axis = norm(self.others[0] - self.others[1])
                    # versor connecting reactive atom neighbors
                    
                    ref = (mol.atomcoords[atomcoords_index][neighbors_of_neighbors_indexes[0][0]] -
                           mol.atomcoords[atomcoords_index][neighbors_indexes[0]])

                    ref = ref - ref @ allene_axis * allene_axis
                    # projection of ref orthogonal to allene_axis (vector rejection)

                    pivot1 = R.align_vectors((allene_axis, ref),
                                             (allene_axis, pivot1))[0].as_matrix() @ pivot1

                pivot2 = norm(np.cross(pivot1, self.vectors[0]))
                        
                self.center = np.array([rot_mat_from_pointer(pivot2, 90) @
                                        rot_mat_from_pointer(pivot1, angle) @
                                        norm(self.vectors[0]) for angle in (0, 90, 180, 270)]) * orb_dim

                self.orb_vecs = np.array([norm(center) for center in self.center])
                # unit vectors connecting reactive atom coord with orbital centers

                self.center += self.coord
                # four vectors defining the position of the four orbital lobes centers



            else: # bent carbene case: three centers, sp2+p
                
                self.center = np.array([-norm(np.mean([norm(v) for v in self.vectors], axis=0))*orb_dim])
                self.orb_vecs = np.array([norm(center) for center in self.center])
                # one sp2 center first

                p_vec = np.cross(norm(self.vectors[0]), norm(self.vectors[1]))
                p_vecs = np.array([norm(p_vec)*orb_dim, -norm(p_vec)*orb_dim])
                p_vecs_norm = np.array([norm(p) for p in p_vecs])
                # adding two p centers

                self.orb_vecs = np.concatenate((self.orb_vecs, p_vecs_norm))
                self.center = np.concatenate((self.center, p_vecs))                

                self.center += self.coord
                # three vectors defining the position of the two p lobes and main sp2 lobe centers


atom_type_dict = {
             'H1' : Single(),
             'C1' : Single(), # deprotonated terminal alkyne?
             'C2' : Sp_or_carbene(), # sp if straight, carbene if bent
             'C3' : Sp2(), # double ball
             'C4' : Sp3(), # one ball: on the back of weakest bond. If can't tell which is which, we ask user
             'N1' : Single(),
             'N2' : Imine(), # one ball on free side
             'N3' : Sp2(), # or one ball on free side?
             'N4' : Sp3(),
             'O1' : Ketone(), # two balls 120° apart. Also for alkoxides, good enough
             'O2' : Ether(), # or alcohol, two balls 109,5° apart
             'S1' : Ketone(),
             'S2' : Ether(),
             'F1' : Single(),
             'Cl1': Single(),
             'Br1': Single(),
             'I1' : Single(),
             }

