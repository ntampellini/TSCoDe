import numpy as np
from parameters import *
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
# TO DO: change with alignment function from linalg_tools
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

    def init(self, mol, i, update=False) -> None:
        '''
        '''
        symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[0][i]
        self.other = mol.atomcoords[0][neighbors_indexes][0]

        if update:

            try:
                key = symbol + ' ' + str(self)
                orb_dim = orb_dim_dict[key]
            except KeyError:
                orb_dim = 0.5*np.linalg.norm(self.coord - self.other)
                print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using half the bonding distance.')

            self.center = np.array([orb_dim * norm(self.coord - self.other) + self.coord])


class Sp2:
    '''
    '''
    def __init__(self):
        pass

    def __repr__(self):
        return 'sp2'

    def init(self, mol, i, update=False) -> None:
        '''
        '''
        symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[0][i]
        self.others = mol.atomcoords[0][neighbors_indexes]

        self.vectors = self.others - self.coord # vectors connecting reactive atom with neighbors
        self.orb_vec = norm(np.mean(np.array([np.cross(norm(self.vectors[0]), norm(self.vectors[1])),
                                              np.cross(norm(self.vectors[1]), norm(self.vectors[2])),
                                              np.cross(norm(self.vectors[2]), norm(self.vectors[0]))]), axis=0))

        self.alignment_matrix = rotation_matrix_from_vectors(np.array([1,0,0]), self.orb_vec)

        if update:

            try:
                key = symbol + ' ' + str(self)
                orb_dim = orb_dim_dict[key]
            except KeyError:
                orb_dim = np.linalg.norm(self.coord - self.others[0])
                print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using a full bonding distance.')
            
            self.center = np.array([self.orb_vec, -self.orb_vec]) * orb_dim      

            self.center += self.coord


class Sp: # BROKEN for sure, needs to fixed, eventually
    '''
    '''
    def __init__(self):
        pass

    def __repr__(self):
        return 'sp'

    def init(self, mol, i, update=False) -> None:
        '''
        '''
        symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[0][i]
        self.others = mol.atomcoords[0][neighbors_indexes]
       

class Sp3:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'sp3'

    def init(self, mol, i, update=False) -> None:
        '''
        '''

        symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[0][i]
        self.others = mol.atomcoords[0][neighbors_indexes]


        if len([atom for atom in self.neighbors_symbols if atom in ['O', 'N', 'Cl', 'Br', 'I']]) == 1: # if we can tell where is the leaving group
            self.leaving_group_coords = self.others[self.neighbors_symbols.index([atom for atom in self.neighbors_symbols if atom in ['O', 'Cl', 'Br', 'I']][0])]
        elif len([atom for atom in self.neighbors_symbols if atom not in ['H']]) == 1: # if no clear leaving group but we only have one atom != H
            self.leaving_group_coords = self.others[self.neighbors_symbols.index([atom for atom in self.neighbors_symbols if atom not in ['H']][0])]
        else: # if we cannot, ask user
            self.leaving_group_coords = self._set_leaving_group(mol.name, neighbors_indexes)

        self.orb_vec = self.coord - self.leaving_group_coords
        self.alignment_matrix = rotation_matrix_from_vectors(np.array([1,0,0]), self.orb_vec)

        if update:

            try:
                key = symbol + ' ' + str(self)
                orb_dim = orb_dim_dict[key]
            except KeyError:
                orb_dim = 1
                print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using 1 A.')

            self.center = np.array([orb_dim * norm(self.orb_vec) + self.coord])

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

    def init(self, mol, i, update=False) -> None:
        '''
        '''

        symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[0][i]
        self.others = mol.atomcoords[0][neighbors_indexes]

        self.vectors = self.others - self.coord # vectors connecting center to each of the two substituents
        self.alignment_matrix = rotation_matrix_from_vectors(np.array([1,0,0]), -np.mean(self.vectors, axis=0))

        if update:

            try:
                key = symbol + ' ' + str(self)
                orb_dim = orb_dim_dict[key]
            except KeyError:
                orb_dim = 1
                print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using 1 A.')

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

    def init(self, mol, i, update=False) -> None:
        '''
        '''

        symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[0][i]
        self.other = mol.atomcoords[0][neighbors_indexes][0]

        self.vector = self.other - self.coord # vector connecting center to substituent
        self.alignment_matrix = rotation_matrix_from_vectors(np.array([1,0,0]), -self.vector)

        if update:

            try:
                key = symbol + ' ' + str(self)
                orb_dim = orb_dim_dict[key]
            except KeyError:
                orb_dim = 1
                print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using 1 A.')

            neighbors_of_neighbor_indexes = list([(a, b) for a, b in mol.graph.adjacency()][neighbors_indexes[0]][1].keys())
            neighbors_of_neighbor_indexes.remove(i)
            neighbors_of_neighbor_indexes.remove(neighbors_indexes[0])

            self.vector = norm(self.vector)*orb_dim

            if len(neighbors_of_neighbor_indexes) == 2:
                a1 = mol.atomcoords[0][neighbors_of_neighbor_indexes[0]]
                a2 = mol.atomcoords[0][neighbors_of_neighbor_indexes[1]]
                pivot = norm(np.cross(a1 - self.coord, a2 - self.coord))

            else:
                r = np.random.rand()
                v0 = self.vector[0]
                v1 = self.vector[1]
                v2 = self.vector[2]
                pivot = np.array([r,r,-r*(v0+v1)/v2])
        
            self.center = np.array([rot_mat_from_pointer(pivot, angle) @ self.vector for angle in (120,240)])

            self.center += self.coord
            # two vectors defining the position of the two orbital lobes centers


class Imine:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'Imine'

    def init(self, mol, i, update=False) -> None:
        '''
        '''

        symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = list([(a, b) for a, b in mol.graph.adjacency()][i][1].keys())
        neighbors_indexes.remove(i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[0][i]
        self.others = mol.atomcoords[0][neighbors_indexes]

        self.vectors = self.others - self.coord # vector connecting center to substituent

        self.alignment_matrix = rotation_matrix_from_vectors(np.array([1,0,0]), -np.mean(self.vectors, axis=0))

        if update:

            try:
                key = symbol + ' ' + str(self)
                orb_dim = orb_dim_dict[key]
            except KeyError:
                orb_dim = 1
                print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using 1 A.')
        
            self.center = np.array([norm(-np.mean(self.vectors, axis=0))*orb_dim])

            self.center += self.coord
            # two vectors defining the position of the two orbital lobes centers


atom_type_dict = {
             'H1' : Single(),
             'C1' : Single(), # deprotonated terminal alkyne?
            #  'C2' : Sp(), # toroidal geometry (or carbene, or vinyl anion - #panik)
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

