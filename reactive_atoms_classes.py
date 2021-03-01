import numpy as np
from parameters import *
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

def norm(vec):
    return vec / np.linalg.norm(vec)

class Single:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'Single Bond'

    def prop(self, reactive_atom_coords:np.array, bonded_atom_coords:np.array, symbol, filename=None, neighbors_indexes=None, neighbors_symbols=None) -> None:
        '''
        Adding the properties of the two atom considered

        :params reactive_atom_coords: coordinates of the reactive atom
        :params bonded_atom_coords: coordinates of the atom bonded at the reactive object
        :return: None
        '''
        self.neighbors_symbols = neighbors_symbols if neighbors_symbols != None else self.neighbors_symbols
        # print(bonded_atom_coords)
        if len(bonded_atom_coords) == 1:
            bonded_atom_coords = bonded_atom_coords[0]
        assert reactive_atom_coords.shape == (3,)
        assert bonded_atom_coords.shape == (3,)

        self.coord = reactive_atom_coords
        self.others = bonded_atom_coords
        self.orb_vec = self.coord - self.others

        try:
            key = symbol + ' ' + str(self)
            orb_dim = orb_dim_dict[key]
        except:
            orb_dim = 0.5*np.linalg.norm(self.coord - self.others)
            print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using half the bonding distance.')

        self.center = np.array([orb_dim * norm(self.coord - self.others) + self.coord])
        return None

class Sp2:
    '''
    '''
    def __init__(self):
        pass

    def __repr__(self):
        return 'sp2'

    def prop(self, reactive_atom_coords:np.array, bonded_atom_coords:np.array, symbol, filename=None, neighbors_indexes=None, neighbors_symbols=None) -> None:
        '''
        Adding the properties of the two atom considered

        :params reactive_atom_coords: coordinates of the reactive atom
        :params bonded_atom_coords: coordinates of the atom bonded at the reactive object
        :return: None
        '''
        self.neighbors_symbols = neighbors_symbols if neighbors_symbols != None else self.neighbors_symbols
        self.coord = reactive_atom_coords
        self.others = bonded_atom_coords

        self.vectors = self.others - self.coord # vectors connecting reactive atom with neighbors
        self.orb_vec = norm(np.mean(np.array([np.cross(norm(self.vectors[0]), norm(self.vectors[1])),
                                              np.cross(norm(self.vectors[1]), norm(self.vectors[2])),
                                              np.cross(norm(self.vectors[2]), norm(self.vectors[0]))]), axis=0))

        try:
            key = symbol + ' ' + str(self)
            orb_dim = orb_dim_dict[key]
        except:
            orb_dim = np.linalg.norm(self.coord - self.others[0])
            print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using a full bonding distance.')
        
        self.center = np.array([self.orb_vec, -self.orb_vec]) * orb_dim

        self.alignment_matrix = R.align_vectors(np.array([[1,0,0]]), np.array([self.center[0]]))[0].as_matrix()

        self.center += self.coord

        return None


class Sp: # BROKEN for sure, needs to fixed, eventually
    '''
    '''
    def __init__(self):
        pass

    def __repr__(self):
        return 'sp'

    def prop(self, reactive_atom_coords:np.array, bonded_atom_coords:np.array, symbol, filename=None, neighbors_indexes=None, neighbors_symbols=None) -> None:
        '''
        Adding the properties of the two atom considered

        :params reactive_atom_coords: coordinates of the reactive atom
        :params bonded_atom_coords: coordinates of the atom bonded at the reactive object
        :return: None
        '''
        self.neighbors_symbols = neighbors_symbols if neighbors_symbols != None else self.neighbors_symbols
        self.coord = reactive_atom_coords
        self.others = bonded_atom_coords
        self.vectors = self.others - self.coord # vectors connecting reactive atom with neighbors
       
        return None


class Sp3:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'sp3'

    def prop(self, reactive_atom_coords:np.array, bonded_atom_coords:np.array, symbol, filename=None, neighbors_indexes=None, neighbors_symbols=None) -> None:
        '''
        Adding the properties of the atom considered

        :params reactive_atom_coords: coordinates of the reactive atom
        :params bonded_atom_coords: coordinates of the atom bonded at the reactive atom
        :return: None
        '''
        assert reactive_atom_coords.shape == (3,)
        assert bonded_atom_coords.shape[1] == 3

        self.neighbors_symbols = neighbors_symbols if neighbors_symbols != None else self.neighbors_symbols
        self.coord = reactive_atom_coords
        self.others = bonded_atom_coords

        if len([atom for atom in self.neighbors_symbols if atom in ['O', 'N', 'Cl', 'Br', 'I']]) == 1: # if we can tell where is the leaving group
            self.leaving_group_coords = self.others[self.neighbors_symbols.index([atom for atom in self.neighbors_symbols if atom in ['O', 'Cl', 'Br', 'I']][0])]
        else: # if we cannot, ask user if we have not already
            if not filename == None:
                self.leaving_group_coords = self._set_leaving_group(filename, neighbors_indexes)

        self.orb_vec = self.coord - self.leaving_group_coords

        try:
            key = symbol + ' ' + str(self)
            orb_dim = orb_dim_dict[key]
        except:
            orb_dim = 1
            print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using 1 A.')

        self.center = np.array([orb_dim * norm(self.orb_vec) + self.coord])

        self.alignment_matrix = R.align_vectors(np.array([[1,0,0]]), np.array([self.center[0]]))[0].as_matrix()

        return None

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
                '\nThen go to Tools -> Constraints -> Constrain, and close the GUI.'
                '\nBond view toggle with Ctrl+B\n') % (filename))

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

    def prop(self, reactive_atom_coords:np.array, bonded_atom_coords:np.array, symbol, filename=None, neighbors_indexes=None, neighbors_symbols=None) -> None:
        '''
        Adding the properties of the two atom considered

        :params reactive_atom_coords: coordinates of the reactive atom
        :params bonded_atom_coords: coordinates of the atom bonded at the reactive object
        :return: None
        '''
        self.neighbors_symbols = neighbors_symbols if neighbors_symbols != None else self.neighbors_symbols

        self.coord = reactive_atom_coords
        self.others = bonded_atom_coords

        self.vectors = self.others - self.coord # vectors connecting center to each of the two substituents

        try:
            key = symbol + ' ' + str(self)
            orb_dim = orb_dim_dict[key]
        except:
            orb_dim = 1
            print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using 1 A.')



        self.vectors = orb_dim * np.array([norm(v) for v in self.vectors]) # making both vectors a fixed, defined length
        
        self.alignment_matrix = R.align_vectors(np.array([[1,0,0]]), np.array([-np.mean(self.vectors, axis=0)]))[0].as_matrix()

        alignment_on_z = R.align_vectors(np.array([[0,0,1]]), np.array([self.vectors[0]]))[0].as_matrix()

        rotoreflexion_matrix = np.array([[ np.cos(2*np.pi/4), np.sin(2*np.pi/4),  0],
                                         [-np.sin(2*np.pi/4), np.cos(2*np.pi/4),  0],
                                         [                 0,                 0, -1]])

        self.center = np.array([v @ alignment_on_z @ rotoreflexion_matrix @ np.linalg.inv(alignment_on_z) for v in self.vectors])
        self.center += self.coord
        # two vectors defining the position of the two orbital lobes centers


        return None

atom_type_dict = {
             'H1' : Single(),
             'C1' : Single(),
             'C2' : Sp(), # toroidal geometry
             'C3' : Sp2(), # double ball
             'C4' : Sp3(), # one ball: on the back of weakest bond. If can't tell which is which, one big ball
             'N1' : Single(),
            #  'N2' : 'imine', # one ball on free side
             'N3' : Sp2(), # or one ball on free side?
             'N4' : Sp3(),
            #  'O1' : 'ketone-like', # two balls 120° apart. Also for alkoxides, good enough
             'O2' : Ether(), # or alcohol, two balls 109,5° apart
            #  'S1' : 'ketone-like',
             'S2' : Ether(),
             'F1' : Single(),
             'Cl1': Single(),
             'Br1': Single(),
             'I1' : Single(),
             }

