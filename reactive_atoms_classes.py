import numpy as np
from parameters import *
from periodictable import core, covalent_radius
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

def norm(vec):
    return vec / np.linalg.norm(vec)

class Single:
    '''
    '''
    def __init__(self, rel_radii):
        self.rel_radii = rel_radii
    
    def __repr__(self):
        return 'Single Bond'

    def prop(self, reactive_atom_coords:np.array, bonded_atom_coords:np.array, symbol, neighbors_symbols=None) -> None:
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
        self.other = bonded_atom_coords
        self.orb_vec = self.coord - self.other

        try:
            key = symbol + ' ' + str(self)
            orb_dim = orb_dim_dict[key]
        except:
            orb_dim = 0.5*np.linalg.norm(self.coord - self.other)
            print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using half the bonding distance.')

        self.center = np.array([orb_dim * norm(self.coord - self.other) + self.coord])
        return None

class Sp2:
    '''
    '''
    def __init__(self, rel_radii):
        self.rel_radii = rel_radii

    def __repr__(self):
        return 'sp2'

    def prop(self, reactive_atom_coords:np.array, bonded_atom_coords:np.array, symbol, neighbors_symbols=None) -> None:
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
    def __init__(self, rel_radii):
        self.rel_radii = rel_radii

    def __repr__(self):
        return 'sp'

    def prop(self, reactive_atom_coords:np.array, bonded_atom_coords:np.array, symbol, neighbors_symbols=None) -> None:
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
    def __init__(self, rel_radii):
        self.rel_radii = rel_radii
    
    def __repr__(self):
        return 'sp3'

    def prop(self, reactive_atom_coords:np.array, bonded_atom_coords:np.array, symbol, neighbors_symbols=None) -> None:
        '''
        Adding the properties of the two atom considered

        :params reactive_atom_coords: coordinates of the reactive atom
        :params bonded_atom_coords: coordinates of the atom bonded at the reactive object
        :return: None
        '''
        assert reactive_atom_coords.shape == (3,)
        assert bonded_atom_coords.shape[1] == 3

        self.neighbors_symbols = neighbors_symbols if neighbors_symbols != None else self.neighbors_symbols
        self.coord = reactive_atom_coords
        self.other = bonded_atom_coords

        if len([atom for atom in self.neighbors_symbols if atom in ['O', 'N', 'Cl', 'Br', 'I']]) == 1: # if we can tell where is the leaving group
            self.leaving_group_found = True
            leaving_group_coords = self.other[self.neighbors_symbols.index([atom for atom in self.neighbors_symbols if atom in ['O', 'Cl', 'Br', 'I']][0])]
            self.orb_vec = self.coord - leaving_group_coords

            try:
                key = symbol + ' ' + str(self)
                orb_dim = orb_dim_dict[key]
            except:
                orb_dim = 1
                print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using 1 A.')

            self.center = np.array([orb_dim * norm(self.orb_vec) + self.coord])
        else:
            self.leaving_group_found = False
            self.center = np.array([self.coord])
            self.orb_vec = self.coord
            print('ATTENTION: COULD NOT TELL LEAVING GROUP ATOM ON SP3 REACTIVE CENTER, APPROXIMATING ORBITALS AS A SPHERE ON THE SP3 ATOM!')

        self.alignment_matrix = R.align_vectors(np.array([[1,0,0]]), np.array([self.center[0]]))[0].as_matrix()

        return None


class Ether:
    '''
    '''
    def __init__(self, rel_radii):
        self.rel_radii = rel_radii
    
    def __repr__(self):
        return 'Ether'

    def prop(self, reactive_atom_coords:np.array, bonded_atom_coords:np.array, symbol, neighbors_symbols=None) -> None:
        '''
        Adding the properties of the two atom considered

        :params reactive_atom_coords: coordinates of the reactive atom
        :params bonded_atom_coords: coordinates of the atom bonded at the reactive object
        :return: None
        '''
        self.neighbors_symbols = neighbors_symbols if neighbors_symbols != None else self.neighbors_symbols

        self.coord = reactive_atom_coords
        self.other = bonded_atom_coords

        self.vectors = self.other - self.coord # vectors connecting center to each of the two substituents

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


pt = core.PeriodicTable(table="H=1")
covalent_radius.init(pt)
c_radii = pt[6].covalent_radius

atom_type_dict = {
             'H1' : Single(pt[1].covalent_radius/c_radii),
             'C1' : Single(1),
             'C2' : Sp(1), # toroidal geometry
             'C3' : Sp2(1), # double ball
             'C4' : Sp3(1), # one ball: on the back of weakest bond. If can't tell which is which, one big ball
             'N1' : Single(pt[7].covalent_radius/c_radii),
             'N2' : 'imine', # one ball on free side
             'N3' : Sp2(pt[7].covalent_radius/c_radii), # or one ball on free side?
             'N4' : Sp3(pt[7].covalent_radius/c_radii),
             'O1' : 'ketone-like', # two balls 120° apart. Also for alkoxides, good enough
             'O2' : Ether(pt[8].covalent_radius/c_radii), # or alcohol, two balls 109,5° apart
             'S1' : 'ketone-like',
             'S2' : Ether(pt[16].covalent_radius/c_radii),
             'F1' : Single(pt[9].covalent_radius/c_radii),
             'Cl1': Single(pt[17].covalent_radius/c_radii),
             'Br1': Single(pt[35].covalent_radius/c_radii),
             'I1' : Single(pt[53].covalent_radius/c_radii),
             }

