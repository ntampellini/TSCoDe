import numpy as np
from parameters import *
from periodictable import core, covalent_radius
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

def _rotate_scalar_field(array, rotations):
    '''
    :params array:      an input scalar field with shape (a,b,c)
    :params rotations:  tuple of shape (1,3) with rotation 
                        angles along x, y, z axes, in degrees
    :params pivot:      pivot of the rotation
    :return:            rotated array
    '''
    for axis, angle in enumerate(rotations):
        axes = [0,1,2]
        axes.remove(axis)
        array = ndimage.rotate(array, angle, axes=tuple(axes), reshape=False)
    return array

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

        self.center = np.array([orb_dim * (self.coord - self.other) / np.linalg.norm(self.coord - self.other) + self.coord])
        return None

    def stamp_orbital(self, box_shape, voxel_dim:float=VOXEL_DIM, stamp_size=STAMP_SIZE, hardness=HARDNESS):
        '''
        create the fake orbital above the reacting atom

        :params box: conformational density scalar values array (conf_dens)
        :param voxel_dim:    size of the square Voxel side, in Angstroms
        :param stamp_size:   radius of sphere used as stamp for carbon atoms, in Angstroms
        :param hardness:     steepness of radial probability decay (gaussian, k in e^(-kr^2))
        :return: None
        '''
        self.orb_dens = np.zeros(box_shape, dtype=float)
        stamp_size *= self.rel_radii
        stamp_len = round(stamp_size/voxel_dim)
        self.stamp = np.zeros((stamp_len, stamp_len, stamp_len), dtype=float)
        for x in range(stamp_len):                                   # probably optimizable
            for y in range(stamp_len):
                for z in range(stamp_len):
                    r_2 = (x - stamp_len/2)**2 + (y - stamp_len/2)**2 + (z - stamp_len/2)**2
                    self.stamp[x, y, z] = np.exp(-hardness*voxel_dim/stamp_size*r_2)
        
        x = round((self.center[0][0] - stamp_size/2)/voxel_dim + box_shape[0]/2)
        y = round((self.center[0][1] - stamp_size/2)/voxel_dim + box_shape[1]/2)
        z = round((self.center[0][2] - stamp_size/2)/voxel_dim + box_shape[2]/2)

        number_of_slabs_to_add = 0
        while True:
            try:
                self.orb_dens[x:x+stamp_len, y:y+stamp_len, z:z+stamp_len] += self.stamp
                break
            except ValueError:
                slab = np.zeros((1, self.orb_dens.shape[1], self.orb_dens.shape[2]), dtype=float)
                self.orb_dens = np.concatenate((self.orb_dens, slab))
                number_of_slabs_to_add += 1
                # print(f'Slab added: {number_of_slabs_to_add}')

        return 0, number_of_slabs_to_add


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
        self.orb_vec = np.mean(np.array([np.cross(self.vectors[0], self.vectors[1]),
                                         np.cross(self.vectors[1], self.vectors[2]),
                                         np.cross(self.vectors[2], self.vectors[0])]), axis=0)

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

    def stamp_orbital(self, box_shape, voxel_dim:float=VOXEL_DIM, stamp_size=STAMP_SIZE, hardness=HARDNESS):
        '''
        create the fake orbital above the reacting atom

        :params box: conformational density scalar values array (conf_dens)
        :param voxel_dim:    size of the square Voxel side, in Angstroms
        :param stamp_size:   radius of sphere used as stamp for carbon atoms, in Angstroms
        :param hardness:     steepness of radial probability decay (gaussian, k in e^(-kr^2))
        :return: None
        '''
        self.orb_dens = np.zeros(box_shape, dtype=float)
        stamp_size *= self.rel_radii
        stamp_len = round(stamp_size/voxel_dim)
        self.stamp = np.zeros((stamp_len, stamp_len, stamp_len), dtype=float)
        for x in range(stamp_len):                                   # probably optimizable
            for y in range(stamp_len):
                for z in range(stamp_len):
                    r_2 = (x - stamp_len/2)**2 + (y - stamp_len/2)**2 + (z - stamp_len/2)**2
                    self.stamp[x, y, z] = np.exp(-hardness*voxel_dim/stamp_size*r_2)
        
        x = round((self.center[1][0] - stamp_size/2)/voxel_dim + box_shape[0]/2)
        y = round((self.center[1][1] - stamp_size/2)/voxel_dim + box_shape[1]/2)
        z = round((self.center[1][2] - stamp_size/2)/voxel_dim + box_shape[2]/2)

        self.orb_dens[x:x+stamp_len, y:y+stamp_len, z:z+stamp_len] += self.stamp

        x = round((self.center[0][0] - stamp_size/2)/voxel_dim + box_shape[0]/2)
        y = round((self.center[0][1] - stamp_size/2)/voxel_dim + box_shape[1]/2)
        z = round((self.center[0][2] - stamp_size/2)/voxel_dim + box_shape[2]/2)

        self.orb_dens[x:x+stamp_len, y:y+stamp_len, z:z+stamp_len] += self.stamp

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

    def stamp_orbital(self, box_shape, voxel_dim:float=VOXEL_DIM, stamp_size=4, hardness=HARDNESS):
        '''
        create the fake orbital above the reacting atom

        :params box: conformational density scalar values array (conf_dens)
        :param voxel_dim:    size of the square Voxel side, in Angstroms
        :param stamp_size:   radius of sphere used as stamp for carbon atoms, in Angstroms
        :param hardness:     steepness of radial probability decay (gaussian, k in e^(-kr^2))
        :return: None
        '''
        self.orb_dens = np.zeros(box_shape, dtype=float)
        stamp_len = round(stamp_size/voxel_dim*self.rel_radii)
        self.stamp = np.zeros((stamp_len, stamp_len, stamp_len), dtype=float)
        r_max = stamp_len/3
        r_min = stamp_len/6
        for x in range(stamp_len):                                   # probably optimizable
            for y in range(stamp_len):
                for z in range(stamp_len):
                    r_2 = (((x - stamp_len/2)**2 + (y - stamp_len/2)**2)**(1/2) - r_max)**2 + (z - stamp_len/2)**2 - r_min
                    self.stamp[x, y, z] = np.exp(-hardness*voxel_dim/stamp_size*r_2)

        rot_vec = R.align_vectors(np.array([self.vectors[0]]), np.array([[0,0,1]]))[0].as_euler('xyz')*(360/(2*np.pi))
        self.stamp = _rotate_scalar_field(self.stamp, rot_vec)
        # TODO: Rotate the toroid to align it to the C-C axis
        
        x = round((self.coord[0] - stamp_size/2)/voxel_dim + box_shape[0]/2)
        y = round((self.coord[1] - stamp_size/2)/voxel_dim + box_shape[1]/2)
        z = round((self.coord[2] - stamp_size/2)/voxel_dim + box_shape[2]/2)
        self.orb_dens[x:x+stamp_len, y:y+stamp_len, z:z+stamp_len] += self.stamp

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

            self.center = np.array([orb_dim * self.orb_vec / np.linalg.norm(self.orb_vec) + self.coord])
        else:
            self.leaving_group_found = False
            self.center = np.array([self.coord])
            self.orb_vec = self.coord
            print('ATTENTION: COULD NOT TELL LEAVING GROUP ATOM ON SP3 REACTIVE CENTER, APPROXIMATING ORBITALS AS A SPHERE ON THE SP3 ATOM!')

        self.alignment_matrix = R.align_vectors(np.array([[1,0,0]]), np.array([self.center[0]]))[0].as_matrix()

        return None

    def stamp_orbital(self, box_shape, voxel_dim:float=VOXEL_DIM, stamp_size=STAMP_SIZE, hardness=HARDNESS):
        '''
        create the fake orbital above the reacting atom

        :params box: conformational density scalar values array (conf_dens)
        :param voxel_dim:    size of the square Voxel side, in Angstroms
        :param stamp_size:   radius of sphere used as stamp for carbon atoms, in Angstroms
        :param hardness:     steepness of radial probability decay (gaussian, k in e^(-kr^2))
        :return: None
        '''
        self.orb_dens = np.zeros(box_shape, dtype=float)
        stamp_size = stamp_size*self.rel_radii if self.leaving_group_found else stamp_size*self.rel_radii*2
        stamp_len = round(stamp_size/voxel_dim)
        self.stamp = np.zeros((stamp_len, stamp_len, stamp_len), dtype=float)
        for x in range(stamp_len):                                   # probably optimizable
            for y in range(stamp_len):
                for z in range(stamp_len):
                    r_2 = (x - stamp_len/2)**2 + (y - stamp_len/2)**2 + (z - stamp_len/2)**2
                    self.stamp[x, y, z] = np.exp(-hardness*voxel_dim/stamp_size*r_2)
        
        x = round((self.center[0][0] - stamp_size/2)/voxel_dim + box_shape[0]/2)
        y = round((self.center[0][1] - stamp_size/2)/voxel_dim + box_shape[1]/2)
        z = round((self.center[0][2] - stamp_size/2)/voxel_dim + box_shape[2]/2)

        number_of_slabs_to_add = 0
        while True:
            try:
                self.orb_dens[x:x+stamp_len, y:y+stamp_len, z:z+stamp_len] += self.stamp
                break
            except ValueError:
                slab = np.zeros((1, self.orb_dens.shape[1], self.orb_dens.shape[2]), dtype=float)
                self.orb_dens = np.concatenate((self.orb_dens, slab))
                number_of_slabs_to_add += 1
                # print(f'Slab added: {number_of_slabs_to_add}')

        return 0, number_of_slabs_to_add

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



        self.vectors = orb_dim * np.array([v / np.linalg.norm(v) for v in self.vectors]) # making both vectors a fixed, defined length
        
        self.alignment_matrix = R.align_vectors(np.array([[1,0,0]]), np.array([-np.mean(self.vectors, axis=0)]))[0].as_matrix()

        alignment_on_z = R.align_vectors(np.array([[0,0,1]]), np.array([self.vectors[0]]))[0].as_matrix()

        rotoreflexion_matrix = np.array([[ np.cos(2*np.pi/4), np.sin(2*np.pi/4),  0],
                                         [-np.sin(2*np.pi/4), np.cos(2*np.pi/4),  0],
                                         [                 0,                 0, -1]])

        self.center = np.array([v @ alignment_on_z @ rotoreflexion_matrix @ np.linalg.inv(alignment_on_z) for v in self.vectors])
        self.center += self.coord
        # two vectors defining the position of the two orbital lobes centers


        return None

    def stamp_orbital(self, box_shape, voxel_dim:float=VOXEL_DIM, stamp_size=STAMP_SIZE, hardness=HARDNESS):
        '''
        create the fake orbital above the reacting atom

        :params box: conformational density scalar values array (conf_dens)
        :param voxel_dim:    size of the square Voxel side, in Angstroms
        :param stamp_size:   radius of sphere used as stamp for carbon atoms, in Angstroms
        :param hardness:     steepness of radial probability decay (gaussian, k in e^(-kr^2))
        :return: None
        '''
        self.orb_dens = np.zeros(box_shape, dtype=float)
        stamp_size *= self.rel_radii
        stamp_len = round(stamp_size/voxel_dim)
        self.stamp = np.zeros((stamp_len, stamp_len, stamp_len), dtype=float)
        for x in range(stamp_len):                                   # probably optimizable
            for y in range(stamp_len):
                for z in range(stamp_len):
                    r_2 = (x - stamp_len/2)**2 + (y - stamp_len/2)**2 + (z - stamp_len/2)**2
                    self.stamp[x, y, z] = np.exp(-hardness*voxel_dim/stamp_size*r_2)
        
        x = round((self.center[0][0] - stamp_size/2)/voxel_dim + box_shape[0]/2)
        y = round((self.center[0][1] - stamp_size/2)/voxel_dim + box_shape[1]/2)
        z = round((self.center[0][2] - stamp_size/2)/voxel_dim + box_shape[2]/2)

        number_of_slabs_to_add = 0
        while True:
            try:
                self.orb_dens[x:x+stamp_len, y:y+stamp_len, z:z+stamp_len] += self.stamp
                break
            except ValueError:
                slab = np.zeros((1, self.orb_dens.shape[1], self.orb_dens.shape[2]), dtype=float)
                self.orb_dens = np.concatenate((self.orb_dens, slab))
                number_of_slabs_to_add += 1

        x = round((self.center[1][0] - stamp_size/2)/voxel_dim + box_shape[0]/2)
        y = round((self.center[1][1] - stamp_size/2)/voxel_dim + box_shape[1]/2)
        z = round((self.center[1][2] - stamp_size/2)/voxel_dim + box_shape[2]/2)

        while True:
            try:
                self.orb_dens[x:x+stamp_len, y:y+stamp_len, z:z+stamp_len] += self.stamp
                break
            except ValueError:
                slab = np.zeros((1, self.orb_dens.shape[1], self.orb_dens.shape[2]), dtype=float)
                self.orb_dens = np.concatenate((self.orb_dens, slab))
                number_of_slabs_to_add += 1

        # print(f'Slabs added to ether: {number_of_slabs_to_add}')

        return 0, number_of_slabs_to_add



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
             'N3' : 'amine', # one ball on free side
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

