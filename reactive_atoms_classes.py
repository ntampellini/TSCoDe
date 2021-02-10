import numpy as np
from tables import *
# from scipy import


class Single:
    '''
    '''
    def __init__(self, rel_radii):
        self.rel_radii = rel_radii
    
    def __repr__(self):
        return 'Single Bond'

    def prop(self, reactive_atom_coords:np.array, bonded_atom_coords:np.array) -> None:
        '''
        Adding the properties of the two atom considered

        :params reactive_atom_coords: coordinates of the reactive atom
        :params bonded_atom_coords: coordinates of the atom bonded at the reactive object
        :return: None
        '''
        if len(bonded_atom_coords) == 1:
            bonded_atom_coords = bonded_atom_coords[0]
        assert reactive_atom_coords.shape == (3,)
        assert bonded_atom_coords.shape == (3,)

        self.coord = reactive_atom_coords
        self.other = bonded_atom_coords
        self.center = 2*self.coord - self.other
        return None

    def stamp_orbital(self, box: np.array, voxel_dim:float=0.3, stamp_size=2, hardness=5):
        '''
        create the fake orbital above the reacting atom

        :params box: conformational density scalar values array (conf_dens)
        :param voxel_dim:    size of the square Voxel side, in Angstroms
        :param stamp_size:   radius of sphere used as stamp for carbon atoms, in Angstroms
        :param hardness:     steepness of radial probability decay (gaussian, k in e^(-kr^2))
        :return: None
        '''
        self.orb_dens = np.zeros(box.shape, dtype=float)
        stamp_len = round(stamp_size/voxel_dim*self.rel_radii)
        self.stamp = np.zeros((stamp_len, stamp_len, stamp_len), dtype=float)
        for x in range(stamp_len):                                   # probably optimizable
            for y in range(stamp_len):
                for z in range(stamp_len):
                    r_2 = (x - stamp_len/2)**2 + (y - stamp_len/2)**2 + (z - stamp_len/2)**2
                    self.stamp[x, y, z] = np.exp(-hardness*voxel_dim/stamp_size*r_2)
        
        x = round((self.center[0] - stamp_size/2)/voxel_dim  )
        y = round((self.center[1] - stamp_size/2)/voxel_dim  )
        z = round((self.center[2] - stamp_size/2)/voxel_dim  )
        print(f'Trying to print at {x,y,z}')
        self.orb_dens[x:x+stamp_len, y:y+stamp_len, z:z+stamp_len] += self.stamp

        return None


class Sp2:
    '''
    '''
    def __init__(self, rel_radii):
        self.rel_radii = rel_radii

    def __repr__(self):
        return 'sp2'

    def prop(self, reactive_atom_coords:np.array, bonded_atom_coords:np.array) -> None:
        '''
        Adding the properties of the two atom considered

        :params reactive_atom_coords: coordinates of the reactive atom
        :params bonded_atom_coords: coordinates of the atom bonded at the reactive object
        :return: None
        '''
        self.coord = reactive_atom_coords
        self.others = bonded_atom_coords
        self.vectors = self.others - self.coord # vectors connecting reactive atom with neighbors
        self.center = np.mean(np.array([np.cross(self.vectors[0], self.vectors[1]),
                                        np.cross(self.vectors[1], self.vectors[2]),
                                        np.cross(self.vectors[2], self.vectors[0])]), axis=0)
        
        return None

    def stamp_orbital(self, box: np.array, voxel_dim:float=0.3, stamp_size=2, hardness=5):
        '''
        create the fake orbital above the reacting atom

        :params box: conformational density scalar values array (conf_dens)
        :param voxel_dim:    size of the square Voxel side, in Angstroms
        :param stamp_size:   radius of sphere used as stamp for carbon atoms, in Angstroms
        :param hardness:     steepness of radial probability decay (gaussian, k in e^(-kr^2))
        :return: None
        '''
        self.orb_dens = np.zeros(box.shape, dtype=float)
        stamp_len = round(stamp_size/voxel_dim*self.rel_radii)
        self.stamp = np.zeros((stamp_len, stamp_len, stamp_len), dtype=float)
        for x in range(stamp_len):                                   # probably optimizable
            for y in range(stamp_len):
                for z in range(stamp_len):
                    r_2 = (x - stamp_len/2)**2 + (y - stamp_len/2)**2 + (z - stamp_len/2)**2
                    self.stamp[x, y, z] = np.exp(-hardness*voxel_dim/stamp_size*r_2)
        
        x = round((self.center[0] - stamp_size/2 + self.coord[0])/voxel_dim + box.shape[0]/2)
        y = round((self.center[1] - stamp_size/2 + self.coord[1])/voxel_dim + box.shape[1]/2)
        z = round((self.center[2] - stamp_size/2 + self.coord[2])/voxel_dim + box.shape[2]/2)
        try:
            self.orb_dens[x:x+stamp_len, y:y+stamp_len, z:z+stamp_len] += self.stamp
        except: pass

        x = round((-self.center[0] - stamp_size/2 + self.coord[0])/voxel_dim + box.shape[0]/2)
        y = round((-self.center[1] - stamp_size/2 + self.coord[1])/voxel_dim + box.shape[1]/2)
        z = round((-self.center[2] - stamp_size/2 + self.coord[2])/voxel_dim + box.shape[2]/2)
        try:
            self.orb_dens[x:x+stamp_len, y:y+stamp_len, z:z+stamp_len] += self.stamp
        except: pass

        return None

