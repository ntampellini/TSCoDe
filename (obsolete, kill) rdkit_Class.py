from cclib.io import ccread
import os
import sys
import numpy as np
from openbabel import openbabel as ob
from subprocess import DEVNULL, STDOUT, check_call
from time import time
from pprint import pprint

from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolDescriptors, AllChem
from rdkit_conformational_search import csearch


class Density_object:
    '''
    Conformational density 3D object from a conformational ensemble.

    filename            Input file name. Can be anything, .xyz preferred
    reactive_atoms      Index of atoms that will link during the desired reaction.
                        May be either int or list of int.
    return              Writes a new filename_aligned.xyz file and returns its name


    '''

    def __init__(self, filename, reactive_atoms, debug=False, T=298.15):
        '''
        Initializing class properties: reading conformational ensemble file, aligning
        conformers to first and centering them in origin.

        '''

        def _generate_and_align_ensemble(filename, reactive_atoms):

            '''
            Generate and align a set of conformers to the first one, writing a new ensemble file.
            Alignment is done on reactive atom(s) and its immediate neighbors.

            filename            Input file name, must be a single structure file
            reactive_atoms      Index of atoms that will link during the desired reaction.
                                May be either int or list of int.
            return              Writes a new filename_aligned.xyz file and returns its name

            '''
            try:
                if type(reactive_atoms) == int:
                    reactive_atoms = [reactive_atoms]
            except:
                raise Exception('Unrecognized reactive atoms IDs. Argument must either be one int or a list of ints.')


            converted_name = filename.split('.')[0] + '.sdf'
            check_call(f'obabel {filename} -o sdf -O {converted_name}'.split(), stdout=DEVNULL, stderr=STDOUT)    # Bad, we should improve this

            ensemble, energies = csearch(converted_name)

            os.remove(converted_name)

            self.energies = np.array(energies) - min(energies)

            Chem.rdMolAlign.AlignMolConformers(ensemble, reactive_atoms)

            outname = filename.split('.')[0] + '_aligned_rdkit.sdf'
            writer = Chem.SDWriter(outname)
            for i, e in enumerate(self.energies):
                if e < 10:
                    writer.write(ensemble, confId=i)
                    
            print(f'Conformational Search on {filename} : {len([e for e in energies if e < 10])} conformers found.')

            # http://rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html?highlight=align#rdkit.Chem.rdMolAlign.AlignMolConformers

            return outname


        self.rootname = filename.split('.')[0]

        ccread_object = ccread(_generate_and_align_ensemble(filename, reactive_atoms))
        coordinates = np.array(ccread_object.atomcoords)

        if all([len(coordinates[i])==len(coordinates[0]) for i in range(1, len(coordinates))]):     # Checking that ensemble has constant lenght
            if debug: print(f'DEBUG--> Initializing object {filename}\nDEBUG--> Found {len(coordinates)} structures with {len(coordinates[0])} atoms')
        else:
            raise Exception(f'Ensemble not recognized, no constant lenght. len are {[len(i) for i in coordinates]}')

        self.centroid = np.sum(np.sum(coordinates, axis=0), axis=0) / (len(coordinates) * len(coordinates[0]))
        if debug: print('DEBUG--> Centroid was', self.centroid)
        self.atomcoords = coordinates - self.centroid
        # After reading aligned conformers, they are stored as self.atomcoords only after being aligned to origin

        self.atoms = np.array([atom for structure in self.atomcoords for atom in structure])       # single list with all atom positions
        if debug: print(f'DEBUG--> Total of {len(self.atoms)} atoms')

        self.name = filename
        self.atomnos = ccread_object.atomnos
        self.T = T

        try:
            os.remove(self.rootname + '_ensemble.xyz')
            os.remove(self.rootname + '_ensemble_aligned.xyz')
        except Exception as e:
            try:
                os.remove(self.rootname + '_aligned.xyz')
            except Exception as e: pass

    def compute_CoDe(self, voxel_dim:float=0.3, stamp_size=2, hardness=5, debug=False):
        '''
        Computing conformational density for the ensemble.

        :param voxel_dim:    size of the square Voxel side, in Angstroms
        :param stamp_size:   radius of sphere used as stamp, in Angstroms
        :param hardness:     steepness of radial probability decay (gaussian, k in e^(-kr^2))
        :return:             writes a new filename_aligned.xyz file and returns its name

        '''
        stamp_len = round((stamp_size/voxel_dim))                    # size of the box, in voxels
        if debug: print('DEBUG--> Stamp size (sphere diameter) is', stamp_len, 'voxels')
        self.stamp = np.zeros((stamp_len, stamp_len, stamp_len))
        for x in range(stamp_len):                                   # probably optimizable
            for y in range(stamp_len):
                for z in range(stamp_len):
                    r_2 = (x - stamp_len/2)**2 + (y - stamp_len/2)**2 + (z - stamp_len/2)**2
                    self.stamp[x, y, z] = np.exp(-hardness*voxel_dim/stamp_size*r_2)
        # defining single matrix to use as a stamp, basically a 3D sphere with values decaying with e^(-kr^2)
        # https://www.desmos.com/calculator/3tdiw1of3r

        # _write_cube(self.stamp, voxel_dim)         # DEBUG


        x_coords = np.array([pos[0] for pos in self.atoms])  # isolating x, y and z from self.atoms
        y_coords = np.array([pos[1] for pos in self.atoms])
        z_coords = np.array([pos[2] for pos in self.atoms])

        min_x = min(x_coords)
        min_y = min(y_coords)
        min_z = min(z_coords)

        size = (max(x_coords) - min_x,
                max(y_coords) - min_y,
                max(z_coords) - min_z)

        outline = 2*stamp_len

        shape = (int(np.ceil(size[0]/voxel_dim)) + outline,
                 int(np.ceil(size[1]/voxel_dim)) + outline,
                 int(np.ceil(size[2]/voxel_dim)) + outline)

        # size of box, in number of voxels (i.e. matrix items), is defined by how many of them are needed to include all atoms
        # given the fact that they measure {voxel_dim} Angstroms. To these numbers, an "outline" of 2{stamp_len} voxels is added to
        # each axis to ensure that there is enough space for upcoming stamping of density information for atoms close to the boundary.


        self.ensemble_origin = -1*np.array([(shape[0] + outline)/2*voxel_dim, 
                                            (shape[1] + outline)/2*voxel_dim,
                                            (shape[2] + outline)/2*voxel_dim])        # vector connecting ensemble centroid with box origin (corner), used later

        # self.ensemble_origin = -0.5 * np.array(size) + stamp_size



        self.box = np.zeros(shape, dtype=float)
        if debug: print('DEBUG--> Box shape is', shape, 'voxels')
        # defining box dimensions based on molecule size and voxel input

        for i, conformer in enumerate(self.atomcoords):                     # adding density to array elements
            if self.energies[i] < 10:                                       # cutting at rel. E +10 kcal/mol 
                for atom in conformer:
                    x_pos = round((atom[0] - min_x) / voxel_dim + outline/2)
                    y_pos = round((atom[1] - min_y) / voxel_dim + outline/2)
                    z_pos = round((atom[2] - min_z) / voxel_dim + outline/2)
                    weight = np.exp(-self.energies[i]*503.2475342795285/self.T)     # conformer structures are weighted on their relative energy (Boltzmann)
                    self.box[x_pos : x_pos + stamp_len, y_pos : y_pos + stamp_len, z_pos : z_pos + stamp_len] += self.stamp*weight

        self.box = 100 * self.box / max(self.box.reshape((1,np.prod(shape)))[0])  # normalize box values to the range 0 - 100

        self.voxdim = voxel_dim

    def write_cubefile(self):
        '''
        Writes a Gaussian .cube file in the working directory with the conformational density information
        
        '''
        cubename = self.name.split('.')[0] + '_CoDe.cube'
        
        try:
            with open(cubename, 'w') as f:
                f.write(' CoDe Cube File - Conformational Density Cube File, generated by TSCoDe (git repo link)\n')
                f.write(' OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')
                f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(len(self.atomcoords)*len(self.atomnos),
                                                                            self.ensemble_origin[0],
                                                                            self.ensemble_origin[1],
                                                                            self.ensemble_origin[2]))  #atom number, position of molecule relative to cube origin
                
                f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(self.box.shape[0], self.voxdim, 0.000000, 0.000000))
                f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(self.box.shape[1], 0.000000, self.voxdim, 0.000000))
                f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(self.box.shape[2], 0.000000, 0.000000, self.voxdim))
                # number of voxels along x, x length of voxel. Minus in front of voxel number is for specifying Angstroms over Bohrs.
                # http://paulbourke.net/dataformats/cube


                for i in range(len(self.atomcoords)):
                    for index, atom in enumerate(self.atomcoords[i]):
                        f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\t{: >12}\n'.format(self.atomnos[index], 0.000000, atom[0], atom[1], atom[2]))

                count = 0
                total = np.prod(self.box.shape)
                print_list = []


                for x in range(self.box.shape[0]):        # Main loop: z is inner, y is middle and x is outer.
                    for y in range(self.box.shape[1]):
                        for z in range(self.box.shape[2]):
                            print_list.append('{:.5e} '.format(self.box[x, y, z]).upper())
                            count += 1
                            # loadbar(count, total, prefix='Writing .cube file... ')
                            if count % 6 == 5:
                                print_list.append('\n')
                print_list.append('\n')
                f.write(''.join(print_list))
                print(f'Wrote file {cubename} - {total} scalar values')
        except Exception as e:
            raise Exception(f'No CoDe data in {self.name} Density Object Class: write_cube method should be called only after compute_CoDe method.')


os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('Resources')

test = Density_object('funky_single.xyz', [15, 17], debug=True)
