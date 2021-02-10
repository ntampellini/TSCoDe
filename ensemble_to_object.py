from cclib.io import ccread
import os
import sys
import numpy as np
import networkx as nx
# from openbabel import openbabel as ob
# from time import time
from subprocess import DEVNULL, STDOUT, check_call
from rdkit_conformational_search import csearch
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolDescriptors, AllChem
from rdkit_conformational_search import csearch
from scipy.spatial.transform import Rotation as R
from tables import atom_type_dict, pt
import warnings
warnings.simplefilter("ignore", UserWarning)


# def loadbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
# 	percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
# 	filledLength = int(length * iteration // total)
# 	bar = fill * filledLength + '-' * (length - filledLength)
# 	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
# 	if iteration == total:
# 		print()

def _write_cube(array, voxdim):
    with open('Stamp_test.cube', 'w') as f:
        f.write(' CoDe Cube File - Test generated by TSCoDe\n')
        f.write(' OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')
        f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(1, 0, 0, 0))
        
        f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(array.shape[0], 1.88973*voxdim, 0.000000, 0.000000))
        f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(array.shape[1], 0.000000, 1.88973*voxdim, 0.000000))
        f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(array.shape[2], 0.000000, 0.000000, 1.88973*voxdim))
        f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(6, 0.000000, 0.000000, 0.000000))

        # number of voxels along x, x length of voxel. Minus in front of voxel number is for specifying Angstroms over Bohrs.
        # http://paulbourke.net/dataformats/cube

        print_list = []

        count = 0
        for x in range(len(array)):        # Main loop: z is inner, y is middle and x is outer.
            for y in range(len(array[x])):
                for z in range(len(array[y])):
                    print_list.append('{:.5e} '.format(array[x, y, z]).upper())
                    count += 1
                    if count % 6 == 5:
                        print_list.append('\n')
        print_list.append('\n')
        f.write(''.join(print_list))


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
        def _sort_xyz(filename, reactive_atoms):
            with open(filename, 'r') as f:
                lines = f.readlines()

            hs, not_hs= [], []
            ra = []
            for line in lines:
                if line[0] == 'H':
                    hs.append(line)
                elif line == '\n':
                    pass
                else:
                    not_hs.append(line)

            for index in reactive_atoms:
                for i, line in enumerate(not_hs[2:]):
                    if line in lines[index+1]:
                        ra.append(i)
                        break

            with open(filename, 'w') as f:
                f.writelines(not_hs)
                f.writelines(hs)

            return ra

        def _alignment_indexes(mol, reactive_atoms):
            '''
            Return the indexes to align the molecule to, given a list of
            atoms that should be reacting. List is composed by reactive atoms
            plus adjacent atoms.
            :param mol: rdkit Mol class molecule object
            :param reactive atoms: int or list of ints

            '''
            matrix = Chem.GetAdjacencyMatrix(mol)
            graph = nx.from_numpy_matrix(matrix)
            indexes = set()

            for atom in reactive_atoms:
                indexes |= set(list([(a, b) for a, b in graph.adjacency()][atom][1].keys()))
                indexes.add(atom)
            if debug: print('DEBUG--> Alignment indexes are', list(indexes))
            return list(indexes)

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

            if debug: print()

            converted_name = filename.split('.')[0] + '_sorted.xyz'
            check_call(f'obabel {filename} -o xyz -O {converted_name}'.split(), stdout=DEVNULL, stderr=STDOUT)    # Bad, we should improve this
            self.reactive_indexes = _sort_xyz(converted_name, reactive_atoms)   # sorts atoms indexes so that hydrogen atoms are after heavy atoms. For aligment purposes

            sdf_converted_name = filename.split('.')[0] + '.sdf'
            check_call(f'obabel {converted_name} -o sdf -O {sdf_converted_name}'.split(), stdout=DEVNULL, stderr=STDOUT)    # Bad, we should improve this
            # os.remove(converted_name)

            old_mol, ensemble, energies = csearch(sdf_converted_name)  # performs csearch, also returns old mol for aligment purposes
            self.energies = np.array(energies) - min(energies)
            os.remove(sdf_converted_name)

            self.rdkit_mol_object = ensemble

            alignment_indexes = _alignment_indexes(self.rdkit_mol_object, self.reactive_indexes)

            if alignment_indexes is None:               # Eventually we should raise an error I think, but we could also warn and leave this
                Chem.rdMolAlign.AlignMolConformers(ensemble)
                print(f'ATOM ALIGMENT INDEX(ES) NOT UNDERSTOOD: Ensemble for {self.rootname} aligned on all atoms, may generate undesired results.')
            else:
                Chem.rdMolAlign.AlignMolConformers(ensemble, alignment_indexes)
                
            # http://rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html?highlight=align#rdkit.Chem.rdMolAlign.AlignMolConformers

            outname = filename.split('.')[0] + '_aligned_rdkit.sdf' # Writes sigle conformers to files then convert them to one ensemble.
            writer = Chem.SDWriter(outname)                         # Really not ideal but RDKit doesn't seem to be able to output the
            for i, e in enumerate(self.energies):                   # .xyz ensemble required by ccread, so I don't see another way.
                if e < 10:
                    writer.write(ensemble, confId=i)
            writer.close()        

            if debug: print(f'DEBUG--> Conformational Search on {filename} : {len(self.energies)} conformers found.')
            if debug: print(f'DEBUG--> Relative energies are : {self.energies}')


            xyz_outname = filename.split('.')[0] + '_aligned_rdkit.xyz'
            check_call(f'obabel {outname} -o xyz -O {xyz_outname}'.split(), stdout=DEVNULL, stderr=STDOUT)    # Bad, we should improve this(?)
            os.remove(outname)

            ccread_object = ccread(xyz_outname)
            # os.remove(xyz_outname)

            return ccread_object

        def _orient_along_x(array, vector):
            '''
            :params array:    array of atomic coordinates arrays: len(array) structures with len(array[i]) atoms
            :params vector:   list of shape (1,3) with anchor vector to align to the x axis
            :return:          array, aligned so that vector is on x
            '''
            assert array.shape[1] == 3
            assert vector.shape == (3,)
            rotation_matrix = R.align_vectors(np.array([[1,0,0]]), np.array([vector]))[0].as_matrix()
            return np.array([rotation_matrix @ v for v in array])

        self.rootname = filename.split('.')[0]

        ccread_object = _generate_and_align_ensemble(filename, reactive_atoms)
        coordinates = np.array(ccread_object.atomcoords)

        if all([len(coordinates[i])==len(coordinates[0]) for i in range(1, len(coordinates))]):     # Checking that ensemble has constant lenght
            if debug: print(f'DEBUG--> Initializing object {filename}\nDEBUG--> Found {len(coordinates)} structures with {len(coordinates[0])} atoms')
        else:
            raise Exception(f'Ensemble not recognized, no constant lenght. len are {[len(i) for i in coordinates]}')

        self.centroid = np.sum(np.sum(coordinates, axis=0), axis=0) / (len(coordinates) * len(coordinates[0]))
        if debug: print('DEBUG--> Centroid was', self.centroid)
        self.atomcoords = coordinates - self.centroid

        if type(self.reactive_indexes) is int:
            reactive_vector = np.mean(np.array([structure[self.reactive_indexes] for structure in self.atomcoords]), axis=0)
        else:
            reactive_vector = []
            for structure in self.atomcoords:
                for index in self.reactive_indexes:
                    reactive_vector.append(structure[index])
            reactive_vector = np.mean(np.array(reactive_vector), axis=0)
            
        self.atomcoords = np.array([_orient_along_x(structure, reactive_vector) for structure in self.atomcoords])
        # After reading aligned conformers, they are stored as self.atomcoords after being translated to origin and aligned the reactive atom(s) to x axis.

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

    def compute_CoDe(self, voxel_dim:float=0.3, stamp_size=2, hardness=5, breadth=2, debug=False):
        '''
        Computing conformational density for the ensemble.

        :param voxel_dim:    size of the square Voxel side, in Angstroms
        :param stamp_size:   radius of sphere used as stamp for carbon atoms, in Angstroms
        :param hardness:     steepness of radial probability decay (gaussian, k in e^(-kr^2))
        :param breadth:      value that divides relative energies before calculating relative Boltzmann weight.
                             If = 1, normal Boltzmann distribution is used. If >1, keeps more high-energy conformers.
        :return:             None

        '''

        stamp_len = round((stamp_size/voxel_dim))                    # size of the box, in voxels
        self.stamp = {}
        if debug: print('DEBUG--> Carbon stamp size (sphere diameter) is', stamp_len, 'voxels')

        for atom in set(self.atomnos):
            rel_radii = pt[atom].covalent_radius / pt[6].covalent_radius                   # relative radii based on sp3 carbon
            if debug: print(f'DEBUG--> {pt[atom]} is {round(rel_radii, 2)} times the radius of C')
            new_stamp_len = round(stamp_len*rel_radii)
            new_stamp_size = stamp_size*rel_radii
            self.stamp[atom] = np.zeros((new_stamp_len, new_stamp_len, new_stamp_len))
            for x in range(new_stamp_len):                                   # probably optimizable
                for y in range(new_stamp_len):
                    for z in range(new_stamp_len):
                        r_2 = (x - new_stamp_len/2)**2 + (y - new_stamp_len/2)**2 + (z - new_stamp_len/2)**2
                        self.stamp[atom][x, y, z] = np.exp(-hardness*voxel_dim/new_stamp_size*r_2)

        # defining matrices to use as stamps, basically 3D spheres with values decaying with e^(-kr^2)
        # stamps for other elements are defined based on relative radii with carbon
        # https://www.desmos.com/calculator/3tdiw1of3r

        # _write_cube(self.stamp[6], voxel_dim)         # DEBUG


        x_coords = np.array([pos[0] for pos in self.atoms])  # isolating x, y and z from self.atoms
        y_coords = np.array([pos[1] for pos in self.atoms])
        z_coords = np.array([pos[2] for pos in self.atoms])

        min_x = min(x_coords)
        min_y = min(y_coords)
        min_z = min(z_coords)

        size = (max(x_coords) - min_x,
                max(y_coords) - min_y,
                max(z_coords) - min_z)
        if debug: print('DEBUG--> Size of box in A is', size, 'A')

        outline = 2*stamp_len

        # big_atoms = {el:el.covalent_radius for el in pt if el.covalent_radius != None and el.covalent_radius > 2*0.76}

        # if any([pt[atom] in big_atoms for atom in self.atomnos]):
        #     biggest_atom_size = max([big_atoms[atom] for atom in self.atomnos if atom in big_atoms.keys()])
        #     outline = 2*biggest_atom_size
        #     if debug: print(f'DEBUG--> Big atom found! ({list(big_atoms.keys())[list(big_atoms.values()).index(biggest_atom_size)]})')
        # THIS SHOULD WORK BUT I NEED TO CHECK NOT TO MAKE CONFUSION WITH LATER "OUTLINE" DEFINITION

        shape = (int(np.ceil(size[0]/voxel_dim)) + outline,
                 int(np.ceil(size[1]/voxel_dim)) + outline,
                 int(np.ceil(size[2]/voxel_dim)) + outline)

        # size of box, in number of voxels (i.e. matrix items), is defined by how many of them are needed to include all atoms
        # given the fact that they measure {voxel_dim} Angstroms. To these numbers, an "outline" of 2{stamp_len} voxels is added to
        # each axis to ensure that there is enough space for upcoming stamping of density information for atoms close to the boundary.
        # This means we are safe to stamp every atom up to twice the carbon size.


        self.ensemble_origin = -1*np.array([size[0]/2 + stamp_size, 
                                            size[1]/2 + stamp_size,
                                            size[2]/2 + stamp_size])        # vector connecting ensemble centroid with box origin (corner), used later



        self.conf_dens = np.zeros(shape, dtype=float)
        if debug: print('DEBUG--> Box shape is', shape, 'voxels')
        # defining box dimensions based on molecule size and voxel input


        for i, conformer in enumerate(self.atomcoords):                     # adding density to array elements
            if self.energies[i] < 10:                                       # cutting at rel. E +10 kcal/mol 
                for j, atom in enumerate(conformer):
                    new_stamp_len = len(self.stamp[self.atomnos[j]][0])
                    x_pos = round((atom[0] / voxel_dim) + shape[0]/2 - new_stamp_len/2)
                    y_pos = round((atom[1] / voxel_dim) + shape[1]/2 - new_stamp_len/2)
                    z_pos = round((atom[2] / voxel_dim) + shape[2]/2 - new_stamp_len/2)
                    weight = np.exp(-self.energies[i] / breadth * 503.2475342795285 / self.T)                 # conformer structures are weighted on their relative energy (Boltzmann)
                    self.conf_dens[x_pos : x_pos + new_stamp_len,
                             y_pos : y_pos + new_stamp_len,
                             z_pos : z_pos + new_stamp_len] += self.stamp[self.atomnos[j]] * weight

        self.conf_dens = 100 * self.conf_dens / max(self.conf_dens.reshape((1,np.prod(shape)))[0])  # normalize box values to the range 0 - 100

        self.voxdim = voxel_dim

    def write_map(self, scalar_field, mapname='map'):
        '''
        Writes a Gaussian .cube file in the working directory with the scalar field information
        
        '''
        cubename = self.name.split('.')[0] + '_' + mapname + '.cube'
        
        try:
            with open(cubename, 'w') as f:
                f.write(f' CoDe Cube File - {mapname} Cube File, generated by TSCoDe (git repo link)\n')
                f.write(' OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')
                f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(len(self.atomcoords)*len(self.atomnos),
                                                                            1.88973*self.ensemble_origin[0],
                                                                            1.88973*self.ensemble_origin[1],
                                                                            1.88973*self.ensemble_origin[2]))
                                                                            #atom number, position of molecule relative to cube origin
                
                f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(scalar_field.shape[0], 1.88973*self.voxdim, 0.000000, 0.000000))
                f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(scalar_field.shape[1], 0.000000, 1.88973*self.voxdim, 0.000000))
                f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\n'.format(scalar_field.shape[2], 0.000000, 0.000000, 1.88973*self.voxdim))
                # number of voxels along x, x length of voxel. 1.88973 converts Angstroms to Bohrs.
                # http://paulbourke.net/dataformats/cube


                for i in range(len(self.atomcoords)):
                    for index, atom in enumerate(self.atomcoords[i]):
                        f.write('{: >4}\t{: >12}\t{: >12}\t{: >12}\t{: >12}\n'.format(self.atomnos[index], 0.000000, 1.88973*atom[0], 1.88973*atom[1], 1.88973*atom[2]))

                count = 0
                total = np.prod(scalar_field.shape)
                print_list = []


                for x in range(scalar_field.shape[0]):        # Main loop: z is inner, y is middle and x is outer.
                    for y in range(scalar_field.shape[1]):
                        for z in range(scalar_field.shape[2]):
                            print_list.append('{:.5e} '.format(scalar_field[x, y, z]).upper())
                            count += 1
                            # loadbar(count, total, prefix='Writing .cube file... ')
                            if count % 6 == 5:
                                print_list.append('\n')
                print_list.append('\n')
                f.write(''.join(print_list))
        except Exception:
            raise Exception(f'No CoDe data in {self.name} Density Object Class: write_cube method should be called only after compute_CoDe method.')
        print(f'Wrote file {cubename} - {total} scalar values')

        vmdname = self.name.split('.')[0] + '_' + mapname + '.vmd'
        with open(vmdname, 'w') as f:
            string = ('display resetview\n'
                      'mol new {%s} type {cube} first 0 last -1 step 1 waitfor 1\n'
                      'mol representation CPK 0.500000 0.300000 20.000000 15.000000\n'
                      'mol color Name\n'
                      'mol selection all\n'
                      'mol material Opaque\n'
                      'mol addrep top\n'
                      'mol modcolor 1 top Volume 0\n'
                      'mol modstyle 1 top Isosurface 10.000000 0 0 0 1 1\n'
                      
                    #   'mol color Volume 0\n'
                    #   'mol representation Isosurface 10.000000 0 0 0 1 1\n'
                    #   'mol selection all\n'
                    #   'mol material Opaque\n'
                      'mol addrep top\n'
                      'mol modstyle 2 top Isosurface 10.000000 0 1 0 1 1\n'
                      'mol modmaterial 2 top Transparent\n'
                      
                      
                      
                      ) % (cubename)
            f.write(string)

    def compute_orbitals(self, debug=False):
        '''
        '''
        self.orb_dens = np.zeros(self.conf_dens.shape, dtype=float)
        for index in self.reactive_indexes:
            symbol = pt[self.atomnos[index]].symbol
            atom = self.rdkit_mol_object.GetAtoms()[index]
            neighbors_indexes = [a.GetIdx() for a in atom.GetNeighbors()]
            neighbors = len(neighbors_indexes)
            atom_type = atom_type_dict[symbol + str(neighbors)]
            if debug: print(f'DEBUG--> Reactive atom {index} is a {symbol} atom of {atom_type} type')
            atom_type.prop(self.atomcoords[0][index], self.atomcoords[0][neighbors_indexes])
            atom_type.stamp_orbital(self.conf_dens)
            self.orb_dens += atom_type.orb_dens








# TO DO: o = to do, x = done, w = working on it
# 
#   x   initialize density object class, loading conformer ensemble
#   x   write CoDe function, creating the scalar field
#   x   set up the conformational analysis inside the program
#   x   CoDe: weigh conformations based on their energy in box stamping
#   x   write a function that exports self.conf_dens to gaussian .cube format (VMD-readable)
#   x   implement different radius for different elements
#   x   reorder .sdf file generated and find out what the new reactive_indexes are
#   x   align entire ensemble based on reactive_vector and bulk around atom(s)
#   x   align conformers based on reactive atoms
#
#
#   o   initialize function that docks another object to current CoDe object
#   o   define the scoring function that ranks blob arrangements: reactive distance (+) and clashes (-)
#   o   sys.exit() if reactive atom is hydrogen - suggest to input their own ensemble
#   o   check if deuterium swap of hydrogen is a viable way of using it as reactive atom
#   o   implement the use of externally generated ensembles
#   o   move function defined in __init__ outside
#   o   create parameters file (vox_dim and stuff) -> IN CAPS
#   o   Alkyne sp: toroid, oriented along C-C axis
#   o   sp3: single sphere
#   o   enlarge box of conformational_density












# quit()

if __name__ == '__main__':

    # Completely WIP, aim is to translate a conformational ensemble 
    # in a probability density array, stored as a class property

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('Resources')

    test = Density_object('dienamine.xyz', 7, debug=True)
    # test = Density_object('funky_single.xyz', [15, 17], debug=True)
    # test = Density_object('CFClBrI.xyz', 2, debug=True)

    
    test.compute_CoDe(debug=True)

    test.compute_orbitals(debug=True)

    test.write_map(test.conf_dens, mapname='CoDe')

    test.write_map(test.orb_dens, mapname='orbitals')


    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # from mayavi import mlab
    # src = mlab.pipeline.scalar_field(test.box)
    # graph = mlab.pipeline.volume(src)
    # graph.edit_traits()
    # mlab.show()

