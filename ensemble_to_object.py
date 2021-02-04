from cclib.io import ccread
import os
import numpy as np
from openbabel import openbabel as ob

# from spyrmsd import rmsd

def loadbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
	percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
	if iteration == total:
		print()


class Density_object:
    '''
    Conformational density 3D object from a conformational ensemble.

    filename            Input file name. Can be anything, .xyz preferred
    reactive_atoms      Index of atoms that will link during the desired reaction.
                        May be either int or list of int.
    return              Writes a new filename_aligned.xyz file and returns its name


    '''

    def __init__(self, filename, reactive_atoms, debug=False):
        '''
        Initializing class properties: reading conformational ensemble file, aligning
        conformers to first and centering them in origin.

        '''
        def _generate_ensemble(filename):
        '''
        Performs a conformational search with the openbabel package and returns {filename}_ensemble.xyz name

        '''
        rootname, ext = filename.split('.')
        mol_parser = ob.OBConversion()
        mol_parser.SetInFormat(ext)
        mol = ob.OBMol()
        mol_parser.ReadFile(mol, filename)
        cs = ob.OBConformerSearch()
        cs.Setup(mol,
                        30,  # numConformers
                        5,   # numChildren
                        5,   # mutability
                        25); # convergence
        cs.Search()
        cs.GetConformers(mol)
        print(f'Performed Conformational search on {filename}: {mol.NumConformers()} conformers found.')

        OBConversion conv;
        mol_parser.SetOutFormat('xyz')
        for i in range(len(mol.NumConformers)):
            mol.SetConformer(i)
            mol_parser.Write(mol)




        def _align_ensemble(filename, reactive_atoms): # needs to be rewritten to align based on reactive atoms

            '''
            Align a set of conformers to the first one, writing a new ensemble file.
            Alignment is done on reactive atom(s) and its immediate neighbors.

            filename            Input file name. Can be anything, .xyz preferred
            reactive_atoms      Index of atoms that will link during the desired reaction.
                                May be either int or list of int.
            return              Writes a new filename_aligned.xyz file and returns its name

            '''

            rootname, ext = filename.split('.')
            mol_parser = ob.OBConversion()
            mol_parser.SetInFormat(ext)
            mol = ob.OBMol()
            notatend = mol_parser.ReadFile(mol, filename)
            allmols = []
            while notatend:
                allmols.append(mol)
                mol = ob.OBMol()
                notatend = mol_parser.Read(mol)
            # Crazy but this is standard Openbabel procedure. Anyway,looks like a pybel method is available
            # to doall this actually https://bmcchem.biomedcentral.com/articles/10.1186/1752-153X-2-5

            del mol
            ref = allmols[0]                  # Setting reference molecule to align all others, and aligning all to first
            constructor = ob.OBAlign()
            constructor.SetRefMol(ref)
            for m in allmols[1:]:
                constructor.SetTargetMol(m)
                constructor.Align()           # TO DO: ALIGN TO REACTIVE INDEXES AND NOT RANDOMLY - NEIGHBORS OF INDEX 6 SHOULD BE 1, 7, 19
                constructor.UpdateCoords(m)

            mol_parser.SetOutFormat('.xyz')
            outlist = []
            for i, m in enumerate(allmols):
                name = rootname + f'_aligned_{i}.xyz'
                mol_parser.WriteFile(m, name)
                outlist.append(name)
            mol_parser.CloseOutFile()
            # CloseOutFile() call required to close last molecule stream since mol_parser won't be called again
            name = rootname + '_aligned_ensemble.xyz'
            # mol_parser.FullConvert(outlist, name, None)
                # I really don't know why this doesn't work, i want to cry
                # http://openbabel.org/api/current/classOpenBabel_1_1OBConversion.shtml#a9d12b0f7f38951d2d1065fc7ddae4229
                # This should help on function syntax but I don't know much C++. I had to take a stupid detour from os.system :(
            os.system(f'obabel {rootname}_aligned*.xyz -o xyz -O {name}')
            for m in outlist:
                os.remove(m)
            return name
           
        try:
            ccread_object = ccread(filename)
            if len(ccread_object.atomcoords) == 1:
                filename = _generate_ensemble(filename)
        except:
            raise Exception('The input file cannot be read through cclib package. Try a .xyz conformational ensemble or a single .xyz file.')

        if os.path.isfile(filename.split('.')[0] + '_aligned_ensemble.xyz'):
            ccread_object = ccread(filename.split('.')[0] + '_aligned_ensemble.xyz')
        else:
            ccread_object = ccread(_align_ensemble(filename, reactive_atoms))

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
        # self.energies = np.array([x for structure in self.atomcoords]) 

    def compute_CoDe(self, voxel_dim:float=0.1, stamp_size=1, hardness=1, debug=False):
        '''
        Computing conformational density for the ensemble.

        :param voxel_dim:    size of the square Voxel side, in Angstroms
        :param stamp_size:   radius of sphere used as stamp, in Angstroms
        :hardness:           steepness of radial probability decay (gaussian, k in e^(-kr^2))
        :return:             writes a new filename_aligned.xyz file and returns its name

        '''
        stamp_len = round((stamp_size/voxel_dim))
        if debug: print('DEBUG--> Stamp size (sphere diameter) is', stamp_len, 'voxels')
        self.stamp = np.zeros((stamp_len, stamp_len, stamp_len))
        for x in range(stamp_len - 1):                                   # probably optimizable
            for y in range(stamp_len - 1):
                for z in range(stamp_len - 1):
                    r_2 = (x - stamp_len/2)**2 + (y - stamp_len/2)**2 + (z - stamp_len/2)**2
                    self.stamp[x, y, z] = np.exp(-hardness*r_2)
        # defining single matrix to use as a stamp, basically a 3D sphere with values decaying with e^(-kr^2)

        x_coords = np.array([pos[0] for pos in self.atoms])  # isolating x, y and z from self.atoms
        y_coords = np.array([pos[1] for pos in self.atoms])
        z_coords = np.array([pos[2] for pos in self.atoms])

        min_x = min(x_coords)
        min_y = min(y_coords)
        min_z = min(z_coords)

        self.ensemble_origin = np.array([min_x + stamp_len*voxel_dim, 
                                         min_y + stamp_len*voxel_dim,
                                         min_z + stamp_len*voxel_dim])        # vector connecting ensemble centroid with box origin (corner), used later

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

        self.box = np.zeros(shape, dtype=float)
        if debug: print('DEBUG--> Box shape is', shape, 'voxels')
        # defining box dimensions based on molecule size and voxel input

        for atom in self.atoms:                                          # adding density to array elements
            x_pos = round((atom[0] - min_x) / voxel_dim + outline/2)
            y_pos = round((atom[1] - min_y) / voxel_dim + outline/2)
            z_pos = round((atom[2] - min_z) / voxel_dim + outline/2)
            # print('Stamping to xyz:', x_pos, y_pos, z_pos)
            self.box[x_pos : x_pos + stamp_len, y_pos : y_pos + stamp_len, z_pos : z_pos + stamp_len] += self.stamp
        self.box = self.box / max(self.box.reshape((1,np.prod(shape)))[0])  # normalize box values

        self.voxdim = voxel_dim

    def write_cubefile(self, debug=False):
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
                if debug: print(f'DEBUG--> Writing .cube file with {total} scalar values')
                print_list = []
                for x in range(self.box.shape[0]):        # Main loop: z is inner, y is middle and x is outer.
                    for y in range(self.box.shape[1]):
                        for z in range(self.box.shape[2]):
                            print_list.append('{:.5e} '.format(self.box[x, y, z]).upper())
                            count += 1
                            loadbar(count, total, prefix='Writing .cube file... ')
                            if count % 6 == 5:
                                print_list.append('\n')
                print_list.append('\n')
                f.write(''.join(print_list))
        except:
            raise Exception(f'No CoDe data in {self.name} Density Object Class: write_cube method should be called only after compute_CoDe method.')


# TO DO: o = to do, x = done
# 
#   x   initialize density object class, loading conformer ensemble
#   o   align conformer ensemble besed on reactive atoms
#   x   write CoDe function, creating the scalar field
#   o   set up the conformational analysis inside the program
#   o   CoDe: weigh conformations based on their energy in box stamping - requires
#           substituting .xyz extension for ensemble or autonomously ranking conformations. We'll see.
#   o   initialize function that docks another object to current CoDe object
#   o   define the scoring function that ranks blob arrangements: reactive distance (+) and clashes (-)
#   x   write a function that exports self.box to gaussian .cube format (VMD-readable)
#   o   fix .cube file generated not aligned with density object












# quit()

if __name__ == '__main__':

    # Completely WIP, aim is to translate a conformational ensemble 
    # in a probability density array, stored as a class property

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('Resources')
    test = Density_object('dienamine.xyz', 6, debug=True)
    # test.compute_CoDe(0.5, debug=True)

    # test.write_cubefile(debug=True)


    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # from mayavi import mlab
    # src = mlab.pipeline.scalar_field(test.box)
    # graph = mlab.pipeline.volume(src)
    # graph.edit_traits()
    # mlab.show()

