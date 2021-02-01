from cclib.io import ccread
import os
import numpy as np
from openbabel import openbabel as ob
from mayavi import mlab

# from spyrmsd import rmsd



class Density_object:
    '''
    Conformational density 3D object from a conformational ensemble

    '''

    def __init__(self, filename, debug=False):
        '''
        Initializing class properties: reading conformational ensemble file, aligning
        conformers to first and centering them in origin.

        '''
        def _align_ensemble(filename):

            '''
            Align a set of conformers to the first one, writing a new ensemble file

            :param filename:  input file name. Can be anything, .xyz preferred
            :return:          writes a new filename_aligned.xyz file and returns its name

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
                constructor.Align()
                constructor.UpdateCoords(m)

            mol_parser.SetOutFormat('.xyz')
            outlist = []
            for i, m in enumerate(allmols):
                name = rootname + f'_aligned_{i}.xyz'
                mol_parser.WriteFile(m, name)
                outlist.append(name)
            mol_parser.CloseOutFile()
            # CloseOutFile() call required to close last molecule stream since mol_parser won't be called again
            name = rootname + '_aligned.xyz'
            # mol_parser.FullConvert(outlist, name, None)
                # I really don't know why this doesn't work, i want to cry
                # http://openbabel.org/api/current/classOpenBabel_1_1OBConversion.shtml#a9d12b0f7f38951d2d1065fc7ddae4229
                # This should help on function syntax but I don't know much C++. I had to take a stupid detour from os.system :(
            os.system(f'obabel {rootname}_aligned*.xyz -o xyz -O {name}')
            for m in outlist:
                os.remove(m)
            return name

        ccread_object = ccread(_align_ensemble(filename))
        coordinates = np.array(ccread_object.atomcoords)

        if all([len(coordinates[i])==len(coordinates[0]) for i in range(1, len(coordinates))]):     # Checking that ensemble has constant lenght
            if debug: print(f'DEBUG--> Initializing object {filename}\nDEBUG--> Found {len(coordinates)} structures with {len(coordinates[0])} atoms')
        else:
            raise Exception(f'Ensemble not recognized, no constant lenght. len are {[len(i) for i in coordinates]}')

        # self.old = np.array([atom for structure in coordinates for atom in structure])   # debug, kill this
        center_of_mass = np.sum(np.sum(coordinates, axis=0), axis=0) / (len(coordinates) * len(coordinates[0]))
        if debug: print('DEBUG--> Center of mass is', center_of_mass)
        self.atomcoords = coordinates - center_of_mass
        # After reading aligned conformers, they are stored as self.atomcoords only after being aligned to origin

        self.atoms = np.array([atom for structure in self.atomcoords for atom in structure])       # single list with all atom positions
        if debug: print(f'DEBUG--> Total of {len(self.atoms)} atoms')

    def compute_CoDe(self, voxel_dim:float=0.1, stamp_size=1, hardness=5, debug=False):
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
        # defining single matrix to use as a stamp, basically a 3D sphere

        x_coords = np.array([pos[0] for pos in self.atoms])  # isolating x, y and z from self.atoms
        y_coords = np.array([pos[1] for pos in self.atoms])
        z_coords = np.array([pos[2] for pos in self.atoms])
        size = (max(x_coords) - min(x_coords), max(y_coords) - min(y_coords), max(z_coords) - min(z_coords))
        shape = (int(np.ceil(size[0]/voxel_dim)) + stamp_len, int(np.ceil(size[1]/voxel_dim)) + stamp_len, int(np.ceil(size[2]/voxel_dim)) + stamp_len)
        # size of box, in number of voxels (i.e. matrix items), is defined by how many of them are needed to include all atoms
        # given the fact that they measure {voxel_dim} Angstroms. To these numbers, {stamp_len} voxels are added to each axis
        # to ensure that there is enough space for upcoming stamping of density information for atoms close to the boundary.
        if debug: print('DEBUG--> Box shape is', shape, 'voxels')
        self.box = np.zeros(shape, dtype=float)
        # defining box dimensions based on molecule size and voxel input

        offset = round(stamp_len/2)
        for atom in self.atoms:                                          # adding density to array elements
            x_pos = round(atom[0]/voxel_dim) - offset
            y_pos = round(atom[1]/voxel_dim) - offset
            z_pos = round(atom[2]/voxel_dim) - offset
            print('Stamping to xyz:', x_pos, y_pos, z_pos)
            self.box[x_pos : x_pos + stamp_len, y_pos : y_pos + stamp_len, z_pos : z_pos + stamp_len] += self.stamp
        # self.box = self.box / max(self.box.reshape((1,np.prod(shape))))  # normalize box values
















# quit()

if __name__ == '__main__':

    # Completely WIP, aim is to translate a conformational ensemble 
    # in a probability density array, stored as a class property

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('Resources')
    test = Density_object('dienamine.xyz', debug=True)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = [atom[0] for atom in test.atoms]
    y = [atom[1] for atom in test.atoms]
    z = [atom[2] for atom in test.atoms]

    # plot = ax.scatter(x, y, z, color='b')
    # plt.show()

    test.compute_CoDe(0.1, debug=True)

    src = mlab.pipeline.scalar_field(test.box)
    graph = mlab.pipeline.volume(src)
    graph.edit_traits()
    mlab.show()

