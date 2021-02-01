from cclib.io import ccread
import os
import numpy as np
# from spyrmsd import rmsd

class Density_object:
    """
    Conformational density 3D object from a conformational ensemble

    """
    def __init__(self, filename, debug=False):

        ccread_object = ccread(filename)
        self.atomcoords = np.array(ccread_object.atomcoords)

        if all([len(self.atomcoords[i])==len(self.atomcoords[0]) for i in range(1, len(self.atomcoords))]):     # Checking that ensemble has constant lenght
            if debug: print(f'\nDEBUG--> Initializing object {filename}\nDEBUG--> Found {len(self.atomcoords)} structures with {len(self.atomcoords[0])} atoms')
        else:
            raise Exception(f'Ensemble not recognized, no constant lenght. len are {[len(i) for i in self.atomcoords]}')

        self.atoms = np.array([atom for structure in self.atomcoords for atom in structure])                    # single list with all atom positions
        if debug: print(f'DEBUG--> Total of {len(self.atoms)} atoms\nDEBUG--> Atoms are {self.atoms}')

if __name__ == '__main__':

    # Completely WIP, aim is to translate a conformational ensemble 
    # in a probability density array, stored as a class property

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('Resources')
    test = Density_object('dienamine.xyz', debug=False)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = [atom[0] for atom in test.atoms]
    y = [atom[1] for atom in test.atoms]
    z = [atom[2] for atom in test.atoms]

    plot = ax.scatter(x, y, z)
    plt.show()
