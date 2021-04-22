import os
import sys
import warnings
import numpy as np
import networkx as nx
from rdkit import Chem
from parameters import *
from copy import deepcopy
from cclib.io import ccread
from rdkit.Chem import AllChem
from reactive_atoms_classes import *
from rdkit_conformational_search import csearch
from scipy.spatial.transform import Rotation as R
from subprocess import DEVNULL, STDOUT, check_call
warnings.simplefilter("ignore", UserWarning)

class CCReadError(Exception):
    pass

def kabsch(filename, indexes=None):
    '''
    Reads filename, aligns it based on indexes and
    writes an aligned .xyz file (filename_aligned.xyz)
    returning its name

    '''

    from rmsd import kabsch, centroid

    data = ccread(filename)
    reference, *targets = data.atomcoords
    reference = np.array(reference)
    targets = np.array(targets)

    indexes = slice(0,len(reference)) if not indexes else indexes

    r = reference - centroid(reference[indexes])
    ts = np.array([t - centroid(t[indexes]) for t in targets])

    output = []
    output.append(r)
    for target in ts:
        matrix = kabsch(r, target)
        output.append([matrix @ vector for vector in target])

    outname = filename.split('.')[0] + '_aligned.xyz'
    with open(outname, 'w') as f:
        for i, structure in enumerate(output):
            f.write(str(len(structure)))
            f.write(f'\nAligned Conformer {i}\n')
            for i, atom in enumerate(structure):
                f.write('%-5s %-8s %-8s %-8s\n' % (pt[data.atomnos[i]].symbol, round(atom[0], 6), round(atom[1], 6), round(atom[2], 6)))
        
    return outname

def graphize(coords, atomnos):
    '''
    :params coords: atomic coordinates as 3D vectors
    :params atomnos: atomic numbers as a list
    :return connectivity graph
    '''
    def d_min(e1, e2):
        return 1.2 * (pt[e1].covalent_radius + pt[e2].covalent_radius)
        # return 0.2 + (pt[e1].covalent_radius + pt[e2].covalent_radius)
    # if this is somewhat prone to bugs, this might help https://cccbdb.nist.gov/calcbondcomp1x.asp

    matrix = np.zeros((len(coords),len(coords)))
    for i in range(len(coords)):
        for j in range(i,len(coords)):
            if np.linalg.norm(coords[i]-coords[j]) < d_min(atomnos[i], atomnos[j]):
                matrix[i][j] = 1

    return nx.from_numpy_matrix(matrix)


class Hypermolecule:
    '''
    Conformational density and orbital density 3D object from a conformational ensemble.
    '''

    def __repr__(self):
        return self.rootname + f' {[str(atom) for atom in self.reactive_atoms_classes]}, ID = {id(self)}'

    def __init__(self, filename, reactive_atoms=None, debug=False, T=298.15):
        '''
        Initializing class properties: reading conformational ensemble file, aligning
        conformers to first and centering them in origin.

        :params filename:           Input file name. Can be anything, .xyz preferred
        :params reactive_atoms:     Index of atoms that will link during the desired reaction.
                                    May be either int or list of int.
        '''
        self.rootname = filename.split('.')[0]
        self.name = filename
        self.T = T
        self.debug = debug

        if reactive_atoms is None:
            reactive_atoms = self._set_reactive_atoms(filename)

        ccread_object = ccread(filename)
        if ccread_object is None:
            raise CCReadError(f'Cannot read file {filename}')

        coordinates = np.array(ccread_object.atomcoords)

        self.reactive_indexes = np.array(reactive_atoms)
        # alignment_indexes = self._alignment_indexes(ccread_object.atomcoords[0], ccread_object.atomnos, self.reactive_indexes)
        
        self.atomnos = ccread_object.atomnos
        self.position = np.array([0,0,0], dtype=float)  # used in Docker class
        self.rotation = np.identity(3)                  # used in Docker class - rotation matrix

        assert all([len(coordinates[i])==len(coordinates[0]) for i in range(1, len(coordinates))])     # Checking that ensemble has constant length
        if self.debug: print(f'DEBUG--> Initializing object {filename}\nDEBUG--> Found {len(coordinates)} structures with {len(coordinates[0])} atoms')


        self.centroid = np.sum(np.sum(coordinates, axis=0), axis=0) / (len(coordinates) * len(coordinates[0]))
        if self.debug: print('DEBUG--> Centroid was', self.centroid)
        self.atomcoords = coordinates - self.centroid
        self.graph = graphize(self.atomcoords[0], self.atomnos)
        # show_graph(self)
        self._inspect_reactive_atoms() # sets reactive atoms properties to rotate the ensemble correctly afterwards

        reactive_vector = []
        for structure in self.atomcoords:
            for index in self.reactive_indexes:
                reactive_vector.append(structure[index])
        reactive_vector = np.mean(np.array(reactive_vector), axis=0)

        self.atomcoords = np.array([self._orient_along_x(structure, reactive_vector) for structure in self.atomcoords])
        self.atomcoords = self._align_ensemble(filename, self.reactive_indexes)
        # After reading aligned conformers, they are stored as self.atomcoords after being translated to origin and aligned the reactive atom(s) to x axis.

        for reactive_atom, index in zip(self.reactive_atoms_classes, self.reactive_indexes):
                       
            reactive_atom.init(self, index, update=True)
            # update properties into reactive_atom class

        self.atoms = np.array([atom for structure in self.atomcoords for atom in structure])       # single list with all atom positions
        if self.debug: print(f'DEBUG--> Total of {len(self.atoms)} atoms')

        # self._compute_hypermolecule()

        self.centers = np.concatenate([r_atom.center for r_atom in self.reactive_atoms_classes])
        self.orb_vers = np.concatenate([norm(r_atom.center - r_atom.coord) for r_atom in self.reactive_atoms_classes])

    def _set_reactive_atoms(self, filename):
        '''
        Manually set the molecule reactive atoms from the ASE GUI, imposing
        constraints on the desired atoms.

        '''
        from ase import Atoms
        from ase.gui.gui import GUI
        from ase.gui.images import Images

        data = ccread(filename)
        coords = data.atomcoords[0]
        labels = ''.join([pt[i].symbol for i in data.atomnos])

        atoms = Atoms(labels, positions=coords)

        while atoms.constraints == []:
            print(('\nPlease, manually select the reactive atom(s) for molecule %s.'
                    '\nRotate with right click and select atoms by clicking. Multiple selections can be done by Ctrl+Click.'
                    '\nWith desired atom(s) selected, go to Tools -> Constraints -> Constrain, then close the GUI.') % (filename))

            GUI(images=Images([atoms]), show_bonds=True).run()

        return list(atoms.constraints[0].get_indices())

    def _get_ensemble_energies(self, filename):
        '''
        Reads file and returns an rdkit.Mol object with the first molecule
        and a list with the computed energy of all molecules at MMFF level.

        '''
        
        sdf_name = filename.split('.')[0] + '.sdf'
        check_call(f'obabel {filename} -o sdf -O {sdf_name}'.split(), stdout=DEVNULL, stderr=STDOUT)    # Bad, we should improve this

        mols = Chem.SDMolSupplier(sdf_name)
        # mols = [Chem.AddHs(m) for m in supplier]
        # mols = [m for m in supplier]
        energies = []
        for m in mols:
            # m = Chem.AddHs(m)
            # print(m.GetNumAtoms())
            ff = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m))
            ff.Initialize()
            ff.Minimize(maxIts=200)
            energies.append(ff.CalcEnergy())

        # self.smiles = Chem.MolToSmiles(mols[0])

        del mols
        os.remove(sdf_name)

        return energies


    def _compute_hypermolecule(self):
        '''
        '''

        BREADTH = 1e6
        self.energies = [0 for _ in self.atomcoords]
        # TODO: remove?

        self.hypermolecule_atomnos = []
        clusters = {i:{} for i in range(len((self.atomnos)))}  # {atom_index:{cluster_number:[position,times_found]}}
        for i, atom_number in enumerate(self.atomnos):
            atoms_arrangement = [conformer[i] for conformer in self.atomcoords]
            cluster_number = 0
            clusters[i][cluster_number] = [atoms_arrangement[0], 1]  # first structure has rel E = 0 so its weight is surely 1
            self.hypermolecule_atomnos.append(atom_number)
            radii = pt[atom_number].covalent_radius
            for j, atom in enumerate(atoms_arrangement[1:]):

                weight = np.exp(-self.energies[j+1] / BREADTH * 503.2475342795285 / self.T)
                # print(f'Atom {i} in conf {j+1} weight is {weight} - rel. E was {self.energies[j+1]}')

                for cluster_number, reference in deepcopy(clusters[i]).items():
                    if np.linalg.norm(atom - reference[0]) < radii:
                        clusters[i][cluster_number][1] += weight
                    else:
                        clusters[i][max(clusters[i].keys())+1] = [atom, weight]
                        self.hypermolecule_atomnos.append(atom_number)

        self.weights = [[] for _ in range(len(self.atomnos))]
        self.hypermolecule = []

        for i in range(len(self.atomnos)):
            for _, data in clusters[i].items():
                self.weights[i].append(data[1])
                self.hypermolecule.append(data[0])

        def flatten(array):
            out = []
            def rec(l):
                for e in l:
                    if type(e) in [list, np.ndarray]:
                        rec(e)
                    else:
                        out.append(float(e))
            rec(array)
            return out

        self.hypermolecule = np.asarray(self.hypermolecule)
        self.weights = np.array(self.weights).flatten()
        self.weights = np.array([weights / np.sum(weights) for weights in self.weights])
        self.weights = flatten(self.weights)

        self.dimensions = (max([coord[0] for coord in self.hypermolecule]) - min([coord[0] for coord in self.hypermolecule]),
                            max([coord[1] for coord in self.hypermolecule]) - min([coord[1] for coord in self.hypermolecule]),
                            max([coord[2] for coord in self.hypermolecule]) - min([coord[2] for coord in self.hypermolecule]))
    
    def _alignment_indexes(self, coords, atomnos, reactive_atoms):
        '''
        Return the indexes to align the molecule to, given a list of
        atoms that should be reacting. List is composed by reactive atoms
        plus adjacent atoms.
        :param coords: coordinates of a single molecule
        :param reactive atoms: int or list of ints
        :return: list of indexes
        '''

        self.graph = graphize(coords, atomnos)

        indexes = set()

        for atom in reactive_atoms:
            indexes |= set(list([(a, b) for a, b in self.graph.adjacency()][atom][1].keys()))
        if self.debug: print('DEBUG--> Alignment indexes are', list(indexes))
        return list(indexes)

    def _align_ensemble(self, filename, reactive_atoms):
        '''
        Align a set of conformers to the first one.
        Alignment is done on reactive atom(s) and its immediate neighbors.
        If ensembles has readable energies, they are kept, otherwise thy are calculated at MMFF level.

        filename            Input file name, must be a single structure file
        reactive_atoms      Index of atoms that will link during the desired reaction.
                            May be either int or list of int.
        return              Aligned coordinates

        '''
        # try:
        #     if type(reactive_atoms) == int:
        #         reactive_atoms = [reactive_atoms]
        # except Exception:
        #     raise Exception('Unrecognized reactive atoms IDs. Argument must either be one int or a list of ints.')

        self.reactive_indexes = np.array(reactive_atoms)

        if self.debug: print(f'DEBUG--> Read Conformational ensemble from file {filename} : {len(self.energies)} conformers found.')

        data = ccread(filename)  # if we could read energies directly from ensemble, actually take those
        try:
            assert len(data.scfenergies) == len(data.atomcoords)
            self.energies = data.scfenergies / 23.06054194532933
            self.energies -= min(self.energies)

            if self.debug: print(f'DEBUG--> Read relative energies from ensemble : {self.energies} kcal/mol')

        except:
            # self.energies = self._get_ensemble_energies(filename)
            self.energies = np.zeros(len(self.atomcoords), dtype=float)
            self.energies = np.array(self.energies) - np.min(self.energies)
            if self.debug: print(f'DEBUG--> Computed relative energies for the ensemble : {self.energies} kcal/mol')

        alignment_indexes = self._alignment_indexes(data.atomcoords[0], data.atomnos, self.reactive_indexes)

        del data

        outname = kabsch(filename, alignment_indexes)

        ccread_object = ccread(outname)
        os.remove(outname)    # <--- FINAL ALIGNED ENSEMBLE

        return ccread_object.atomcoords


    def _orient_along_x(self, array, vector):
        # TODO: check if this is still required
        '''
        :params array:    array of atomic coordinates arrays: len(array) structures with len(array[i]) atoms
        :params vector:   list of shape (1,3) with anchor vector to align to the x axis
        :return:          array, aligned so that vector is on x
        '''

        assert array.shape[1] == 3
        assert vector.shape == (3,)

        if len(self.reactive_atoms_classes) == 1:
            if str(self.reactive_atoms_classes[0]) == 'Single Bond':
                rotation_matrix = R.align_vectors(np.array([[1,0,0]]), np.array([vector]))[0].as_matrix()
                return np.array([rotation_matrix @ v for v in array])

            if str(self.reactive_atoms_classes[0]) in ('sp2','sp3','Ether'):
                return np.array([self.reactive_atoms_classes[0].alignment_matrix @ v for v in array])

        # if more than one reactive atom, only a rough alignment is done
        rotation_matrix = R.align_vectors(np.array([[1,0,0]]), np.array([vector]))[0].as_matrix()
        return np.array([rotation_matrix @ v for v in array])


    def _inspect_reactive_atoms(self):
        '''
        Control the type of reactive atoms and sets the class attribute self.reactive_atoms_classes
        '''
        self.reactive_atoms_classes = []

        for index in self.reactive_indexes:
            symbol = pt[self.atomnos[index]].symbol

            neighbors_indexes = list([(a, b) for a, b in self.graph.adjacency()][index][1].keys())
            neighbors_indexes.remove(index)

            atom_type = deepcopy(atom_type_dict[symbol + str(len(neighbors_indexes))])

            atom_type.init(self, index)
            # setting the pointer to reactive_atom class

            self.reactive_atoms_classes.append(atom_type)
            if self.debug: print(f'DEBUG--> Reactive atom {index+1} is a {symbol} atom of {atom_type} type. It is bonded to {neighbors} atom(s): {atom_type.neighbors_symbols}')
            # understanding the type of reactive atom in order to align the ensemble correctly and build the correct pseudo-orbitals

    def write_hypermolecule(self):
        '''
        '''

        hyp_name = self.rootname + '_hypermolecule.xyz'
        with open(hyp_name, 'w') as f:
            f.write(str(sum([len(atom.center) for atom in self.reactive_atoms_classes]) + len(self.hypermolecule)))
            f.write(f'\nHypermolecule originated from {self.rootname}\n')
            orbs =np.vstack([atom_type.center for atom_type in self.reactive_atoms_classes]).ravel()
            orbs = orbs.reshape((int(len(orbs)/3), 3))
            for orb in orbs:
                f.write('%-5s %-8s %-8s %-8s\n' % ('D', round(orb[0], 6), round(orb[1], 6), round(orb[2], 6)))
            for i, atom in enumerate(self.hypermolecule):
                f.write('%-5s %-8s %-8s %-8s\n' % (pt[self.hypermolecule_atomnos[i]].symbol, round(atom[0], 6), round(atom[1], 6), round(atom[2], 6)))

        
        print('Written .xyz orbital file -', hyp_name)



if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    test = {

        # 1 : ('Resources/indole/indole_ensemble.xyz', 6),
        # 2 : ('Resources/SN2/amine_ensemble.xyz', 10),
        # 3 : ('Resources/dienamine/dienamine_ensemble.xyz', 6),
        # 4 : ('Resources/SN2/flex_ensemble.xyz', [3, 5]),
        # 5 : ('Resources/SN2/flex_ensemble.xyz', None),

        # 6 : ('Resources/SN2/MeOH_ensemble.xyz', 1),
        # 7 : ('Resources/SN2/CH3Br_ensemble.xyz', 0),
        # 8 : ('Resources/bulk/tax.xyz', None),
        # 9 : ('Resources/DA/diene.xyz', (2,7)),
        # 10 : ('Resources/DA/dienophile.xyz', (3,5)),
        # 11 : ('Resources/SN2/MeOH_ensemble.xyz', (1,5)),
        # 12 : ('Resources/SN2/ketone_ensemble.xyz', 5),
        # 13 : ('Resources/NOH.xyz', (0,1)),

        14 : ('Resources/acid_ensemble.xyz', (3,25)),
        15 : ('Resources/dienamine/dienamine_ensemble.xyz', (6,23)),
        16 : ('Resources/maleimide.xyz', (0,5))


            }

    for i in (14, 15, 16):
        t = Hypermolecule(test[i][0], test[i][1])
        t._compute_hypermolecule()
        t.write_hypermolecule()


    quit()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def set_axes_equal(ax: plt.Axes):
        """Set 3D plot axes to equal scale.

        Make axes of 3D plot have equal scale so that spheres appear as
        spheres and cubes as cubes.  Required since `ax.axis('equal')`
        and `ax.set_aspect('equal')` don't work on 3D.
        """
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        _set_axes_radius(ax, origin, radius)

    def _set_axes_radius(ax, origin, radius):
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])


    col = {'H':'lightgrey',
           'C':'grey',
           'O':'tab:red',
           'N':'tab:blue',
           'S':'gold',
           'Br':'brown'}

    # for obj in [Hypermolecule(path, indexes) for path, indexes in test.values()]:
    # # for obj in [Hypermolecule('Resources/acid_ensemble.xyz', (3,25))]:
    #     labels_dict = {i:pt[n].symbol for i, n in enumerate(obj.atomnos)}

    #     color_list = [col[i] for i in labels_dict.values()]
    #     pos = nx.spring_layout(obj.graph)
    #     nx.draw_networkx_nodes(obj.graph, pos=pos, node_size=1000, nodelist=obj.reactive_indexes, node_color='coral', alpha=0.5)
    #     nx.draw(obj.graph, pos=pos, labels=labels_dict, node_color=color_list)
    #     plt.show()

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     x = [v[0] for v in obj.hypermolecule]
    #     y = [v[1] for v in obj.hypermolecule]
    #     z = [v[2] for v in obj.hypermolecule]

    #     ax.set_box_aspect([1,1,1])
    #     plot = ax.scatter(x, y, z, s=100, label=obj.rootname, c=obj.weights, vmin=0, vmax=1, cmap='YlOrRd', alpha=0.5)
    #     plt.colorbar(plot)
    #     set_axes_equal(ax)
    #     plt.tight_layout()
    #     plt.show()


    def show_graph(obj):
        import matplotlib.pyplot as plt
        labels_dict = {i:pt[n].symbol for i, n in enumerate(obj.atomnos)}

        color_list = [col[i] for i in labels_dict.values()]
        pos = nx.spring_layout(obj.graph)
        nx.draw_networkx_nodes(obj.graph, pos=pos, node_size=1000, nodelist=obj.reactive_indexes, node_color='coral', alpha=0.5)
        nx.draw(obj.graph, pos=pos, labels=labels_dict, node_color=color_list)
        plt.show()

    t = Hypermolecule(test[14][0], test[14][1])

    # en = test._get_ensemble_energies('Resources/funky/funky_ensemble.xyz')
    # en = test._get_ensemble_energies('Resources/SN2/flex_ensemble.xyz')
    # min_en = min(en)
    # # print([e - min_en for e in en])
    # print(en)


    # TO DO:

    # Probably a better idea for getting connectivity matrix is by using ase.geometry get_bonds() method. Low priority.