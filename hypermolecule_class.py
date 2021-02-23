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

        if not reactive_atoms:
            reactive_atoms = self._set_reactive_atoms(filename)

        ccread_object = self._align_ensemble(filename, reactive_atoms)

        coordinates = np.array(ccread_object.atomcoords)


        self.atomnos = ccread_object.atomnos
        self.position = np.array([0,0,0], dtype=float)  # used in Docker class
        self.rotation = np.identity(3)                  # used in Docker class - rotation matrix

        if all([len(coordinates[i])==len(coordinates[0]) for i in range(1, len(coordinates))]):     # Checking that ensemble has constant length
            if self.debug: print(f'DEBUG--> Initializing object {filename}\nDEBUG--> Found {len(coordinates)} structures with {len(coordinates[0])} atoms')
        else:
            raise Exception(f'Ensemble not recognized, no constant lenght. len are {[len(i) for i in coordinates]}')

        self.centroid = np.sum(np.sum(coordinates, axis=0), axis=0) / (len(coordinates) * len(coordinates[0]))
        if self.debug: print('DEBUG--> Centroid was', self.centroid)
        self.atomcoords = coordinates - self.centroid

        self._inspect_reactive_atoms() # sets reactive atoms properties to rotate the ensemble correctly afterwards

        if type(self.reactive_indexes) is int:
            reactive_vector = np.mean(np.array([structure[self.reactive_indexes] for structure in self.atomcoords]), axis=0)
        else:
            reactive_vector = []
            for structure in self.atomcoords:
                for index in self.reactive_indexes:
                    reactive_vector.append(structure[index])
            reactive_vector = np.mean(np.array(reactive_vector), axis=0)

        self.atomcoords = np.array([self._orient_along_x(structure, reactive_vector) for structure in self.atomcoords])
        # After reading aligned conformers, they are stored as self.atomcoords after being translated to origin and aligned the reactive atom(s) to x axis.

        for i, index in enumerate(self.reactive_indexes):
            symbol = pt[self.atomnos[index]].symbol
            atom_type = self.reactive_atoms_classes[i]

            neighbors_indexes = list([(a, b) for a, b in self.graph.adjacency()][index][1].keys())
            neighbors_indexes.remove(index)
            
            atom_type.prop(self.atomcoords[0][index], self.atomcoords[0][neighbors_indexes], symbol)
            # pumping updated properties into reactive_atom class

        self.atoms = np.array([atom for structure in self.atomcoords for atom in structure])       # single list with all atom positions
        if self.debug: print(f'DEBUG--> Total of {len(self.atoms)} atoms')

        self._compute_hypermolecule()

        self.centers = np.concatenate([r_atom.center for r_atom in self.reactive_atoms_classes])
        self.orb_vers = np.concatenate([norm(r_atom.center - r_atom.coord) for r_atom in self.reactive_atoms_classes])

    def _set_reactive_atoms(self, filename):
        '''
        Manually set the molecule reactive atoms from the ASE GUI, imposing
        constraints on the desired atoms.

        '''
        from ase import Atoms
        from ase.visualize import view

        data = ccread(filename)
        coords = data.atomcoords[0]
        labels = ''.join([pt[i].symbol for i in data.atomnos])

        atoms = Atoms(labels, positions=coords)

        while atoms.constraints == []:
            print(('\nPlease, manually select the reactive atom(s) for molecule %s.'
                   '\nSelect an atom by clicking on it, multiple selection can be done by Ctrl+Click.'
                   '\nWith desired atom(s) selected, go to Tools -> Constraints -> Constrain, then close the GUI.'
                   '\nBond view toggle with Ctrl+B\n') % (filename))
            atoms.edit()

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

        self.smiles = Chem.MolToSmiles(mols[0])

        del mols
        os.remove(sdf_name)

        return energies


    def _compute_hypermolecule(self):
        '''
        '''
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


        self.hypermolecule = np.asarray(self.hypermolecule)
        self.weights = np.array(self.weights).flatten()
        self.weights = np.asarray([weights / np.sum(weights) for i, weights in enumerate(self.weights)])

        self.dimensions = (max([coord[0] for coord in self.hypermolecule]) - min([coord[0] for coord in self.hypermolecule]),
                            max([coord[1] for coord in self.hypermolecule]) - min([coord[1] for coord in self.hypermolecule]),
                            max([coord[2] for coord in self.hypermolecule]) - min([coord[2] for coord in self.hypermolecule]))

    
    def _alignment_indexes(self, coords, reactive_atoms):
        '''
        Return the indexes to align the molecule to, given a list of
        atoms that should be reacting. List is composed by reactive atoms
        plus adjacent atoms.
        :param coords: coordinates of a single molecule
        :param reactive atoms: int or list of ints
        :return: list of indexes
        '''

        matrix = np.zeros((len(coords),len(coords)))
        for i in range(len(coords)):
            for j in range(i,len(coords)):
                if np.linalg.norm(coords[i]-coords[j]) < 1.6:
                    matrix[i][j] = 1

        self.graph = nx.from_numpy_matrix(matrix)

        # import matplotlib.pyplot as plt
        # nx.draw(self.graph, with_labels=True)
        # plt.show()

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
        return              Writes a new filename_aligned.xyz file and returns its name

        '''
        try:
            if type(reactive_atoms) == int:
                reactive_atoms = [reactive_atoms]
        except Exception:
            raise Exception('Unrecognized reactive atoms IDs. Argument must either be one int or a list of ints.')

        self.reactive_indexes = np.array(reactive_atoms) - 1 # they count from 1, we count from 0

        self.energies = self._get_ensemble_energies(filename)

        if self.debug: print(f'DEBUG--> Read Conformational ensemble from file {filename} : {len(self.energies)} conformers found.')

        data = ccread(filename)  # if we could read energies directly from ensemble, actually take those
        try:
            assert len(data.scfenergies) == len(data.atomcoords)
            self.energies = data.scfenergies / 23.06054194532933
            self.energies -= min(self.energies)

            if self.debug: print(f'DEBUG--> Read relative energies from ensemble : {self.energies} kcal/mol')

        except:
            self.energies = np.array(self.energies) - min(self.energies)
            if self.debug: print(f'DEBUG--> Computed relative energies for the ensemble : {self.energies} kcal/mol')

        alignment_indexes = self._alignment_indexes(data.atomcoords[0], self.reactive_indexes)

        del data

        if alignment_indexes is None:               # Eventually we should raise an error I think, but we could also warn and leave this
            print(f'UNABLE TO UNDERSTAND REACTIVE ATOMS INDEX(ES): Ensemble for {self.rootname} aligned on all atoms, may generate undesired results.')

        outname = kabsch(filename, alignment_indexes)

        ccread_object = ccread(outname)
        os.remove(outname)    # <--- FINAL ALIGNED ENSEMBLE

        return ccread_object


    def _orient_along_x(self, array, vector):
        # IMPLEMENT DIFFERENT ALIGNMENT ON THE BASE OF SELF.REACTIVE_ATOMS_CLASSES
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

            neighbors = len(neighbors_indexes)
            atom_type = deepcopy(atom_type_dict[symbol + str(neighbors)])

            atom_type.prop(self.atomcoords[0][index], self.atomcoords[0][neighbors_indexes], symbol, [pt[self.atomnos[i]].symbol for i in neighbors_indexes])
            # pumping required properties into reactive_atom class:
            # reactive atom coordinates, symbol, neighbors coordinates, neighbor symbols

            self.reactive_atoms_classes.append(atom_type)
            if self.debug: print(f'DEBUG--> Reactive atom {index+1} is a {symbol} atom of {atom_type} type. It is bonded to {neighbors} atom(s): {atom_type.neighbors_symbols}')
            # understanding the type of reactive atom in order to align the ensemble correctly and build the correct pseudo-orbitals

    # def show_drawing(self):
    #     '''
    #     Shows a plot made with rdkit to confirm that reactive atoms selected are correct.

    #     '''
    #     from rdkit.Chem import Draw
    #     from rdkit.Chem.Draw import rdMolDraw2D
    #     import matplotlib.pyplot as plt
    #     import matplotlib.image as mpimg

    #     mol = Chem.MolFromSmiles(self.smiles)
    #     for atom in mol.GetAtoms():
    #         atom.SetAtomMapNum(0)
    #     d = rdMolDraw2D.MolDraw2DCairo(500, 500)

    #     indexes = [self.reactive_indexes] if type(self.reactive_indexes) is int else self.reactive_indexes
    #     indexes = [int(i) for i in indexes]

    #     rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=indexes)

    #     d.drawOptions().addStereoAnnotation = True
    #     # d.drawOptions().addAtomIndices = True
    #     d.DrawMolecule(mol)
    #     d.FinishDrawing()
    #     d.WriteDrawingText('temp.png')
    #     img = mpimg.imread('temp.png')

    #     plot = plt.imshow(img)
    #     plt.tight_layout()
    #     plt.axis('off')
    #     plt.show()



    def write_hypermolecule(self):
        '''
        '''

        hyp_name = self.rootname + '_hypermolecule.xyz'
        with open(hyp_name, 'w') as f:
            f.write(str(sum([len(atom.center) for atom in self.reactive_atoms_classes]) + len(self.hypermolecule)))
            f.write(f'\nHypermolecule originated from {self.rootname}\n')
            orbs =np.array([atom_type.center for atom_type in self.reactive_atoms_classes]).flatten()
            orbs = orbs.reshape((int(len(orbs)/3), 3))
            for orb in orbs:
                f.write('%-5s %-8s %-8s %-8s\n' % ('D', round(orb[0], 6), round(orb[1], 6), round(orb[2], 6)))
            for i, atom in enumerate(self.hypermolecule):
                f.write('%-5s %-8s %-8s %-8s\n' % (pt[self.hypermolecule_atomnos[i]].symbol, round(atom[0], 6), round(atom[1], 6), round(atom[2], 6)))

        
        print('Written .xyz orbital file -', hyp_name)



if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # test = Hypermolecule('Resources/indole/indole_ensemble.xyz', 6, debug=True)
    # test = Hypermolecule('Resources/SN2/amine_ensemble.xyz', 11, debug=True)
    # test = Hypermolecule('Resources/dienamine/dienamine_ensemble.xyz', 7, debug=True)
    # test = Hypermolecule('Resources/SN2/flex_ensemble.xyz', [3, 5], debug=True)
    test = Hypermolecule('Resources/SN2/flex_ensemble.xyz', debug=True)
    # test.show_drawing()
    test.write_hypermolecule()
    # en = test._get_ensemble_energies('Resources/funky/funky_ensemble.xyz')
    # en = test._get_ensemble_energies('Resources/SN2/flex_ensemble.xyz')
    # min_en = min(en)
    # # print([e - min_en for e in en])
    # print(en)