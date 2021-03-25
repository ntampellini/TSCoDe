'''
TSCoDe - Transition state Seeker from Conformational Density

(Work in Progress)

'''
from hypermolecule_class import Hypermolecule, pt
import numpy as np
from copy import deepcopy
from parameters import *
from pprint import pprint
from scipy.spatial.transform import Rotation as R
import os
import time
from cclib.io import ccread
from openbabel import openbabel as ob
from ase import Atoms
from ase.visualize import view
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.calculators.mopac import MOPAC
from optimization_methods import *


from subprocess import DEVNULL, STDOUT, check_call


def loadbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
	percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
	if iteration == total:
		print()

def norm(vec):
    return vec / np.linalg.norm(vec)

def cartesian_product(*arrays):
    return np.stack(np.meshgrid(*arrays), -1).reshape(-1, len(arrays))

def calc_positioned_conformers(self):
    self.positioned_conformers = np.array([[self.rotation @ v + self.position for v in conformer] for conformer in self.atomcoords])

def write_xyz(coords:np.array, atomnos:np.array, output, title='TEST'):
    '''
    output is of _io.TextIOWrapper type

    '''
    assert atomnos.shape[0] == coords.shape[0]
    assert coords.shape[1] == 3
    string = ''
    string += str(len(coords))
    string += f'\n{title}\n'
    for i, atom in enumerate(coords):
        string += '%s     % .6f % .6f % .6f\n' % (pt[atomnos[i]].symbol, atom[0], atom[1], atom[2])
    output.write(string)

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.

    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rot_mat_from_pointer(pointer, angle):
    '''
    Returns the rotation matrix that rotates a system around the given pointer
    of angle degrees. The algorithm is based on scipy quaternions.
    :params pointer: a 3D vector
    :params angle: a int/float, in degrees
    :return rotation_matrix: matrix that applied to a point, rotates it along the pointer
    '''
    assert pointer.shape[0] == 3

    pointer = norm(pointer)
    angle *= np.pi/180

    quat = np.array([np.sin(angle/2)*pointer[0],
                    np.sin(angle/2)*pointer[1],
                    np.sin(angle/2)*pointer[2],
                    np.cos(angle/2)])            # normalized quaternion, scalar last (i j k w)

    return R.from_quat(quat).as_matrix()

def polygonize(lengths):
    '''
    Returns coordinates for the polygon vertexes used in cyclical TS construction,
    as a list of vector couples specifying starting and ending point of each pivot 
    vector. For bimolecular TSs, returns vertexes for the centered superposition of
    two segments. For trimolecular TSs, returns triangle vertexes.

    :params vertexes: list of floats, used as polygon side lenghts.
    :return vertexes_out: list of vectors couples (start, end)
    '''
    assert len(lengths) in (2,3)
    lengths = sorted(lengths)
    arr = np.zeros((len(lengths),2,3))

    if len(lengths) == 2:
        arr[0,0] = np.array([-lengths[0]/2,0,0])
        arr[0,1] = np.array([+lengths[0]/2,0,0])
        arr[1,0] = np.array([-lengths[1]/2,0,0])
        arr[1,1] = np.array([+lengths[1]/2,0,0])

        vertexes_out = np.vstack(([arr],[arr]))
        vertexes_out[1,1] *= -1
        # THIS WORKS

    else:
        arr[0,1] = np.array([lengths[0],0,0])
        arr[1,0] = np.array([lengths[0],0,0])

        a = np.power(lengths[0], 2)
        b = np.power(lengths[1], 2)
        c = np.power(lengths[2], 2)
        x = (a-b+c)/(2*a**0.5)
        y = (c-x**2)**0.5

        arr[1,1] = np.array([x,y,0])
        arr[2,0] = np.array([x,y,0])

        vertexes_out = np.vstack(([arr],[arr],[arr],[arr],
                                  [arr],[arr],[arr],[arr]))

        swaps = [(1,2),(2,1),(3,1),(3,2),(4,0),(5,0),(5,1),(6,0),(6,2),(7,0),(7,1),(7,2)]

        for t,v in swaps:
            # triangle, vector couples to be swapped
            vertexes_out[t,v][[0,1]] = vertexes_out[t,v][[1,0]]

    return vertexes_out

class Docker:
    def __init__(self, *objects):
        self.objects = list(*objects)
        self.objects = sorted(self.objects, key=lambda x: len(x.atomnos), reverse=True)
    
    def setup(self, repeat=1):
        self.repeat = repeat

    def get_constrained_indexes(self):
        '''
        '''
        if all([len(molecule.reactive_atoms_classes) == 2 for molecule in self.objects]):
            # SHIET(?)

        else:
        # Two molecules, string algorithm, one constraint
            return [int(self.objects[0].reactive_indexes[0]),
                    int(self.objects[1].reactive_indexes[0] + len(self.objects[0].atomcoords[0]))]






    def string_embed(self):
        '''
        return threads: return embedded structures, with position and rotation attributes set, ready to be pumped
        into self.structures. Algorithm used is the "string" algorithm (see docs).
        '''
        assert len(self.objects) == 2
        # NOTE THAT THIS APPROACH WILL ONLY WORK FOR TWO MOLECULES, AND A REVISION MUST BE DONE TO GENERALIZE IT (BUT WOULD IT MAKE SENSE?)

        centers_indexes = cartesian_product(*[np.array(range(len(molecule.centers))) for molecule in self.objects])
        # for two mols with 3 and 2 centers: [[0 0][0 1][1 0][1 1][2 0][2 1]]
        
        threads = []
        for _ in range(len(centers_indexes)*self.repeat):
            threads.append([deepcopy(obj) for obj in self.objects])

        for t, thread in enumerate(threads): # each run is a different "regioisomer", repeated self.repeat times

            indexes = centers_indexes[t % len(centers_indexes)] # looping over the indexes that define "regiochemistry"
            repeated = True if t // len(centers_indexes) > 0 else False

            for i, molecule in enumerate(thread[1:]): #first molecule is always frozen in place, other(s) are placed with an orbital criterion

                ref_orb_vec = thread[i].centers[indexes[i]]  # absolute, arbitrarily long
                # reference molecule is the one just before molecule, so the i-th for how the loop is set up
                mol_orb_vec = molecule.centers[indexes[i+1]]

                ref_orb_vers = thread[i].orb_vers[indexes[i]]  # unit length, relative to orbital orientation
                # reference molecule is the one just before
                mol_orb_vers = molecule.orb_vers[indexes[i+1]]

                molecule.rotation = rotation_matrix_from_vectors(mol_orb_vers, -ref_orb_vers)

                molecule.position = thread[i].rotation @ ref_orb_vec + thread[i].position - molecule.rotation @ mol_orb_vec

                if repeated:

                    pointer = molecule.rotation @ mol_orb_vers
                    rotation = (np.random.rand()*2. - 1.) * np.pi    # random angle (radians) between -pi and pi
                    
                    delta_rot = rot_mat_from_pointer(pointer, rotation)
                    molecule.rotation = delta_rot @ molecule.rotation

                    molecule.position = thread[i].rotation @ ref_orb_vec + thread[i].position - molecule.rotation @ mol_orb_vec

        return threads

    def cyclical_embed(self):
        '''
        return threads: return embedded structures, with position and rotation attributes set, ready to be pumped
        into self.structures. Algorithm used is the "cyclical" algorithm (see docs).
        '''

        import itertools as it
        import more_itertools as mit
        import numpy as np

        def get_mols_indexes(n:int):
            '''
            params n: number of sides of the polygon to be constructed
            return: list of n-shaped tuples with indexes of pivots for each unique polygon.
                    These polygons are unique under rotations and reflections.
            '''
            perms = list(it.permutations(range(n)))
            ordered_perms = set([sorted(mit.circular_shifts(p))[0] for p in perms])
            unique_perms = []
            for p in ordered_perms:
                if sorted(mit.circular_shifts(reversed(p)))[0] not in unique_perms:
                    unique_perms.append(p)
            return sorted(unique_perms)
        
        def set_pivots(mol):
            '''
            params mol: Hypermolecule class
            Function sets the mol.pivots attribute, that is a list
            containing each vector connecting two orbitals on different atoms
            '''

            indexes = cartesian_product(*[range(len(atom.center)) for atom in mol.reactive_atoms_classes])
            # indexes of vectors in mol.center. Reactive atoms are necessarily 2 and so for one center on atom 0 and 
            # 2 centers on atom 2 we get [[0,0], [0,1], [1,0], [1,1]]

            mol.pivots = []
            mol.pivot_starts = []

            for i,j in indexes:
                v1 = mol.reactive_atoms_classes[0].center[i]
                v2 = mol.reactive_atoms_classes[1].center[j]
                pivot = v1 - v2
                mol.pivots.append(pivot)
                mol.pivot_starts.append(v1)

        print('Initializing CYCLICAL EMBED...')

        for molecule in self.objects:
            set_pivots(molecule)

        pivots_indexes = cartesian_product(*[range(len(mol.pivots)) for mol in self.objects])
        # indexes of pivots in each molecule self.pivots list. For three mols with 2 pivots each: [[0,0,0], [0,0,1], [0,1,0], ...]
       
        threads = []
        for p, pi in enumerate(pivots_indexes):

            loadbar(p, len(pivots_indexes), prefix=f'Embedding structures ')
            
            pivots = [self.objects[m].pivots[pi[m]] for m in range(len(self.objects))]
            # getting the active pivots for this run
            
            pivot_starts = [self.objects[m].pivot_starts[[np.all(r) for r in self.objects[m].pivots == pivots[m]].index(True)] for m in range(len(pivots))]
            # getting the origin of each pivot active in this run

            norms = np.linalg.norm(pivots, axis=1)
            # getting the pivots norms to feed into the polygonize function

            if True:
                # TO DO: checks that the disposition has the desired atoms facing each other

                for vecs in polygonize(norms):
                # getting vertexes to embed molecules with and iterating over start/end points

                    systematic_angles = cartesian_product(*[range(self.repeat) for _ in self.objects]) * 360/self.repeat

                    for angles in systematic_angles:

                        threads.append([deepcopy(obj) for obj in self.objects])
                        thread = threads[-1]
                        # generating the thread we are going to modify and setting the thread pointer

                        for i, vec_pair in enumerate(vecs):
                        # setting molecular positions and rotations (embedding)
                        # i is the molecule index, vecs is a tuple of start and end positions
                        # for the pivot vector

                            start, end = vec_pair

                            thread[i].position = start - pivot_starts[i]

                            angle = angles[i]
                            step_rotation = rot_mat_from_pointer(pivots[i], angle)
                            alignment_rotation = rotation_matrix_from_vectors(pivots[i], end-start)
                            thread[i].rotation = step_rotation @ alignment_rotation

            else:
                print('# TO DO: Rejected embed: not matching imposed criterion')
        loadbar(1, 1, prefix=f'Embedding structures ')
        return threads

######################################################################################################### RUN

    def run(self, debug=False):
        '''
        '''
        print()       

        t_start = time.time()


        if all([len(molecule.reactive_atoms_classes) == 2 for molecule in self.objects]):
        # Generating all possible combinations of conformations based on one of two algs
            embedded_structures = self.cyclical_embed()
        else:
            embedded_structures = self.string_embed()

                            # print(f'Random rotation of {rotation / np.pi * 180} degrees performed on candidate {t}')

############################################################################################################################

            # with open('ouroboros_setup.xyz', 'a') as f:
            #     structure = np.array([thread[0].rotation @ v + thread[0].position for v in thread[0].hypermolecule])
            #     atomnos = thread[0].hypermolecule_atomnos
            #     for molecule in thread[1:]:
            #         s = np.array([molecule.rotation @ v + molecule.position for v in molecule.hypermolecule])
            #         structure = np.concatenate((structure, s))
            #         atomnos = np.concatenate((atomnos, molecule.hypermolecule_atomnos))
            #     write_xyz(structure, atomnos, f, title=f'Arrangement_{t}')
        # quit()

############################################################################################################################

        atomnos = np.concatenate([molecule.atomnos for molecule in objects])
        # just a way not to lose track of atomic numbers associated with coordinates

        try:
            os.remove('TS_out.xyz')
        except:
            pass
        
        # Calculating new coordinates for embedded_structures and storing them in self.structures

        conf_number = [len(molecule.atomcoords) for molecule in objects]
        conf_indexes = cartesian_product(*[np.array(range(i)) for i in conf_number])
        # first index of each vector is the conformer number of the first molecule and so on

        self.structures = np.zeros((int(len(conf_indexes)*int(len(embedded_structures))), len(atomnos), 3)) # like atomcoords property, but containing multimolecular arrangements

        for geometry_number, geometry in enumerate(embedded_structures):

            for molecule in geometry:
                calc_positioned_conformers(molecule)

            for i, conf_index in enumerate(conf_indexes): # 0, [0,0,0] then 1, [0,0,1] then 2, [0,1,1]
                count_atoms = 0

                for molecule_number, conformation in enumerate(conf_index): # 0, 0 then 1, 0 then 2, 0 (first [] of outer for loop)
                    coords = geometry[molecule_number].positioned_conformers[conformation]
                    n = len(geometry[molecule_number].atomnos)
                    self.structures[geometry_number*len(conf_indexes)+i][count_atoms:count_atoms+n] = coords
                    count_atoms += n

        print(f'Generated {len(self.structures)} transition state candidates')

        ################################################# SANITY CHECK

        constrained_indexes = self.get_constrained_indexes()

        graphs = [mol.graph for mol in self.objects]
        
        msk = np.array([sanity_check(structure, atomnos, constrained_indexes, graphs, max_new_bonds=3) for structure in self.structures])
        self.structures = self.structures[msk]
        print(f'Discarded {len([b for b in msk if b == False])} candidates for compenetration ({len([b for b in msk if b == True])} left)')
        # Performing a sanity check for excessive compenetration on generated structures, discarding the ones that look too bad

        ################################################# PRUNING: SIMILARITY

        self.structures, mask = prune_conformers(self.structures, self.energies, debug=True)
        self.energies = self.energies[mask]
        print(f'Rejected {len(np.where(mask == False)[0])} candidates for similarity')

        ################################################# GEOMETRY OPTIMIZATION

        self.energies = np.zeros(len(self.structures))
        t_start = time.time()
        for i, structure in enumerate(deepcopy(self.structures)):
            loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')
            try:
                intermediate_geometry, _ = Hookean_optimization(structure, atomnos, constrained_indexes, graphs, calculator='Mopac', method='PM7')
                self.structures[i], self.energies[i] = optimize(intermediate_geometry, atomnos, constrained_indexes, graphs, calculator='Mopac', method='PM7')
            except ValueError:
                # ase will throw a ValueError if the output lacks a space in the "FINAL POINTS AND DERIVATIVES" table.
                # This occurs when one or more of them is not defined, that is when the calculation did not end well.
                # The easiest solution is to reject the structure and go on.
                self.structures[i] = None
                self.energies[i] = np.inf

        loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')
        t_end = time.time()
        print(f'Mopac PM7 optimization took {round(t_end-t_start, 2)} s ({round((t_end-t_start)/len(self.structures), 2)} s per structure)')

        ################################################# PRUNING: ENERGY

        self.energies = self.energies - np.min(self.energies)
        # print(self.energies)
        mask = self.energies < THRESHOLD_KCAL
        self.structures = self.structures[mask]
        self.energies = self.energies[mask]
        print(f'Rejected {len(np.where(mask == False)[0])} candidates for energy (Threshold set to {THRESHOLD_KCAL} kcal/mol)')

        ################################################# PRUNING: SIMILARITY (AGAIN)

        self.structures, mask = prune_conformers(self.structures, self.energies, debug=True)
        self.energies = self.energies[mask]
        print(f'Rejected {len(np.where(mask == False)[0])} candidates for similarity')

        ################################################# OUTPUT

        outname = 'TS_out.xyz'
        with open(outname, 'w') as f:        
            for i, structure in enumerate(self.structures):
                write_xyz(structure, atomnos, f, title=f'TS candidate {i}')
        print(f'Wrote {len(self.structures)} TS structures to {outname} file')


if __name__ == '__main__':

    try:
        os.remove('ouroboros_setup.xyz')
    except:
        pass

    # a = ['Resources/SN2/MeOH_ensemble.xyz', 1]
    a = ['Resources/dienamine/dienamine_ensemble.xyz', 6]
    # a = ['Resources/SN2/amine_ensemble.xyz', 22]
    # a = ['Resources/indole/indole_ensemble.xyz', 6]


    # b = ['Resources/SN2/CH3Br_ensemble.xyz', 0]
    b = ['Resources/SN2/ketone_ensemble.xyz', 2]


    # inp = [a,b]

    c = ['Resources/SN2/MeOH_ensemble.xyz', (1,5)]
    inp = [c,c,c]

    objects = [Hypermolecule(m[0], m[1]) for m in inp]

    docker = Docker(objects) # initialize docker with molecule density objects
    docker.setup(repeat=3) # set variables

    os.chdir('Resources/SN2')

    docker.run(debug=True)

    path = os.path.join(os.getcwd(), 'TS_out.vmd')
    # check_call(f'vmd -e {path}'.split(), stdout=DEVNULL, stderr=STDOUT)
    os.system(f'vmd -e {path}')