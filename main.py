'''
TSCoDe - Transition state Seeker from Conformational Density

(Work in Progress)

'''
from hypermolecule_class import Hypermolecule, pt
import numpy as np
from copy import deepcopy
from parameters import *
import os
import time
from cclib.io import ccread
from ase import Atoms
from ase.visualize import view
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.calculators.mopac import MOPAC
from optimization_methods import *
from subprocess import DEVNULL, STDOUT, check_call
from linalg_tools import *

class ZeroCandidatesError(Exception):
    pass

class InputError(Exception):
    pass

stamp = time.ctime().replace(' ','_').replace(':','-')
log = open(f'TSCoDe_log_{stamp}.txt', 'a', buffering=1)

def log_print(string=''):
    print(string)
    string += '\n'
    log.write(string)

def loadbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
	percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
	if iteration == total:
		print()

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

class Docker:
    def __init__(self, *objects):
        self.objects = list(*objects)
    
    def setup(self, steps=6, optimize=True):
        '''
        :params steps: int, number of steps for the sampling of rotations. A value of 3 corresponds
                        to three 120° turns for each molecule.
        :params optimize: bool, whether to run the semiempirical optimization calculations
        '''
        self.rotation_steps = int(steps)
        self.optimize = optimize

        if len(self.objects) in (2,3):
        # Calculating the number of conformation combinations based on embed type
            if all([len(molecule.reactive_atoms_classes) == 2 for molecule in self.objects]):

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
                    mol.pivot_means = []

                    for i,j in indexes:
                        v1 = mol.reactive_atoms_classes[0].center[i]
                        v2 = mol.reactive_atoms_classes[1].center[j]
                        pivot = v2 - v1
                        mol.pivots.append(pivot)
                        mol.pivot_means.append(np.mean((v1,v2), axis=0))

                for molecule in self.objects:
                    set_pivots(molecule)

                self.embed = 'cyclical'
                self.candidates = self.rotation_steps**len(self.objects)*np.prod([len(mol.atomcoords) for mol in self.objects])
                self.candidates *= np.prod([len(mol.pivots) for mol in self.objects])*8
                # The number 8 is the number of different triangles originated from three oriented vectors


            elif all([len(molecule.reactive_atoms_classes) == 1 for molecule in self.objects]) and len(self.objects) == 2:
                self.embed = 'string'
                self.candidates = self.rotation_steps**len(self.objects)*np.prod([len(mol.atomcoords) for mol in self.objects])
                self.candidates *= 2
                # The number 2 is the number of different arrangementsof two oriented vectors (parallel, antiparallel)

            else:
                raise InputError('Bad input - The only molecular configurations accepted are: 1) two or three molecules with two reactive centers each or 2) two molecules with one reactive center each.')
        else:
            raise InputError('Bad input - too many molecules specified (3 max).')

        log_print(f'Setup performed correctly. {self.candidates} candidates will be generated.')


    def get_string_constrained_indexes(self):
        '''
        Get constrained indexes referring to the transition states.
        :return: list of lists consisting in atomic pairs to be constrained.
        '''
        # Two molecules, string algorithm, one constraint for all
        return [[int(self.objects[0].reactive_indexes[0]),
                 int(self.objects[1].reactive_indexes[0] + len(self.objects[0].atomcoords[0]))] for _ in range(len(self.structures))]
                 # TO DO: check if it works and if the second list comprehension is correct

    def get_cyclical_reactive_indexes(self, n):
        '''
        :params n: index of the n-th disposition of vectors yielded by the polygonize function.
        :return: list of index couples, to be constrained during the partial optimization.
        '''
        cumulative_pivots_ids = [[mol.reactive_indexes[0]+sum(len(m.atomnos) for m in self.objects[0:self.objects.index(mol)]),
                                  mol.reactive_indexes[1]+sum(len(m.atomnos) for m in self.objects[0:self.objects.index(mol)])] for mol in self.objects]

        def orient(i,ids,n):
            if swaps[n][i] == True:
                return list(reversed(ids))
            else:
                return ids

        if len(self.objects) == 2:

            swaps = [(0,0),
                     (0,1)]

            oriented = [orient(i,ids,n) for i, ids in enumerate(cumulative_pivots_ids)]
            couples = [[oriented[0][0], oriented[1][0]], [oriented[0][1], oriented[1][1]]]
            return couples

        else:
            swaps = [(0,0,0),
                     (0,0,1),
                     (0,1,0),
                     (0,1,1),
                     (1,0,0),
                     (1,1,0),
                     (1,0,1),
                     (1,1,1)]

            oriented = [orient(i,ids,n) for i, ids in enumerate(cumulative_pivots_ids)]
            couples = [[oriented[0][1], oriented[1][0]], [oriented[1][1], oriented[2][0]], [oriented[2][1], oriented[0][0]]]
            return couples

    def string_embed(self):
        '''
        return threads: return embedded structures, with position and rotation attributes set, ready to be pumped
        into self.structures. Algorithm used is the "string" algorithm (see docs).
        '''
        assert len(self.objects) == 2
        # NOTE THAT THIS APPROACH WILL ONLY WORK FOR TWO MOLECULES, AND A REVISION MUST BE DONE TO GENERALIZE IT (BUT WOULD IT MAKE SENSE?)

        log_print('Initializing string embed...')

        constrained_indexes = self.get_string_constrained_indexes()

        centers_indexes = cartesian_product(*[np.array(range(len(molecule.centers))) for molecule in self.objects])
        # for two mols with 3 and 2 centers: [[0 0][0 1][1 0][1 1][2 0][2 1]]
        
        threads = []
        for _ in range(len(centers_indexes)*self.rotation_steps):
            threads.append([deepcopy(obj) for obj in self.objects])

        for t, thread in enumerate(threads): # each run is a different "regioisomer", repeated self.rotation_steps times

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
                    # rotation = (np.random.rand()*2. - 1.) * np.pi    # random angle (radians) between -pi and pi
                    rotation = t % self.rotation_steps * 360/self.rotation_steps       # sistematic incremental step angle
                    
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
        # this was actually ready for n-gons implementation, but I think we will never use with n>3
        

        log_print('Initializing cyclical embed...')

        pivots_indexes = cartesian_product(*[range(len(mol.pivots)) for mol in self.objects])
        # indexes of pivots in each molecule self.pivots list. For three mols with 2 pivots each: [[0,0,0], [0,0,1], [0,1,0], ...]
       
        threads = []
        constrained_indexes = []
        for p, pi in enumerate(pivots_indexes):

            loadbar(p, len(pivots_indexes), prefix=f'Embedding structures ')
            
            pivots = [self.objects[m].pivots[pi[m]] for m in range(len(self.objects))]
            # getting the active pivots for this run
            
            pivot_means = [self.objects[m].pivot_means[[np.all(r) for r in self.objects[m].pivots == pivots[m]].index(True)] for m in range(len(pivots))]
            # getting the mean point of each pivot active in this run

            norms = np.linalg.norm(pivots, axis=1)
            # getting the pivots norms to feed into the polygonize function

            for v, vecs in enumerate(polygonize(norms)):
            # getting vertexes to embed molecules with and iterating over start/end points

                ids = self.get_cyclical_reactive_indexes(v)
                # get indexes of atoms that face each other

                if True: # if all([couple in couples for couple in ids]) or smth
                    # TO DO: checks that the disposition has the desired atoms facing each other

                    systematic_angles = cartesian_product(*[range(self.rotation_steps) for _ in self.objects]) * 360/self.rotation_steps

                    for angles in systematic_angles:

                        threads.append([deepcopy(obj) for obj in self.objects])
                        thread = threads[-1]
                        # generating the thread we are going to modify and setting the thread pointer

                        constrained_indexes.append(ids)
                        # Save indexes to be constrained later in the optimization step

                        for i, vec_pair in enumerate(vecs):
                        # setting molecular positions and rotations (embedding)
                        # i is the molecule index, vecs is a tuple of start and end positions
                        # for the pivot vector

                            start, end = vec_pair
                            angle = angles[i]

                            alignment_rotation = rotation_matrix_from_vectors(pivots[i], end-start)
                            # OK ABOVE

                            # step_rotation = rot_mat_from_pointer(alignment_rotation @ pivots[i], angle)
                            step_rotation = rot_mat_from_pointer(end-start, angle)
                            # center_of_rotation = np.mean(vec_pair, axis=0)
                            center_of_rotation = alignment_rotation @ pivot_means[i]

                            thread[i].rotation = step_rotation @ alignment_rotation

                            # pos = center_of_rotation - thread[i].rotation @ pivot_means[i]
                            # thread[i].position = step_rotation @ (pos + center_of_rotation) - center_of_rotation
                            pos = np.mean(vec_pair, axis=0) - alignment_rotation @ pivot_means[i]
                            thread[i].position = center_of_rotation - step_rotation @ center_of_rotation + pos

                else:
                    log_print('# TO DO: Rejected embed: not matching imposed criterion')

        loadbar(1, 1, prefix=f'Embedding structures ')

        self.dispositions_constrained_indexes = np.array(constrained_indexes)

        return threads

######################################################################################################### RUN

    def run(self, debug=False):
        '''
        '''

        assert self.candidates < 1e9
        # TO DO: perform some action if this number is crazy high

        print()

        head = ' '.join([f'{mol.name}/{mol.reactive_indexes}' for mol in self.objects])
        log.write('TSCoDe - input structures/indexes were: ' + head)
        log.write('\n')
        log.write(f'rotation_steps was {self.rotation_steps}: {round(360/self.rotation_steps, 2)} degrees turns performed\n')

        t_start_run = time.time()

        if self.embed == 'cyclical':
            embedded_structures = self.cyclical_embed()
        else:
            embedded_structures = self.string_embed()


        atomnos = np.concatenate([molecule.atomnos for molecule in objects])
        # cumulative list of atomic numbers associated with coordinates

        try:
            os.remove('TS_out.xyz')
        except:
            pass

        # Calculating new coordinates for embedded_structures and storing them in self.structures

        conf_number = [len(molecule.atomcoords) for molecule in objects]
        conf_indexes = cartesian_product(*[np.array(range(i)) for i in conf_number])
        # first index of each vector is the conformer number of the first molecule and so on

        self.structures = np.zeros((int(len(conf_indexes)*int(len(embedded_structures))), len(atomnos), 3)) # like atomcoords property, but containing multimolecular arrangements
        n_of_constraints = self.dispositions_constrained_indexes.shape[1]
        self.constrained_indexes = np.zeros((int(len(conf_indexes)*int(len(embedded_structures))), n_of_constraints, 2))
        # we will be getting constrained indexes for each combination of conformations from the general self.disposition_constrained_indexes array

        for geometry_number, geometry in enumerate(embedded_structures):

            for molecule in geometry:
                calc_positioned_conformers(molecule)

            for i, conf_index in enumerate(conf_indexes): # 0, [0,0,0] then 1, [0,0,1] then 2, [0,1,1]
                count_atoms = 0

                for molecule_number, conformation in enumerate(conf_index): # (0,0) then (1,0) then (2,0) (first [] of outer for-loop)
                    coords = geometry[molecule_number].positioned_conformers[conformation]
                    n = len(geometry[molecule_number].atomnos)
                    self.structures[geometry_number*len(conf_indexes)+i][count_atoms:count_atoms+n] = coords
                    self.constrained_indexes[geometry_number*len(conf_indexes)+i] = self.dispositions_constrained_indexes[geometry_number]
                    count_atoms += n

        del self.dispositions_constrained_indexes
        # cleaning the old, general data on indexes that ignored conformations

        t_end = time.time()
        log_print(f'Generated {len(self.structures)} transition state candidates ({round(t_end-t_start_run, 2)} s)')

        try:
            ################################################# SANITY CHECK
            
            graphs = [mol.graph for mol in self.objects]

            t_start = time.time()
            mask = np.zeros(len(self.structures), dtype=bool)
            num = len(self.structures)
            ids = [len(mol.atomnos) for mol in self.objects]
            for s, structure in enumerate(self.structures):
                p = True if num > 100 and s % (num // 100) == 0 else False
                if p:
                    loadbar(s, num, prefix=f'Checking structure {s+1}/{num} ')
                # mask[s] = sanity_check(structure, atomnos, self.constrained_indexes[s], graphs, max_new_bonds=3)
                mask[s] = compenetration_check(structure, ids)
            loadbar(1, 1, prefix=f'Checking structure {len(self.structures)}/{len(self.structures)} ')
            self.structures = self.structures[mask]
            self.constrained_indexes = self.constrained_indexes[mask]
            t_end = time.time()

            if np.any(mask == False):
                log_print(f'Discarded {len([b for b in mask if b == False])} candidates for compenetration ({len([b for b in mask if b == True])} left, {round(t_end-t_start, 2)} s)')
            # Performing a sanity check for excessive compenetration on generated structures, discarding the ones that look too bad

            ################################################# PRUNING: SIMILARITY

            if len(self.structures) == 0:
                raise ZeroCandidatesError()

            t_start = time.time()

            before = len(self.structures)
            for k in (4000, 2000, 1000, 500, 200, 100, 50, 10, 5, 2):
                if 4*k < len(self.structures):
                    t_start_int = time.time()
                    self.structures, mask = pre_prune_conformers(self.structures, atomnos, k=k)
                    self.constrained_indexes = self.constrained_indexes[mask]
                    t_end_int = time.time()
                    log.write(f'similarity pre-processing (k={k}) - {round(t_end_int-t_start_int, 2)} s - kept {len([b for b in mask if b == True])}/{len(mask)}\n')
            
            self.structures, mask = prune_conformers(self.structures, atomnos, np.array([0 for _ in range(len(self.structures))]))
            t_end = time.time()
            log.write(f'similarity processing     (k=1) - {round(t_end-t_end_int, 2)} s - kept {len([b for b in mask if b == True])}/{len(mask)}\n')

            self.constrained_indexes = self.constrained_indexes[mask]

            if np.any(mask == False):
                log_print(f'Discarded {before - len(np.where(mask == True)[0])} candidates for similarity ({len([b for b in mask if b == True])} left, {round(t_end-t_start, 2)} s)')

            ################################################# GEOMETRY OPTIMIZATION

            if len(self.structures) == 0:
                raise ZeroCandidatesError()

            self.energies = np.zeros(len(self.structures))
            self.exit_status = np.zeros(len(self.structures), dtype=bool)

            t_start = time.time()
            if self.optimize:
                for i, structure in enumerate(deepcopy(self.structures)):
                    loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')
                    try:
                        t_start_opt = time.time()
                        intermediate_geometry, _ = Hookean_optimization(structure, atomnos, self.constrained_indexes[i], graphs, calculator='Mopac', method='PM7')
                        self.structures[i], self.energies[i], self.exit_status[i] = optimize(intermediate_geometry, atomnos, self.constrained_indexes[i], graphs, calculator='Mopac', method='PM7')
                        t_end_opt = time.time()

                        exit_str = 'CONVERGED' if self.exit_status[i] else 'SCRAMBLED'

                    except ValueError:
                        # ase will throw a ValueError if the output lacks a space in the "FINAL POINTS AND DERIVATIVES" table.
                        # This occurs when one or more of them is not defined, that is when the calculation did not end well.
                        # The easiest solution is to reject the structure and go on.
                        self.structures[i] = None
                        self.energies[i] = np.inf
                        self.exit_status[i] = False
                        exit_str = 'FAILED TO READ FILE'

                    finally:
                        log.write(f'Mopac PM7 optimization: Structure {i} {exit_str} - took {round(t_end_opt-t_start_opt, 2)} s\n')

                loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')
                t_end = time.time()
                log_print(f'Mopac PM7 optimization took {round(t_end-t_start, 2)} s ({round((t_end-t_start)/len(self.structures), 2)} s per structure)')

                ################################################# PRUNING: EXIT STATUS

                mask = self.exit_status
                self.structures = self.structures[mask]
                self.energies = self.energies[mask]
                self.constrained_indexes = self.constrained_indexes[mask]

                if np.any(mask == False):
                    log_print(f'Discarded {len(np.where(mask == False)[0])} candidates because optimizations failed or scrambled some atoms ({len([b for b in mask if b == True])} left)')

                ################################################# PRUNING: ENERGY
            
                if len(self.structures) == 0:
                    raise ZeroCandidatesError()

                self.energies = self.energies - np.min(self.energies)
                mask = self.energies < THRESHOLD_KCAL
                self.structures = self.structures[mask]
                self.energies = self.energies[mask]

                if np.any(mask == False):
                    log_print(f'Discarded {len(np.where(mask == False)[0])} candidates for energy (Threshold set to {THRESHOLD_KCAL} kcal/mol)')

                ################################################# PRUNING: SIMILARITY (AGAIN)

                if len(self.structures) == 0:
                    raise ZeroCandidatesError()

                t_start = time.time()
                self.structures, mask = prune_conformers(self.structures, atomnos, self.energies)
                self.energies = self.energies[mask]
                t_end = time.time()
                
                if np.any(mask == False):
                    log_print(f'Discarded {len(np.where(mask == False)[0])} candidates for similarity ({len([b for b in mask if b == True])} left, {round(t_end-t_start, 2)} s)')

        except ZeroCandidatesError:
            s = 'Sorry, the program did not find any reasonable TS structure. Are you sure the input indexes were correct? If so, try enlarging the search space by specifying a larger "steps" value.'
            log_print(s)
            raise ZeroCandidatesError(s)
        ################################################# OUTPUT

        outname = 'TS_out.xyz'
        with open(outname, 'w') as f:        
            for i, structure in enumerate(self.structures):
                write_xyz(structure, atomnos, f, title=f'TS candidate {i}')
        t_end_run = time.time()
        log_print(f'Wrote {len(self.structures)} TS structures to {outname} file - Total time {round(t_end_run-t_start_run, 2)} s')
        log.close()


if __name__ == '__main__':

    try:
        os.remove('ouroboros_setup.xyz')
    except:
        pass

    # a = ['Resources/SN2/MeOH_ensemble.xyz', 1]
    # a = ['Resources/dienamine/dienamine_ensemble.xyz', 6]
    # a = ['Resources/SN2/amine_ensemble.xyz', 22]
    # a = ['Resources/indole/indole_ensemble.xyz', 6]
    ######################################################
    b = ['Resources/SN2/CH3Br_ensemble.xyz', 0]
    # b = ['Resources/SN2/ketone_ensemble.xyz', 2]
    # inp = [a,b]

    # c = ['Resources/SN2/MeOH_ensemble.xyz', (1,5)]
    # c = ['Resources/DA/CHCH3.xyz', [0,6]]
    # c = ['Resources/SN2/BrCl.xyz', [0,1]]
    c = ['Resources/NOH.xyz', (0,1)]
    inp = [c,c,c]

    # d = ['Resources/DA/diene.xyz', (2,7)]
    # e = ['Resources/DA/dienophile.xyz', (3,5)]
    # inp = [d,e]
    # inp = [e,e,e]

    # d = ['Resources/DA/diene2.xyz', (0,6)]
    # e = ['Resources/DA/dienophile2.xyz', (3,5)]
    # inp = [d,e]

    a = ['Resources/dienamine/dienamine_ensemble.xyz', (6,23)]
    b = ['Resources/acid_ensemble.xyz', (3,25)]
    c = ['Resources/maleimide.xyz', (0,5)]
    inp = (a,b,c)


    objects = [Hypermolecule(m[0], m[1]) for m in inp]

    docker = Docker(objects) # initialize docker with molecule density objects
    docker.setup(steps=3, optimize=False) # set variables

    os.chdir('Resources/SN2')

    docker.run(debug=True)

    path = os.path.join(os.getcwd(), 'TS_out.vmd')
    # check_call(f'vmd -e {path}'.split(), stdout=DEVNULL, stderr=STDOUT)
    print('Opening VMD...')
    with suppress_stdout_stderr():
        os.system(f'vmd -e {path}')

    print()