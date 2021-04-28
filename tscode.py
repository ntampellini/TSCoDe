'''
TSCODE: Transition State Conformational Docker
Version 0.00 - Pre-release
Nicolo' Tampellini - nicolo.tampellini@yale.edu
https://github.com/ntampellini/TSCoDe
(Work in Progress)

'''
from hypermolecule_class import Hypermolecule, pt
import numpy as np
from copy import deepcopy
from parameters import *
import os
import time
from optimization_methods import *
from subprocess import DEVNULL, STDOUT, check_call
from linalg_tools import *
from compenetration import compenetration_check
from prune import prune_conformers
import re
from dataclasses import dataclass

class ZeroCandidatesError(Exception):
    pass

class InputError(Exception):
    pass

stamp = time.ctime().replace(' ','_').replace(':','-')
log = open(f'TSCoDe_log_{stamp}.txt', 'a', buffering=1)



def log_print(string='', p=True):
    if p:
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

@dataclass
class Options:

    __keywords__ = [
                    'SUPRAFAC',   # Only retain suprafacial configurations in cyclical TSs.
                                  # Thought for Diels-Alder reactions.

                    'DEEP',       # Performs a deeper search, retaining more starting points
                                  # for calculations and smaller turning angles.

                    'NOOPT',      # Skip the optimization steps, directly writing structures to file.

                    'CHECKPOINT', # Writes structures to file before the geometry optimization steps.

                    'STEPS',      # Manually specify the number of steps to be taken in scanning 360°
                                  # rotations. The standard value of 6 will perform six 60° turns.
                                  # Syntax: SCAN=n, where n is an integer.

                    'BYPASS',     # Debug keyword. Used to skip all pruning steps and directly output
                                  # all the embedded geometries.

                    'THRESH',     # RMSD threshold (Angstroms) for structure pruning. The smaller, the more
                                  # retained structures. Default is 0.5 A. Syntax: THRESH=n, where n is a number.

                    'DIST',       # Manually imposed distance between specified atom pairs, in Angstroms.
                                  # Syntax uses parenthesis and commas: DIST(a=2.345,b=3.67,c=2.1)

                    'CLASHES',    # Manually specify the max number of clashes and the distance threshold
                                  # at which two atoms are considered clashing. The more forgiving, the more
                                  # structures will reach the geometry optimization step. Syntax:
                                  # CLASHES(num=3,dist=1.2)
                    
                    'NEWBONDS',   # Manually specify the maximum numeber of "new bonds" that a TS structure
                                  # can have to be retained and not to be considered scrambled. Default is 2.
                                  # Syntax: NEWBONDS=2

                    'NOTS',       # Do not perform TS search but just the partial optimization step

                    'LEVEL',      # Manually set the MOPAC theory level to be used, default is PM7.
                                  # Syntax: LEVEL=PM7
                    ]
                    
    # list of keyword names to be used in the first line of program input

    rotation_steps = 6
    pruning_thresh = 0.5
    
    max_clashes = 3
    clash_thresh = 1.2

    max_newbonds = 2

    optimization = True
    TS_optimization = True
    mopac_level = 'PM7'
    suprafacial = False
    checkpoint = False

    bypass = False
    # Default values, updated if _parse_input
    # finds keywords and calls _set_options

    def __repr__(self):
        d = {var:self.__getattribute__(var) for var in dir(self) if var[0:2] != '__'}
        d.pop('bypass')
        return '\n'.join([f'{var} : {d[var]}' for var in d])

class Docker:
    def __init__(self, filename):
        '''
        Initialize the Docker object by reading the input filename (.txt).
        Sets the Option dataclass properties to default and then updates them
        with the user-requested keywords, if there are any.

        '''
        s =  '*************************************************************\n'
        s += '*      TSCODE: Transition State Conformational Docker       *\n'
        s += '*************************************************************\n'
        s += '*                 Version 0.00 - Pre-release                *\n'
        s += "*       Nicolo' Tampellini - nicolo.tampellini@yale.edu     *\n"
        s += '*************************************************************\n'

        log_print(s)

        self.options = Options()
        self.objects = [Hypermolecule(name, c_ids) for name, c_ids in self._parse_input(filename)]
        self.ids = [len(mol.atomnos) for mol in self.objects]
        self._read_pairings(filename)
        self._set_options(filename)

    def _parse_input(self, filename):
        '''
        Reads a textfile and sets the Docker properties for the run.
        Keywords are read from the first non-comment(#), non-blank line
        if there are any, and molecules are read afterward.

        '''

        with open(filename, 'r') as f:
            lines = f.readlines()

        lines = [line for line in lines if line[0] != '#']
        lines = [line for line in lines if line != '\n']
        
        try:
            assert len(lines) in (2,3,4)
            # (optional) keyword line + 2 or 3 lines for molecules

            keywords = [l.split('=')[0] if not '(' in l else l.split('(')[0] for l in lines[0].split()]
            if any(k in self.options.__keywords__ for k in keywords):
                lines = lines[1:]

            inp = []
            for line in lines:
                filename, *reactive_atoms = line.split()
                reactive_indexes = tuple([int(re.sub('[^0-9]', '', i)) for i in reactive_atoms])
                inp.append((filename, reactive_indexes))

            return inp
            
        except Exception as e:
            print(e)
            raise InputError(f'Error in reading molecule input for {filename}. Please check your syntax.')

    def _set_options(self, filename):
        '''
        Set the options dataclass parameters from a list of given keywords. These will be used
        during the run to vary the search depth and/or output.
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()

        lines = [line for line in lines if line[0] != '#']
        lines = [line for line in lines if line != '\n']
        
        try:
            keywords = [l.split('=')[0] if not '(' in l else l.split('(')[0] for l in lines[0].split()]
            if any(k in self.options.__keywords__ for k in keywords):

                if not all(k in self.options.__keywords__ for k in keywords):
                    raise SyntaxError(f'One (or more) keywords were not understood. Please check your syntax. ({keywords})')

                keywords_list = lines[0].split()

                if 'SUPRAFAC' in keywords_list:
                    self.options.suprafacial = True

                if 'DEEP' in keywords_list:
                    self.options.pruning_thresh = 0.2
                    self.options.rotation_steps = 12
                    self.options.max_clashes = 5
                    self.options.clash_thresh = 1

                if 'STEPS' in [k.split('=')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('STEPS')]
                    self.options.rotation_steps = int(kw.split('=')[1])

                if 'THRESH' in [k.split('=')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('THRESH')]
                    self.options.pruning_thresh = float(kw.split('=')[1])

                if 'NOOPT' in keywords_list:
                    self.options.optimization = False
                    
                if 'CHECKPOINT' in keywords_list:
                    self.options.checkpoint = True

                if 'BYPASS' in keywords_list:
                    self.options.bypass = True

                if 'DIST' in [k.split('(')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('(')[0] for k in keywords_list].index('DIST')]
                    orb_string = kw[5:-1]
                    # orb_string looks like 'a=2.345,b=3.456,c=2.22'

                    self._set_custom_orbs(orb_string)

                if 'CLASHES' in [k.split('(')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('(')[0] for k in keywords_list].index('CLASHES')]
                    clashes_string = kw[8:-1]
                    # clashes_string looks like 'num=3,dist=1.2'

                    for piece in clashes_string.split(','):
                        s = piece.split('=')
                        if s[0] == 'num':
                            self.options.max_clashes = int(s[1])
                        elif s[0] == 'dist':
                            self.options.clash_thresh = float(s[1])
                        else:
                            raise SyntaxError((f'Syntax error in CLASHES keyword -> CLASHES({clashes_string}).' +
                                                'Correct syntax looks like: CLASHES(num=3,dist=1.2)'))
                
                if 'NEWBONDS' in [k.split('=')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('NEWBONDS')]
                    self.options.max_newbonds = int(kw.split('=')[1])

                if 'NOTS' in keywords_list:
                    self.options.TS_optimization = False

                if 'LEVEL' in [k.split('=')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('LEVEL')]
                    self.options.mopac_level = kw.split('=')[1]


        except Exception as e:
            print(e)
            raise InputError(f'Error in reading keywords from {filename}. Please check your syntax.')

    def _read_pairings(self, filename):
        '''
        Reads atomic pairings to be respected from the input file, if any are present.

        '''
        with open(filename, 'r') as f:
            lines = f.readlines()

        lines = [line for line in lines if line[0] != '#']
        lines = [line for line in lines if line != '\n']
        # discard comments and blank lines
        

        keywords = [l.split('=')[0] if not '(' in l else l.split('(')[0] for l in lines[0].split()]
        if any(k in self.options.__keywords__ for k in keywords):
            lines = lines[1:]
        # if we have a keyword line, discard it        

        parsed = []
        self.pairings_dict = {i:[] for i in range(len(self.objects))}

        for i, line in enumerate(lines):
        # i is also molecule index in self.objects

            pairings = line.split()[1:]
            # remove the molecule name, keep pairs only ['2a','5b']

            pairings = [[int(j[:-1]), j[-1]] for j in pairings if j.lower().islower()]

            for pair in pairings:
                self.pairings_dict[i].append(pair[:])
            # appending pairing to dict before
            # calculating its cumulative index

            if i > 0:
                for z in pairings:
                    z[0] += sum(self.ids[:i])
            # getting the cumulative index rather than the molecule index

            for cumulative_pair in pairings:
                parsed.append(cumulative_pair)
        # parsed looks like [[1, 'a'], [9, 'a']] where numbers are
        # cumulative indexes for TSs

        links = {j:[] for j in set([i[1] for i in parsed])}
        for index, tag in parsed:
            links[tag].append(index)
        # storing couples into a dictionary

        pairings = list(links.values())

        self.pairings = [sorted(i) for i in pairings]

        if self.pairings == []:
            s = '--> No atom pairing imposed. Computing all possible dispositions.'
        else:
            pairings = [line.split()]
            s = f'--> Atom pairings imposed are {len(self.pairings)}: {self.pairings} (Cumulative index numbering)\n'
        log_print(s)

    def _set_custom_orbs(self, orb_string):
        '''
        Update the reactive_atoms classes with the user-specified orbital distances.
        :param orb_string: string that looks like 'a=2.345,b=3.456,c=2.22'

        '''
        pairs = [(piece.split('=')[0], float(piece.split('=')[1])) for piece in orb_string.split(',')]
        # TODO: debug from here

        for letter, dist in pairs:
            for index in range(len(self.objects)):
                for pairing in self.pairings_dict[index]:

        # for each pairing specified by the user, check each pairing recorded
        # in the pairing_dict on that molecule.

                    if pairing[1] == letter:
                        for reactive_atom, reactive_index in zip(self.objects[index].reactive_atoms_classes, self.objects[index].reactive_indexes):
                            if reactive_index == pairing[0]:
                                reactive_atom.init(self.objects[index], reactive_index, update=True, orb_dim=dist/2)
                                self.objects[index]._update_orbs()
                                log_print(f'--> Custom distance read: modified orbital of {index+1}. {self.objects[index].name} atom {reactive_index} to {round(dist/2, 3)} A.')
                    # If the letter matches, look for the correct reactive atom on that molecule. When we find the correct match,
                    # set the new orbital center with imposed distance from the reactive atom. The imposed distance is half the 
                    # user-specified one, as the final atomic distances will be given by two halves of this length.
            log_print()

    def _set_pivots(self, mol):
        '''
        params mol: Hypermolecule class
        (Cyclical embed) Function that sets the mol.pivots attribute, that is a list
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

        mol.pivots = np.array(mol.pivots)
        mol.pivot_means = np.array(mol.pivot_means)

        if self.options.suprafacial:
            if len(mol.pivots) == 1:
            # reactive atoms have one center each, pass
                pass

            elif len(mol.pivots) == 2:
            # reactive atoms have one and two centers,
            # respectively. TODO: Apply bridging carboxylic acid correction?
                pass

            elif len(mol.pivots) == 4:
            # reactive atoms have two centers each.
            # Applying suprafacial correction, only keeping
            # the shorter two, as they SHOULD be the suprafacial ones
                norms = np.linalg.norm(mol.pivots, axis=1)
                for sample in norms:
                    to_keep = [i for i in norms if sample >= i]
                    if len(to_keep) == 2:
                        mask = np.array([i in to_keep for i in norms])
                        mol.pivots = mol.pivots[mask]
                        mol.pivot_means = mol.pivot_means[mask]
                        break

    def _setup(self):
        '''
        :params steps: int, number of steps for the sampling of rotations. A value of 3 corresponds
                        to three 120° turns for each molecule.
        :params optimize: bool, whether to run the semiempirical optimization calculations
        '''

        if len(self.objects) in (2,3):
        # Calculating the number of conformation combinations based on embed type
            if all([len(molecule.reactive_atoms_classes) == 2 for molecule in self.objects]):

                for molecule in self.objects:
                    self._set_pivots(molecule)

                self.embed = 'cyclical'
                self.candidates = self.options.rotation_steps**len(self.objects)*np.prod([len(mol.atomcoords) for mol in self.objects])
                self.candidates *= np.prod([len(mol.pivots) for mol in self.objects])
                if len(self.objects) == 3:
                    self.candidates *= 8
                else:
                    self.candidates *= 2
                # The number 8 is the number of different triangles originated from three oriented vectors,
                # while 2 is the disposition of two vectors (parallel, antiparallel). This ends here if
                # no parings are to be respected. If there are any, each one reduces the number of
                # candidates to be computed, and we divide self.candidates number in the next section.

                if self.pairings != []:
                    if len(self.objects) == 2:
                        n = 2
                    else:
                        if len(self.pairings) == 1:
                            n = 4
                        else:
                            n = 8
                    self.candidates /= n
                # if the user specified some pairings to be respected, we have less candidates to check
                self.candidates = int(self.candidates)

            elif all([len(molecule.reactive_atoms_classes) == 1 for molecule in self.objects]) and len(self.objects) == 2:
                self.embed = 'string'
                self.candidates = self.options.rotation_steps*np.prod([len(mol.atomcoords) for mol in self.objects])*np.prod([len(mol.centers) for mol in self.objects])

            else:
                raise InputError('Bad input - The only molecular configurations accepted are: 1) two or three molecules with two reactive centers each or 2) two molecules with one reactive center each.')
        else:
            raise InputError('Bad input - too many molecules specified (3 max).')

        log_print(f'--> Setup performed correctly. {self.candidates} candidates will be generated.\n')

    def get_string_constrained_indexes(self, n):
        '''
        Get constrained indexes referring to the transition states, repeated n times.
        :params n: int
        :return: list of lists consisting in atomic pairs to be constrained.
        '''
        # Two molecules, string algorithm, one constraint for all, repeated n times
        return np.array([[[int(self.objects[0].reactive_indexes[0]),
                           int(self.objects[1].reactive_indexes[0] + self.ids[0])]] for _ in range(n)])

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
            couples = [sorted(c) for c in couples]
            return couples

    def string_embed(self):
        '''
        return threads: return embedded structures, with position and rotation attributes set, ready to be pumped
        into self.structures. Algorithm used is the "string" algorithm (see docs).
        '''
        assert len(self.objects) == 2
        # NOTE THAT THIS APPROACH WILL ONLY WORK FOR TWO MOLECULES, AND A REVISION MUST BE DONE TO GENERALIZE IT (BUT WOULD IT MAKE SENSE?)

        log_print(f'\n--> Initializing string embed ({self.candidates} candidates)')

        centers_indexes = cartesian_product(*[np.array(range(len(molecule.centers))) for molecule in self.objects])
        # for two mols with 3 and 2 centers: [[0 0][0 1][1 0][1 1][2 0][2 1]]
        
        threads = []
        for _ in range(len(centers_indexes)*self.options.rotation_steps):
            threads.append([deepcopy(obj) for obj in self.objects])

        for t, thread in enumerate(threads): # each run is a different "regioisomer", repeated self.options.rotation_steps times

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
                    rotation = t % self.options.rotation_steps * 360/self.options.rotation_steps       # sistematic incremental step angle
                    
                    delta_rot = rot_mat_from_pointer(pointer, rotation)
                    molecule.rotation = delta_rot @ molecule.rotation

                    molecule.position = thread[i].rotation @ ref_orb_vec + thread[i].position - molecule.rotation @ mol_orb_vec

        self.dispositions_constrained_indexes = self.get_string_constrained_indexes(len(threads))

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
        # this was actually ready for n-gons implementation, but I think we will never use it with n>3
        
        def _get_directions(norms):
            '''
            Returns two or three vectors specifying the direction in which each molecule should be aligned
            in the cyclical TS, pointing towards the center of the polygon.
            '''
            assert len(norms) in (2,3)
            if len(norms) == 2:
                return np.array([[0,1,0],
                                 [0,-1,0]])
            else:

                norms = sorted(norms)
                vertexes = np.zeros((3,2))

                vertexes[1] = np.array([norms[0],0])

                a = np.power(norms[0], 2)
                b = np.power(norms[1], 2)
                c = np.power(norms[2], 2)
                x = (a-b+c)/(2*a**0.5)
                y = (c-x**2)**0.5

                vertexes[2] = np.array([x,y])
                # similar to the code from polygonize, to get the active triangle
                # but without the orientation specified in the polygonize function

                a = vertexes[1,0] # first point, x
                b = vertexes[2,0] # second point, x
                c = vertexes[2,1] # second point, y

                x = a/2
                y = (b**2 + c**2 - a*b)/(2*c)
                cc = np.array([x,y,0])
                # coordinates of the triangle circocenter

                v1 = cc - np.mean((vecs[0,1],vecs[2,1]), axis=0)
                v2 = cc - np.mean((vecs[1,1],vecs[0,1]), axis=0)
                v3 = cc - np.mean((vecs[2,1],vecs[1,1]), axis=0)
                # versors connecting center of side with circocenter

                return np.vstack((v1,v2,v3))

        log_print(f'\n--> Initializing cyclical embed ({self.candidates} candidates)')

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

                directions = _get_directions(norms)
                # directions to orient the molecules toward, orthogonal to each vec_pair

                if self.pairings == [] or all([pair in ids for pair in self.pairings]):

                    systematic_angles = cartesian_product(*[range(self.options.rotation_steps) for _ in self.objects]) * 360/self.options.rotation_steps

                    # angle_range = 150
                    # Angular range to be explored in both directions (degrees). A value of 90 would explore from -90 to +90, always in 2*self.options.rotation_steps+1 steps
                    
                    # systematic_angles = cartesian_product(*[range(-self.options.rotation_steps, self.options.rotation_steps+1) for _ in range(len(self.objects))])*angle_range/self.options.rotation_steps

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

                            reactive_coords = self.objects[i].atomcoords[0][self.objects[i].reactive_indexes]
                            # coordinates for the reactive atoms in this run

                            atomic_pivot_mean = np.mean(reactive_coords, axis=0)
                            # mean position of the atoms active in this run 

                            mol_direction = pivot_means[i]-atomic_pivot_mean
                            if np.all(mol_direction == 0.):
                                mol_direction = pivot_means[i]
                                # log.write(f'mol {i} - improper pivot? Thread {len(threads)-1}\n')

                            # Direction in which the molecule should be oriented, based on the meand of reactive
                            # atom positions and the mean point of the active pivot for the run.
                            # If this vector is too small and gets rounded to zero (as it can happen for
                            # "antrafacial" vectors), we fallback to the vector starting from the molecule
                            # center (mean of atomic positions) and ending in pivot_means[i], so to avoid
                            # numeric errors in the next function.
                             
                            alignment_rotation = R.align_vectors((end-start, directions[i]),
                                                                 (pivots[i], mol_direction))[0].as_matrix()
                            # this rotation superimposes the molecular orbitals active in this run (pivots[i] goes to
                            # end-start) and also aligns the molecules so that they face each other (mol_direction
                            # goes to directions[i])
                            
                            axis_of_step_rotation = alignment_rotation @ (reactive_coords[0]-reactive_coords[1])
                            step_rotation = rot_mat_from_pointer(axis_of_step_rotation, angle)
                            # this rotation cycles through all different rotation angles for each molecule

                            center_of_rotation = alignment_rotation @ atomic_pivot_mean
                            # center_of_rotation is the mean point between the reactive atoms so
                            # as to keep the reactive distances constant

                            thread[i].rotation = step_rotation @ alignment_rotation# @ pre_alignment_rot
                            # overall rotation for the molecule is given by the matrices product

                            # pos = np.mean(vec_pair, axis=0) - alignment_rotation @ pre_alignment_rot @ pivot_means[i]
                            pos = np.mean(vec_pair, axis=0) - alignment_rotation @ pivot_means[i]
                            thread[i].position += center_of_rotation - step_rotation @ center_of_rotation + pos
                            # overall position is given by superimposing mean of active pivot (connecting orbitals)
                            # to mean of vec_pair (defining the target position - the side of a triangle for three molecules)


        loadbar(1, 1, prefix=f'Embedding structures ')

        self.dispositions_constrained_indexes = np.array(constrained_indexes)

        return threads

######################################################################################################### RUN

    def run(self, debug=False):
        '''
        '''
        try:
            self._setup()

            assert self.candidates < 1e8, ('ATTENTION! This calculation is probably going to be very big. To ignore this message' +
                                           ' and proceed, run this python file with the -O flag. Ex: python -O tscode.py input.txt')

            head = ''
            for i, mol in enumerate(self.objects):
                s = [atom.symbol+'('+str(atom)+')' for atom in mol.reactive_atoms_classes]
                t = ', '.join([f'{a}->{b}' for a,b in zip(mol.reactive_indexes, s)])
                head += f'    {i+1}. {mol.name}: {t}\n'

            log_print('--> Input structures, reactive indexes and reactive atoms TSCODE type:\n' + head)
            log_print(f'--> Calculation options used were:')
            for line in str(self.options).split('\n'):
                log_print(f'    - {line}')

            t_start_run = time.time()

            if self.embed == 'cyclical':
                embedded_structures = self.cyclical_embed()
            else:
                embedded_structures = self.string_embed()


            atomnos = np.concatenate([molecule.atomnos for molecule in self.objects])
            # cumulative list of atomic numbers associated with coordinates

            try:
                os.remove('TS_out.xyz')
            except:
                pass
            # TODO: rename output through stamp variable


            conf_number = [len(molecule.atomcoords) for molecule in self.objects]
            conf_indexes = cartesian_product(*[np.array(range(i)) for i in conf_number])
            # first index of each vector is the conformer number of the first molecule and so on

            self.structures = np.zeros((int(len(conf_indexes)*int(len(embedded_structures))), len(atomnos), 3)) # like atomcoords property, but containing multimolecular arrangements
            n_of_constraints = self.dispositions_constrained_indexes.shape[1]
            self.constrained_indexes = np.zeros((int(len(conf_indexes)*int(len(embedded_structures))), n_of_constraints, 2), dtype=int)
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
            # Calculating new coordinates for embedded_structures and storing them in self.structures

            del self.dispositions_constrained_indexes
            # cleaning the old, general data on indexes that ignored conformations

            t_end = time.time()
            log_print(f'Generated {len(self.structures)} transition state candidates ({round(t_end-t_start_run, 2)} s)\n')

            if not self.options.bypass:
                try:
                    ################################################# COMPENETRATION CHECK
                    
                    log_print('--> Checking structures for compenetrations')

                    graphs = [mol.graph for mol in self.objects]

                    t_start = time.time()
                    mask = np.zeros(len(self.structures), dtype=bool)
                    num = len(self.structures)
                    for s, structure in enumerate(self.structures):
                        p = True if num > 100 and s % (num // 100) == 0 else False
                        if p:
                            loadbar(s, num, prefix=f'Checking structure {s+1}/{num} ')
                        mask[s] = compenetration_check(structure, self.ids, max_clashes=self.options.max_clashes, thresh=self.options.clash_thresh)

                    loadbar(1, 1, prefix=f'Checking structure {len(self.structures)}/{len(self.structures)} ')
                    self.structures = self.structures[mask]
                    self.constrained_indexes = self.constrained_indexes[mask]
                    t_end = time.time()

                    if np.any(mask == False):
                        log_print(f'Discarded {len([b for b in mask if b == False])} candidates for compenetration ({len([b for b in mask if b == True])} left, {round(t_end-t_start, 2)} s)')
                    log_print()
                    # Performing a sanity check for excessive compenetration on generated structures, discarding the ones that look too bad

                    ################################################# PRUNING: SIMILARITY

                    if len(self.structures) == 0:
                        raise ZeroCandidatesError()

                    log_print('--> Similarity Processing')
                    t_start = time.time()

                    before = len(self.structures)
                    for k in (5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2):
                        if 5*k < len(self.structures):
                            t_start_int = time.time()
                            self.structures, mask = prune_conformers(self.structures, atomnos, max_rmsd=self.options.pruning_thresh, k=k)
                            self.constrained_indexes = self.constrained_indexes[mask]
                            t_end_int = time.time()
                            log_print(f'    - similarity pre-processing   (k={k}) - {round(t_end_int-t_start_int, 2)} s - kept {len([b for b in mask if b == True])}/{len(mask)}')
                    
                    t_start_int = time.time()
                    self.structures, mask = prune_conformers(self.structures, atomnos, max_rmsd=self.options.pruning_thresh)
                    t_end = time.time()
                    log_print(f'    - similarity final processing (k=1) - {round(t_end-t_start_int, 2)} s - kept {len([b for b in mask if b == True])}/{len(mask)}')

                    self.constrained_indexes = self.constrained_indexes[mask]

                    if np.any(mask == False):
                        log_print(f'Discarded {before - len(np.where(mask == True)[0])} candidates for similarity ({len([b for b in mask if b == True])} left, {round(t_end-t_start, 2)} s)')
                    log_print()

                    ################################################# CHECKPOINT SAVE BEFORE OPTIMIZATION

                    if self.options.checkpoint:
                            outname = f'TSCoDe_checkpoint_{stamp}.xyz'
                            with open(outname, 'w') as f:        
                                for i, structure in enumerate(self.structures):
                                    write_xyz(structure, atomnos, f, title=f'TS candidate {i} - Checkpoint before optimization')
                            t_end_run = time.time()
                            log_print(f'--> Checkpoint requested - Wrote {len(self.structures)} TS structures to {outname} file before optimizaiton.\n')

                    ################################################# GEOMETRY OPTIMIZATION


                    if len(self.structures) == 0:
                        raise ZeroCandidatesError()

                    self.energies = np.zeros(len(self.structures))
                    self.exit_status = np.zeros(len(self.structures), dtype=bool)

                    t_start = time.time()
                    if self.options.optimization:

                        log_print(f'--> Structure optimization ({self.options.mopac_level} level)')

                        for i, structure in enumerate(deepcopy(self.structures)):
                            loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')
                            try:
                                t_start_opt = time.time()
                                self.structures[i], self.energies[i], self.exit_status[i] = optimize(structure,
                                                                                                     atomnos,
                                                                                                     graphs,
                                                                                                     self.constrained_indexes[i],
                                                                                                     method=f'{self.options.mopac_level} GEO-OK',
                                                                                                     max_newbonds=self.options.max_newbonds)


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
                                t_end_opt = time.time()
                                log_print(f'    - Mopac {self.options.mopac_level} optimization: Structure {i} {exit_str} - took {round(t_end_opt-t_start_opt, 2)} s', p=False)

                        loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')
                        t_end = time.time()
                        log_print(f'Mopac {self.options.mopac_level} optimization took {round(t_end-t_start, 2)} s ({round((t_end-t_start)/len(self.structures), 2)} s per structure)')

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

                        mask = (self.energies - np.min(self.energies)) < THRESHOLD_KCAL
                        self.structures = self.structures[mask]
                        self.energies = self.energies[mask]

                        _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
                        from optimization_methods import scramble
                        self.energies = scramble(self.energies, sequence)
                        self.structures = scramble(self.structures, sequence)
                        self.constrained_indexes = scramble(self.constrained_indexes, sequence)
                        # sorting structures based on energy

                        if np.any(mask == False):
                            log_print(f'Discarded {len(np.where(mask == False)[0])} candidates for energy (Threshold set to {THRESHOLD_KCAL} kcal/mol)')

                        ################################################# PRUNING: SIMILARITY (POST PARTIAL OPT)

                        if len(self.structures) == 0:
                            raise ZeroCandidatesError()

                        t_start = time.time()
                        self.structures, mask = prune_conformers(self.structures, atomnos, max_rmsd=self.options.pruning_thresh)
                        self.energies = self.energies[mask]
                        t_end = time.time()
                        
                        if np.any(mask == False):
                            log_print(f'Discarded {len(np.where(mask == False)[0])} candidates for similarity ({len([b for b in mask if b == True])} left, {round(t_end-t_start, 2)} s)')
                        log_print()

                        ################################################# TS SEEKING
                        
                        if self.options.TS_optimization:

                            log_print(f'--> Transition state optimization ({self.options.mopac_level} level)')

                            self.exit_status = np.zeros(len(self.structures), dtype=bool)

                            for i, structure in enumerate(deepcopy(self.structures)):
                                loadbar(i, len(self.structures), prefix=f'Optimizing TS {i+1}/{len(self.structures)} ')
                                try:
                                    t_start_opt = time.time()
                                    new_structure, new_energy, self.exit_status[i] = optimize(structure,
                                                                                              atomnos,
                                                                                              graphs,
                                                                                              method=f'{self.options.mopac_level} TS',
                                                                                              max_newbonds=self.options.max_newbonds)

                                    if self.exit_status[i]:
                                        self.structures[i], self.energies[i] = new_structure, new_energy
                                        exit_str = 'TS FOUND'

                                    else:
                                        exit_str = 'TS FAILED'


                                except ValueError:
                                    # ase will throw a ValueError if the output lacks a space in the "FINAL POINTS AND DERIVATIVES" table.
                                    # This occurs when one or more of them is not defined, that is when the calculation did not end well.
                                    # The easiest solution is to reject the structure and go on.
                                    self.structures[i] = None
                                    self.energies[i] = np.inf
                                    self.exit_status[i] = False
                                    exit_str = 'FAILED TO READ FILE'

                                finally:
                                    t_end_opt = time.time()
                                    log_print(f'    - Mopac {self.options.mopac_level} TS optimization: Structure {i} {exit_str} - took {round(t_end_opt-t_start_opt, 2)} s', p=False)

                            loadbar(1, 1, prefix=f'Optimizing TS  {len(self.structures)}/{len(self.structures)} ')
                            t_end = time.time()
                            log_print(f'Mopac {self.options.mopac_level} TS optimization took {round(t_end-t_start, 2)} s ({round((t_end-t_start)/len(self.structures), 2)} s per structure)')

                            ################################################# DETAILING: TS EXIT STATUS

                            _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
                            self.energies = scramble(self.energies, sequence)
                            self.structures = scramble(self.structures, sequence)
                            self.constrained_indexes = scramble(self.constrained_indexes, sequence)
                            # sorting structures based on energy

                            mask = self.exit_status
                            if np.any(mask == True):
                                log_print(f'Succeeded in refining {len(np.where(mask == True)[0])}/{len(mask)} candidates to {self.options.mopac_level} transition states.')
                            log_print()





                except ZeroCandidatesError:
                    t_end_run = time.time()
                    s = ('Sorry, the program did not find any reasonable TS structure. Are you sure the input indexes were correct? If so, try these tips:\n' + 
                         '    - Enlarging the search space by specifying a larger "steps" value with the keyword STEPS=n\n' +
                         '    - Imposing less strict rejection criteria with the DEEP or CLASHES keyword.\n')
                    log_print(f'--> Program termination: No candidates found - Total time {round(t_end_run-t_start_run, 2)} s')
                    log_print(s)
                    raise ZeroCandidatesError(s)

            else:
                self.energies = np.zeros(len(self.structures))

            ################################################# XYZ OUTPUT 

            self.energies = self.energies - np.min(self.energies)

            outname = 'TS_out.xyz'
            with open(outname, 'w') as f:        
                for i, structure in enumerate(self.structures):

                    if self.options.TS_optimization:
                        kind = 'TS - ' if self.exit_status[i] else 'NOT TS - '
                    else:
                        kind = ''

                    write_xyz(structure, atomnos, f, title=f'Structure {i} - {kind}Rel. E. = {round(self.energies[i], 3)} kcal/mol')

            t_end_run = time.time()
            log_print(f'--> Output: Wrote {len(self.structures)} TS structures to {outname} file - Total time {round(t_end_run-t_start_run, 2)} s')

            ################################################# SDF OUTPUT 

            sdf_name = outname.split('.')[0] + '.sdf'
            check_call(f'obabel {outname} -O {sdf_name}')

            ################################################# VMD OUTPUT

            path = os.path.join(os.getcwd(), 'TS_out.vmd')
            with open(path, 'w') as f:
                s = ('display resetview\n' +
                    'mol new {%s.xyz}\n' % (path.strip('.vmd')) +
                    'mol selection index %s\n' % (' '.join([str(i) for i in self.constrained_indexes[0].ravel()])) +
                    'mol representation CPK 0.7 0.5 50 50\n' +
                    'mol color ColorID 7\n' +
                    'mol material Transparent\n' +
                    'mol addrep top')
                f.write(s)

            log_print(f'--> Output: Wrote VMD TS_out.vmd file')
            
            ################################################# END
            log.close()

        except KeyboardInterrupt:
            print('KeyboardInterrupt requested by user. Quitting.')
            quit()


if __name__ == '__main__':

    import sys
    if len(sys.argv) != 2:
        filename = 'test_input.txt'
        # print('\n\tTSCODE correct usage:\n\n\tpython tscode.py input.txt\n\n\tSee documentation for input formatting.\n')
        # quit()
        
    # filename = sys.argv[1]

    docker = Docker(filename)
    # initialize docker from input file
    
    os.chdir('Resources/SN2')

    bypass = False
    docker.run(debug=True)

    path = os.path.join(os.getcwd(), 'TS_out.vmd')
    print('Opening VMD...')
    with suppress_stdout_stderr():
        os.system(f'vmd -e {path}')