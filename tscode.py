'''
TSCODE: Transition State Conformational Docker
Version 0.00 - Pre-release
Nicolo' Tampellini - nicolo.tampellini@yale.edu
https://github.com/ntampellini/TSCoDe
(Work in Progress)

'''
from hypermolecule_class import Hypermolecule, align_structures
import numpy as np
from copy import deepcopy
from parameters import *
import os
import time
from optimization_methods import *
from linalg_tools import *
from compenetration import compenetration_check
from prune import prune_conformers
import re
from dataclasses import dataclass
from itertools import groupby

def clean_directory():

    for f in os.listdir():
        if f.startswith('temp'):
            os.remove(f)
        elif f.endswith('.arc'):
            os.remove(f)
        elif f.endswith('.mop'):
            os.remove(f)

class ZeroCandidatesError(Exception):
    pass

class InputError(Exception):
    pass

def loadbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
	percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
	if iteration == total:
		print()

def calc_positioned_conformers(self):
    self.positioned_conformers = np.array([[self.rotation @ v + self.position for v in conformer] for conformer in self.atomcoords])

def ase_view(mol):
    from ase import Atoms
    from ase.gui.gui import GUI
    from ase.gui.images import Images

    centers = np.vstack([atom.center for atom in mol.reactive_atoms_classes_dict.values()])
    atomnos = np.concatenate((mol.atomnos, [0 for _ in centers]))
    images = []
    for coords in mol.atomcoords:

        totalcoords = np.concatenate((coords, centers))
        images.append(Atoms(atomnos, positions=totalcoords))
    
    GUI(images=Images(images), show_bonds=True).run()

class Pivot:
    '''
    (Cyclical embed)
    Pivot object: vector connecting two lobes of a
    molecule, starting from v1 (first reactive atom in
    mol.reacitve_atoms_classes_dict) and ending on v2.

    For molecules involved in chelotropic reactions,
    that is molecules that undergo a cyclical embed
    while having only one reactive atom, pivots are
    built on that single atom.
    '''
    def __init__(self, v1, v2, index1, index2):
        self.start = v1
        self.end = v2
        self.pivot = v2 - v1
        self.meanpoint = np.mean((v1, v2), axis=0)
        self.index = (index1, index2)
        # the pivot starts from the index1-th
        # center of the first reactive atom
        # and to the index2-th center of the second

    def __repr__(self):
        return f'Pivot object - index {self.index}, norm {round(np.linalg.norm(self.pivot), 3)}, meanpoint {self.meanpoint}'

@dataclass
class Options:

    __keywords__ = [
                    'SUPRAFAC',   # Only retain suprafacial configurations in cyclical TSs.
                                  # Thought for Diels-Alder reactions.

                    'DEEP',       # Performs a deeper search, retaining more starting points
                                  # for calculations and smaller turning angles.

                    'NOOPT',      # Skip the optimization steps, directly writing structures to file.

                    'STEPS',      # Manually specify the number of steps to be taken in scanning 360°
                                  # rotations. The standard value of 6 will perform six 60° turns.
                                  # Syntax: SCAN=n, where n is an integer.

                    'ROTRANGE'    # Manually set the rotation range to be explored (only cyclical)

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

                    'NEB',        # Perform a NEB TS search after the partial optimization step

                    'LEVEL',      # Manually set the MOPAC theory level to be used, default is PM7.
                                  # Syntax: LEVEL=PM7

                    'RIGID',      # Do not bend structures to better build TSs

                    'NONCI',      # Avoid estimating and printing non-covalent interactions

                    'ONLYREFINED',# Discard structures that do not successfully refine bonding distances

                    'LET',        # Overrides safety checks for large calculations

                    'CHECK'

                    ]
                    
    # list of keyword names to be used in the first line of program input

    rotation_range = 45
    rotation_steps = None # This is set later by the _setup() function, based on embed type
    pruning_thresh = 0.5 # default value set later on the basis of number of atoms
    rigid = False
    
    max_clashes = 3
    clash_thresh = 1.2

    max_newbonds = 1

    optimization = True
    neb = False
    mopac_level = 'PM7'
    suprafacial = False
    nci = True
    only_refined = False

    bypass = False
    let = False
    check_structures = False
    # Default values, updated if _parse_input
    # finds keywords and calls _set_options

    def __repr__(self):
        d = {var:self.__getattribute__(var) for var in dir(self) if var[0:2] != '__'}
        d.pop('bypass')
        d.pop('let')
        d.pop('check_structures')
        return '\n'.join([f'{var} : {d[var]}' for var in d])

class Docker:
    def __init__(self, filename):
        '''
        Initialize the Docker object by reading the input filename (.txt).
        Sets the Option dataclass properties to default and then updates them
        with the user-requested keywords, if there are any.

        '''

        self.stamp = time.ctime().replace(' ','_').replace(':','-')[4:-8]
        # replaced ctime yields 'Sun_May_23_18-53-47_2021', only keeping 'May_23_18-53'

        self.logfile = open(f'TSCoDe_{self.stamp}.log', 'a', buffering=1)


        s ='\n*************************************************************\n'
        s += '*      TSCODE: Transition State Conformational Docker       *\n'
        s += '*************************************************************\n'
        s += '*                 Version 0.00 - Test pre-release           *\n'
        s += "*       Nicolo' Tampellini - nicolo.tampellini@yale.edu     *\n"
        s += '*************************************************************\n'

        self.log(s)

        self.options = Options()
        self.objects = [Hypermolecule(name, c_ids) for name, c_ids in self._parse_input(filename)]

        self.ids = [len(mol.atomnos) for mol in self.objects]
        # used to divide molecules in TSs

        for i, mol in enumerate(self.objects):
            for r_atom in mol.reactive_atoms_classes_dict.values():
                if i == 0:
                    r_atom.cumnum = r_atom.index
                else:
                    r_atom.cumnum = r_atom.index + sum(self.ids[:i])
        # appending to each reactive atom the cumulative
        # number indexing in the TS context

        self._read_pairings(filename)
        self._set_options(filename)

        if self.options.check_structures:
            self.log('--> Structures check requested. Shutting down after last window is closed.\n')

            for mol in self.objects:
                ase_view(mol)
            
            self.logfile.close()
            os.remove(f'TSCoDe_{self.stamp}.log')

            quit()

        self._setup()

    def log(self, string='', p=True):
        if p:
            print(string)
        string += '\n'
        self.logfile.write(string)

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

                keywords_list = [word.upper() for word in lines[0].split()]

                if 'SUPRAFAC' in keywords_list:
                    self.options.suprafacial = True

                if 'DEEP' in keywords_list:
                    self.options.pruning_thresh = 0.2
                    self.options.rotation_steps = 12
                    self.options.max_clashes = 5
                    self.options.clash_thresh = 1

                if 'ROTRANGE' in [k.split('=')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('ROTRANGE')]
                    self.options.rotation_range = int(kw.split('=')[1])

                if 'STEPS' in [k.split('=')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('STEPS')]
                    self.options.custom_rotation_steps = int(kw.split('=')[1])

                if 'THRESH' in [k.split('=')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('THRESH')]
                    self.options.pruning_thresh = float(kw.split('=')[1])

                if 'NOOPT' in keywords_list:
                    self.options.optimization = False
                    
                if 'BYPASS' in keywords_list:
                    self.options.bypass = True
                    self.options.optimization = False

                if 'DIST' in [k.split('(')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('(')[0] for k in keywords_list].index('DIST')]
                    orb_string = kw[5:-1].lower().replace(' ','')
                    # orb_string looks like 'a=2.345,b=3.456,c=2.22'

                    self._set_custom_orbs(orb_string)

                if 'CLASHES' in [k.split('(')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('(')[0] for k in keywords_list].index('CLASHES')]
                    clashes_string = kw[8:-1].lower().replace(' ','')
                    # clashes_string looks like 'num=3,dist=1.2'

                    for piece in clashes_string.split(','):
                        s = piece.split('=')
                        if s[0].lower() == 'num':
                            self.options.max_clashes = int(s[1])
                        elif s[0].lower() == 'dist':
                            self.options.clash_thresh = float(s[1])
                        else:
                            raise SyntaxError((f'Syntax error in CLASHES keyword -> CLASHES({clashes_string}).' +
                                                'Correct syntax looks like: CLASHES(num=3,dist=1.2)'))
                
                if 'NEWBONDS' in [k.split('=')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('NEWBONDS')]
                    self.options.max_newbonds = int(kw.split('=')[1])

                if 'NEB' in keywords_list:
                    self.options.neb = True

                if 'LEVEL' in [k.split('=')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('LEVEL')]
                    self.options.mopac_level = kw.split('=')[1].upper()

                if 'RIGID' in keywords_list:
                    self.options.rigid = True

                if 'NONCI' in keywords_list:
                    self.options.nci = False

                if 'ONLYREFINED' in keywords_list:
                    self.options.only_refined = True

                if 'LET' in keywords_list:
                    self.options.let = True

                if 'CHECK' in keywords_list:
                    self.options.check_structures = True

        except SyntaxError as e:
            raise e

        except Exception:
            raise InputError(f'Error in reading keywords from {filename}. Please check your syntax.')

    def _read_pairings(self, filename):
        '''
        Reads atomic pairings to be respected from the input file, if any are present.
        This parsing function is ugly, I know.
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
        unlabeled_list = []
        self.pairings_dict = {i:[] for i in range(len(self.objects))}

        for i, line in enumerate(lines):
        # now i is also the molecule index in self.objects

            fragments = line.split()[1:]
            # remove the molecule name, keep pairs only ['2a','5b']

            unlabeled = []
            pairings = []

            for j in fragments:

                if not j.lower().islower(): # if all we have is a number
                    unlabeled.append(int(j))

                else:
                    index, letters = [''.join(g) for _, g in groupby(j, str.isalpha)]

                    if len(letters) == 1:

                        if letters not in ('a', 'b', 'c'):
                            raise SyntaxError('The only letters allowed for pairings are "a", "b" or "c".')

                        pairings.append([int(index), letters[0]])

                    else:
                        for l in letters:

                            if l not in ('a', 'b', 'c'):
                                raise SyntaxError('The only letters allowed for pairings are "a", "b" or "c".')

                            pairings.append([int(index), l])


            for pair in pairings:
                self.pairings_dict[i].append(pair[:])
            # appending pairing to dict before
            # calculating their cumulative index

            if i > 0:
                for z in pairings:
                    z[0] += sum(self.ids[:i])

                if unlabeled != []:
                    for z in unlabeled:
                        z += sum(self.ids[:i])
                    unlabeled_list.append(z)
            # getting the cumulative index rather than the molecule index

            for cumulative_pair in pairings:
                parsed.append(cumulative_pair)
        # parsed looks like [[1, 'a'], [9, 'a']] where numbers are
        # cumulative indexes for TSs

        links = {j:[] for j in set([i[1] for i in parsed])}
        for index, tag in parsed:
            links[tag].append(index)
        # storing couples into a dictionary

        pairings = list(links.items())
        pairings = sorted(pairings, key=lambda x: x[0])
        # sorting values so that 'a' is the first pairing

        self.pairings_table = {i[0]:sorted(i[1]) for i in pairings}
        # {'a':[3,45]}

        self.pairings = [sorted(i[1]) for i in pairings]
        # getting rid of the letters and sorting the values [34, 3] -> [3, 34]

        for letter, ids in self.pairings_table.items():
            if len(ids) == 1:
                raise SyntaxError(f'Letter \'{letter}\' is only specified once. Please flag the second reactive atom.')
            elif len(ids) > 2:
                raise SyntaxError(f'Letter \'{letter}\' is specified more than two times. Please remove the unwanted letters.')

        if not all([len(mol.reactive_indexes) == 1 for mol in self.objects]): # if not self.embed == 'string', but we that is set afterward by _setup()
            if self.pairings == []:
                s = '--> No atom pairings imposed. Computing all possible dispositions.'
            else:
                s = f'--> Atom pairings imposed are {len(self.pairings)}: {self.pairings} (Cumulative index numbering)\n'
            self.log(s)

        if len(lines) == 3:
        # adding third pairing if we have three molecules and user specified two pairings
        # (used to adjust distances for trimolecular TSs)
            if len(unlabeled_list) == 2:
                third_constraint = list(sorted(unlabeled_list))
                self.pairings.append(third_constraint)
                self.pairings_table['c'] = third_constraint
        
        elif len(lines) == 2:
        # adding second pairing if we have two molecules and user specified one pairing
        # (used to adjust distances for bimolecular TSs)
            if len(unlabeled_list) == 2:
                second_constraint = list(sorted(unlabeled_list))
                self.pairings.append(second_constraint)
                self.pairings_table['b'] = second_constraint

    def _set_custom_orbs(self, orb_string):
        '''
        Update the reactive_atoms classes with the user-specified orbital distances.
        :param orb_string: string that looks like 'a=2.345,b=3.456,c=2.22'

        '''
        self.pairings_dists = [(piece.split('=')[0], float(piece.split('=')[1])) for piece in orb_string.split(',')]
        self.pairings_dists = sorted(self.pairings_dists, key=lambda x: x[0])

        for letter, dist in self.pairings_dists:

            if letter not in self.pairings_table:
                raise SyntaxError(f'Letter \'{letter}\' is specified in DIST but not present in molecules string.')


            for index in range(len(self.objects)):
                for pairing in self.pairings_dict[index]:

        # for each pairing specified by the user, check each pairing recorded
        # in the pairing_dict on that molecule.

                    if pairing[1] == letter:
                        for reactive_index, reactive_atom in self.objects[index].reactive_atoms_classes_dict.items():
                            if reactive_index == pairing[0]:
                                reactive_atom.init(self.objects[index], reactive_index, update=True, orb_dim=dist/2)
                                
                    # If the letter matches, look for the correct reactive atom on that molecule. When we find the correct match,
                    # set the new orbital center with imposed distance from the reactive atom. The imposed distance is half the 
                    # user-specified one, as the final atomic distances will be given by two halves of this length.
            # self.log()

    def _set_pivots(self, mol):
        '''
        params mol: Hypermolecule class
        (Cyclical embed) Function that sets the mol.pivots attribute, that is a list
        containing each vector connecting two orbitals on different atoms or on the
        same atom (for single-reactive atom molecules in chelotropic embedding)
        '''
        mol.pivots = self._get_pivots(mol)

        if len(mol.pivots) == 2:
        # reactive atoms have one and two centers,
        # respectively. Apply bridging carboxylic acid correction.
            symbols = [atom.symbol for atom in mol.reactive_atoms_classes_dict.values()]
            if 'H' in symbols:
                if 'O' in symbols or 'S' in symbols:
                    if max([np.linalg.norm(p.pivot) for p in mol.pivots]) < 4.5:
                        class_types = [str(atom) for atom in mol.reactive_atoms_classes_dict.values()]
                        if 'Single Bond' in class_types and 'Ketone' in class_types:
                        # if we have a bridging acid, remove the longest of the two pivots,
                        # as it would lead to weird structures
                            norms = np.linalg.norm([p.pivot for p in mol.pivots], axis=1)
                            for sample in norms:
                                to_keep = [i for i in norms if sample >= i]
                                if len(to_keep) == 1:
                                    mask = np.array([i in to_keep for i in norms])
                                    mol.pivots = mol.pivots[mask]
                                    break

        if self.options.suprafacial:
            if len(mol.pivots) == 4:
            # reactive atoms have two centers each.
            # Applying suprafacial correction, only keeping
            # the shorter two, as they SHOULD be the suprafacial ones
                norms = np.linalg.norm([p.pivot for p in mol.pivots], axis=1)
                for sample in norms:
                    to_keep = [i for i in norms if sample >= i]
                    if len(to_keep) == 2:
                        mask = np.array([i in to_keep for i in norms])
                        mol.pivots = mol.pivots[mask]
                        break

    def _get_pivots(self, mol):
        '''
        params mol: Hypermolecule class
        (Cyclical embed) Function that yields the molecule pivots. Called by _set_pivots
        and in pre-conditioning (deforming, bending) the molecules in ase_bend.
        '''
        pivots_list = []

        if len(mol.reactive_atoms_classes_dict) == 2:
        # most molecules: dienes and alkenes for Diels-Alder, conjugated ketones for acid-bridged additions

            indexes = cartesian_product(*[range(len(atom.center)) for atom in mol.reactive_atoms_classes_dict.values()])
            # indexes of vectors in reactive_atom.center. Reactive atoms are 2 and so for one center on atom 0 and 
            # 2 centers on atom 2 we get [[0,0], [0,1], [1,0], [1,1]]

            for i,j in indexes:
                c1 = list(mol.reactive_atoms_classes_dict.values())[0].center[i]
                c2 = list(mol.reactive_atoms_classes_dict.values())[1].center[j]
                pivots_list.append(Pivot(c1, c2, i, j))

        elif len(mol.reactive_atoms_classes_dict) == 1:
        # carbenes, oxygen atom in Prilezhaev reaction, SO2 in classic chelotropic reactions

            indexes = cartesian_product(*[range(len(list(mol.reactive_atoms_classes_dict.values())[0].center)) for _ in range(2)])
            indexes = [i for i in indexes if i[0] != i[1] and (sorted(i) == i).all()]
            # indexes of vectors in reactive_atom.center. Reactive atoms is just one, that builds pivots with itself. 
            # pivots with the same index or inverse order are discarded. 2 centers on one atom 2 yield just [[0,1]]
            
            for i,j in indexes:
                c1 = list(mol.reactive_atoms_classes_dict.values())[0].center[i]
                c2 = list(mol.reactive_atoms_classes_dict.values())[0].center[j]
                pivots_list.append(Pivot(c1, c2, i, j))

        return np.array(pivots_list)

    def _setup(self):
        '''
        Setting embed type and calculating the number of conformation combinations based on embed type
        '''

        if len(self.objects) in (2,3):
        # Setting embed type and calculating the number of conformation combinations based on embed type

            cyclical = all([len(molecule.reactive_atoms_classes_dict) == 2 for molecule in self.objects])
            chelotropic = sorted([len(molecule.reactive_atoms_classes_dict) for molecule in self.objects]) == [1,2]
            string = all([len(molecule.reactive_atoms_classes_dict) == 1 for molecule in self.objects]) and len(self.objects) == 2

            if cyclical or chelotropic:

                if cyclical:
                    self.embed = 'cyclical'
                else:
                    self.embed = 'chelotropic'
                    for mol in self.objects:
                        for index, atom in mol.reactive_atoms_classes_dict.items():
                            orb_dim = np.linalg.norm(atom.center[0]-atom.coord)
                            atom.init(mol, index, update=True, orb_dim=orb_dim + 0.2)
                    # Slightly enlarging orbitals for chelotropic embeds, or they will
                    # be always generated too close to each other for how the cyclical embed works          

                self.options.rotation_steps = 6

                if hasattr(self.options, 'custom_rotation_steps'):
                # if user specified a custom value, use it.
                    self.options.rotation_steps = self.options.custom_rotation_steps

                self.candidates = (self.options.rotation_steps+1)**len(self.objects)*np.prod([len(mol.atomcoords) for mol in self.objects])
                
                if len(self.objects) == 3:
                    self.candidates *= 8
                else:
                    self.candidates *= 2
                # The number 8 is the number of different triangles originated from three oriented vectors,
                # while 2 is the disposition of two vectors (parallel, antiparallel). This ends here if
                # no parings are to be respected. If there are any, each one reduces the number of
                # candidates to be computed, and we divide self.candidates number in the next section.

                for molecule in self.objects:
                    self._set_pivots(molecule)

                self.candidates *= np.prod([len(mol.pivots) for mol in self.objects])

                if self.pairings != []:

                    if len(self.objects) == 2 and not chelotropic:
                    # diels-alder-like, if we have a pairing only half
                    # of the total arrangements are to be checked
                        n = 2

                    elif len(self.objects) == 3:

                        if len(self.pairings) == 1:
                            n = 4
                        else: # trimolecular, 2 or 3 pairings imposed
                            n = 8

                    else:
                        n = 1

                    self.candidates /= n
                # if the user specified some pairings to be respected, we have less candidates to check
                self.candidates = int(self.candidates)

            elif string:
                
                self.embed = 'string'
                self.options.rotation_steps = 24

                if hasattr(self.options, 'custom_rotation_steps'):
                # if user specified a custom value, use it.
                    self.options.rotation_steps = self.options.custom_rotation_steps

                self.candidates = self.options.rotation_steps*np.prod([len(mol.atomcoords) for mol in self.objects])*np.prod([len(list(mol.reactive_atoms_classes_dict.values())[0].center) for mol in self.objects])
                
            else:
                raise InputError(('Bad input - The only molecular configurations accepted are:\n' + 
                                  '1) Two or three molecules with two reactive centers each (cyclical embed)\n' + 
                                  '2) Two molecules with one reactive center each (string embed)\n' +
                                  '3) Two molecules, one with a single reactive center and the other with two (chelotropic embed)'))
        else:
            raise InputError('Bad input - too many or too few molecules specified (2 or 3 are required).')


        self.log(f'--> Setup performed correctly. {self.candidates} candidates will be generated.\n')

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
                                  mol.reactive_indexes[1]+sum(len(m.atomnos) for m in self.objects[0:self.objects.index(mol)])] if len(mol.reactive_indexes) == 2 else [
                                  mol.reactive_indexes[0]+sum(len(m.atomnos) for m in self.objects[0:self.objects.index(mol)]),
                                  mol.reactive_indexes[0]+sum(len(m.atomnos) for m in self.objects[0:self.objects.index(mol)])] for mol in self.objects]

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

        self.log(f'\n--> Performing string embed ({self.candidates} candidates)')

        for mol in self.objects:
            mol.centers = list(mol.reactive_atoms_classes_dict.values())[0].center
            mol.orb_vers = np.array([norm(v) for v in list(mol.reactive_atoms_classes_dict.values())[0].orb_vecs])

        centers_indexes = cartesian_product(*[np.array(range(len(mol.centers))) for mol in self.objects])
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
                cc = np.array([x,y])
                # 2D coordinates of the triangle circocenter

                v0, v1, v2 = vertexes

                meanpoint1 = np.mean((v0,v1), axis=0)
                meanpoint2 = np.mean((v1,v2), axis=0)
                meanpoint3 = np.mean((v2,v0), axis=0)

                dir1 = cc - meanpoint1
                dir2 = cc - meanpoint2
                dir3 = cc - meanpoint3
                # 2D direction versors connecting center of side with circumcenter.
                # Now we need to understand if we want these or their negative

                if np.any([np.all(d == 0) for d in (dir1, dir2, dir3)]):
                # We have a right triangle. To aviod numerical
                # errors, a small perturbation is made.
                # This should not happen, but just in case...
                    norms[0] += 1e-5
                    dir1, dir2, dir3 = [t[:-1] for t in _get_directions(norms)]

                angle0_obtuse = (vec_angle(v1-v0, v2-v0) > 90)
                angle1_obtuse = (vec_angle(v0-v1, v2-v1) > 90)
                angle2_obtuse = (vec_angle(v0-v2, v1-v2) > 90)

                dir1 = -dir1 if angle2_obtuse else dir1
                dir2 = -dir2 if angle0_obtuse else dir2
                dir3 = -dir3 if angle1_obtuse else dir3
                # invert the versors sign if circumcenter is
                # one angle is obtuse, because then
                # circumcenter is outside the triangle
                
                dir1 = norm(np.concatenate((dir1, [0])))
                dir2 = norm(np.concatenate((dir2, [0])))
                dir3 = norm(np.concatenate((dir3, [0])))

                return np.vstack((dir1, dir2, dir3))

        def _adjust_directions(self, directions, constrained_indexes, triangle_vectors, v, pivots):
            '''
            For trimolecular TSs, correct molecules pre-alignment. That is, after the initial estimate
            based on pivot triangle circocentrum, systematically rotate each molecule around its pivot
            by fixed increments and look for the arrangement with the smallest deviation from orbital
            parallel interaction. This optimizes the obtainment of poses with the correct inter-reactive
            atoms distances.

            '''
            assert directions.shape[0] == 3

            mols = deepcopy(self.objects)
            p0, p1, p2 = [end - start for start, end in triangle_vectors]
            p0_mean, p1_mean, p2_mean = [np.mean((end, start), axis=0) for start, end in triangle_vectors]

            ############### get triangle vertexes

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
            cc = np.array([x,y])
            # 2D coordinates of the triangle circocenter

            v0, v1, v2 = vertexes

            v0 = np.concatenate((v0, [0]))
            v1 = np.concatenate((v1, [0]))
            v2 = np.concatenate((v2, [0]))

            ############### set up mols -> pos + rot

            alignment_rotation = np.zeros((3,3,3))
            for i in (0,1,2):

                start, end = triangle_vectors[i]

                mol_direction = pivots[i].meanpoint - np.mean(self.objects[i].atomcoords[0][self.objects[i].reactive_indexes], axis=0)
                if np.all(mol_direction == 0.):
                    mol_direction = pivots[i].meanpoint

                mols[i].rotation = R.align_vectors((end-start, directions[i]), (pivots[i].pivot, mol_direction))[0].as_matrix()
                mols[i].position = np.mean(triangle_vectors[i], axis=0) - mols[i].rotation @ pivots[i].meanpoint
 
            ############### set up pairings between reactive atoms

            pairings = [[None, None] for _ in constrained_indexes]
            for i, c in enumerate(constrained_indexes):
                for m, mol in enumerate(self.objects):
                    for index, r_atom in mol.reactive_atoms_classes_dict.items():
                        if r_atom.cumnum == c[0]:
                            pairings[i][0] = (m, index)
                        if r_atom.cumnum == c[1]:
                            pairings[i][1] = (m, index)

            r = np.zeros((3,3), dtype=int)

            for first, second in pairings:
                mol_index = first[0]
                partner_index = second[0]
                reactive_index = first[1]
                r[mol_index, partner_index] = reactive_index

                mol_index = second[0]
                partner_index = first[0]
                reactive_index = second[1]
                r[mol_index, partner_index] = reactive_index

            # r[0,1] is the reactive_index of molecule 0 that faces molecule 1 and so on
            # diagonal of r (r[0,0], r[1,1], r[2,2]) is just unused

            ############### calculate reactive atoms positions

            mol0, mol1, mol2 = mols

            a01 = mol0.rotation @ mol0.atomcoords[0][r[0,1]] + mol0.position
            a02 = mol0.rotation @ mol0.atomcoords[0][r[0,2]] + mol0.position

            a10 = mol1.rotation @ mol1.atomcoords[0][r[1,0]] + mol1.position
            a12 = mol1.rotation @ mol1.atomcoords[0][r[1,2]] + mol1.position

            a20 = mol2.rotation @ mol2.atomcoords[0][r[2,0]] + mol2.position
            a21 = mol2.rotation @ mol2.atomcoords[0][r[2,1]] + mol2.position

            ############### explore all angles combinations

            steps = 6
            angle_range = 30
            step_angle = 2*angle_range/steps
            angles_list = cartesian_product(*[range(steps+1) for _ in range(3)]) * step_angle - angle_range
            # Molecules are rotated around the +angle_range/-angle_range range in the given number of steps.
            # Therefore, the angular resolution between candidates is step_angle (10 degrees)

            candidates = []
            for angles in angles_list:

                rot0 = rot_mat_from_pointer(p0, angles[0])
                new_a01 = rot0 @ a01
                new_a02 = rot0 @ a02
                d0 = p0_mean - np.mean((new_a01, new_a02), axis=0)

                rot1 = rot_mat_from_pointer(p1, angles[1])
                new_a10 = rot1 @ a10
                new_a12 = rot1 @ a12
                d1 = p1_mean - np.mean((new_a10, new_a12), axis=0)

                rot2 = rot_mat_from_pointer(p2, angles[2])
                new_a20 = rot2 @ a20
                new_a21 = rot2 @ a21
                d2 = p2_mean - np.mean((new_a20, new_a21), axis=0)

                cost = 0
                cost += vec_angle(v0 - new_a02, new_a20 - v0)
                cost += vec_angle(v1 - new_a01, new_a10 - v1)
                cost += vec_angle(v2 - new_a21, new_a12 - v2)
                        
                candidates.append((cost, angles, (d0, d1, d2)))

            ############### choose the one with the best alignment, that is minor cost

            cost, angles, directions = sorted(candidates, key=lambda x: x[0])[0]
            
            return np.array(directions)

                
        self.log(f'\n--> Performing {self.embed} embed ({self.candidates} candidates)')

        steps = self.options.rotation_steps + 1
        angle_range = self.options.rotation_range
        systematic_angles = cartesian_product(*[range(steps) for _ in self.objects]) * 2*angle_range/steps - angle_range

        pivots_indexes = cartesian_product(*[range(len(mol.pivots)) for mol in self.objects])
        # indexes of pivots in each molecule self.pivots list. For three mols with 2 pivots each: [[0,0,0], [0,0,1], [0,1,0], ...]
       
        threads = []
        constrained_indexes = []
        for p, pi in enumerate(pivots_indexes):

            loadbar(p, len(pivots_indexes), prefix=f'Embedding structures ')
            
            thread_objects = deepcopy(self.objects)
            # Objects to be used to embed structures. Modified later if necessary.

            pivots = [thread_objects[m].pivots[pi[m]] for m in range(len(self.objects))]
            # getting the active pivot for each molecule for this run
            
            norms = np.linalg.norm(np.array([p.pivot for p in pivots]), axis=1)
            # getting the pivots norms to feed into the polygonize function

            try:
                polygon_vectors = polygonize(norms)

            except TriangleError:
                # Raised if we cannot build a triangle with the given norms.
                # Try to bend the structure if it was close or just skip this triangle and go on.

                deltas = [norms[i] - (norms[i-1] + norms[i-2]) for i in range(3)]

                rel_delta = max([deltas[i]/norms[i] for i in range(3)])
                # s = 'Rejected triangle, delta was %s, %s of side length' % (round(delta, 3), str(round(100*rel_delta, 3)) + ' %')
                # self.log(s, p=False)

                if rel_delta < 0.2 and not self.options.rigid:
                # correct the molecule structure with the longest
                # side if the distances are at most 20% off.

                    index = deltas.index(max(deltas))
                    mol = thread_objects[index]
                    pivot = pivots[index]

                    # ase_view(mol)
                    maxval = norms[index-1] + norms[index-2]
                    bent_mol = ase_bend(self, mol, pivot, 0.9*maxval, traj='ase_bend_v2.traj')
                    # ase_view(bent_mol)

                    ###################################### DEBUGGING PURPOSES

                    # for p in bent_mol.pivots:
                    #     if p.index == pivot.index:
                    #         new_pivot = p
                    # now = np.linalg.norm(new_pivot.pivot)
                    # self.log(f'Corrected Triangle: Side was {round(norms[index], 3)} A, now it is {round(now, 3)}, {round(now/maxval, 3)} % of maximum value for triangle creation', p=False)
                    
                    #########################################################

                    thread_objects[index] = bent_mol

                    pivots = [thread_objects[m].pivots[pi[m]] for m in range(len(self.objects))]
                    # updating the active pivot for each molecule for this run
                    
                    norms = np.linalg.norm(np.array([p.pivot for p in pivots]), axis=1)
                    # updating the pivots norms to feed into the polygonize function

                    polygon_vectors = polygonize(norms)
                    # repeating the failed polygon creation

                else:
                    continue
            
            except TooDifferentLengthsError:

                if not self.options.rigid:
    
                    if self.embed == 'chelotropic':
                        target_length = min(norms)

                    else:
                        maxgap = 3 # in Angstrom
                        gap = abs(norms[0]-norms[1])
                        r = 0.5 + 0.5*(gap/maxgap)
                        r = np.clip(5, 0.5, 1)
                        # r is the ratio for the target_length based on the gap
                        # that deformations will need to cover.
                        # It ranges from 0.5 to 1 and if shifted more toward
                        # the shorter norm as the gap rises. For gaps of more
                        # than 3 Angstroms, basically only the molecule
                        # with the longest pivot is bent.

                        target_length = min(norms)*r + max(norms)*(1-r)
                
                    for i, mol in enumerate(deepcopy(thread_objects)):

                        if len(mol.reactive_indexes) > 1:
                        # do not try to bend molecules that react with a single atom

                            if not tuple(sorted(mol.reactive_indexes)) in list(mol.graph.edges):
                            # do not try to bend molecules where the two reactive indices are bonded

                                bent_mol = ase_bend(self, mol, pivots[i], target_length, method=f'{self.options.mopac_level}')
                                # ase_view(bent_mol) TODO - remove traj
                                thread_objects[i] = bent_mol

                    # Repeating the previous polygonization steps with the bent molecules

                    pivots = [thread_objects[m].pivots[pi[m]] for m in range(len(self.objects))]
                    # updating the active pivot for each molecule for this run
                    
                    norms = np.linalg.norm(np.array([p.pivot for p in pivots]), axis=1)
                    # updating the pivots norms to feed into the polygonize function

                    
                    polygon_vectors = polygonize(norms, override=True)
                    # repeating the failed polygon creation



            directions = _get_directions(norms)
            # directions to orient the molecules toward, orthogonal to each vec_pair

            for v, vecs in enumerate(polygon_vectors):
            # getting vertexes to embed molecules with and iterating over start/end points

                ids = self.get_cyclical_reactive_indexes(v)
                # get indexes of atoms that face each other

                if self.pairings == [] or all([pair in ids for pair in self.pairings]):
                # ensure that the active arrangement has all the pairings that the user specified

                    if len(self.objects) == 3:

                        directions = _adjust_directions(self, directions, ids, vecs, v, pivots)
                        # For trimolecular TSs, the alignment direction previously get is 
                        # just a general first approximation that needs to be corrected
                        # for the specific case through another algorithm.
                        
                    for angles in systematic_angles:

                        threads.append([deepcopy(obj) for obj in thread_objects])
                        thread = threads[-1]
                        # generating the thread we are going to modify and setting the thread pointer

                        constrained_indexes.append(ids)
                        # Save indexes to be constrained later in the optimization step

                        # orb_list = []

                        for i, vec_pair in enumerate(vecs):
                        # setting molecular positions and rotations (embedding)
                        # i is the molecule index, vecs is a tuple of start and end positions
                        # for the pivot vector

                            start, end = vec_pair
                            angle = angles[i]

                            reactive_coords = thread_objects[i].atomcoords[0][thread_objects[i].reactive_indexes]
                            # coordinates for the reactive atoms in this run

                            atomic_pivot_mean = np.mean(reactive_coords, axis=0)
                            # mean position of the atoms active in this run 

                            mol_direction = pivots[i].meanpoint-atomic_pivot_mean
                            if np.all(mol_direction == 0.):
                                mol_direction = pivots[i].meanpoint
                                # log.write(f'mol {i} - improper pivot? Thread {len(threads)-1}\n')

                            # Direction in which the molecule should be oriented, based on the meand of reactive
                            # atom positions and the mean point of the active pivot for the run.
                            # If this vector is too small and gets rounded to zero (as it can happen for
                            # "antrafacial" vectors), we fallback to the vector starting from the molecule
                            # center (mean of atomic positions) and ending in pivot_means[i], so to avoid
                            # numeric errors in the next function.
                             
                            alignment_rotation = R.align_vectors((end-start, directions[i]),
                                                                 (pivots[i].pivot, mol_direction))[0].as_matrix()
                            # this rotation superimposes the molecular orbitals active in this run (pivots[i].pivot
                            # goes to end-start) and also aligns the molecules so that they face each other
                            # (mol_direction goes to directions[i])
                            
                            if len(reactive_coords) == 2:
                                axis_of_step_rotation = alignment_rotation @ (reactive_coords[0]-reactive_coords[1])
                            else:
                                axis_of_step_rotation = alignment_rotation @ pivots[i].pivot
                            # molecules with two reactive atoms are step-rotated around the line connecting
                            # the reactive atoms, while single reactive atom mols around their active pivot

                            step_rotation = rot_mat_from_pointer(axis_of_step_rotation, angle)
                            # this rotation cycles through all different rotation angles for each molecule

                            center_of_rotation = alignment_rotation @ atomic_pivot_mean
                            # center_of_rotation is the mean point between the reactive atoms so
                            # as to keep the reactive distances constant

                            thread[i].rotation = step_rotation @ alignment_rotation# @ pre_alignment_rot
                            # overall rotation for the molecule is given by the matrices product

                            pos = np.mean(vec_pair, axis=0) - alignment_rotation @ pivots[i].meanpoint
                            thread[i].position += center_of_rotation - step_rotation @ center_of_rotation + pos
                            # overall position is given by superimposing mean of active pivot (connecting orbitals)
                            # to mean of vec_pair (defining the target position - the side of a triangle for three molecules)

                            # orb_list.append(start)
                            # orb_list.append(end)

                        ################# DEBUGGING OUTPUT

                        # with open('orbs.xyz', 'a') as f:
                        #     an = np.array([3 for _ in range(len(orb_list))])
                        #     write_xyz(np.array(orb_list), an, f)

                        # totalcoords = np.concatenate([[mol.rotation @ v + mol.position for v in mol.atomcoords[0]] for mol in thread])
                        # totalatomnos = np.concatenate([mol.atomnos for mol in thread])
                        # total_reactive_indexes = np.array(ids).ravel()

                        # with open('debug_step.xyz', 'w') as f:
                        #     write_xyz(totalcoords, totalatomnos, f)
                        #     # write_xyz(totalcoords[total_reactive_indexes], totalatomnos[total_reactive_indexes], f)

                        # # os.system('vmd debug_step.xyz')
                        # os.system(r'vmd -e C:\Users\Nik\Desktop\Coding\TSCoDe\Resources\tri\temp_bonds.vmd')
                        # quit()
                        # TODO - clean


        loadbar(1, 1, prefix=f'Embedding structures ')

        self.dispositions_constrained_indexes = np.array(constrained_indexes)

        return threads

    def _set_default_distances(self):
        '''
        Called before TS refinement if user did not specify all
        (or any) of bonding distances with the DIST keyword.
        '''
        if not hasattr(self, 'pairings_dists'):
            self.pairings_dists = []

        r_atoms = np.array([list(mol.reactive_atoms_classes_dict.values()) for mol in self.objects]).ravel()

        for letter, indexes in self.pairings_table.items():
            index1, index2 = indexes
            for r_atom in r_atoms:
                if index1 == r_atom.cumnum:
                    r_atom1 = r_atom
                if index2 == r_atom.cumnum:
                    r_atom2 = r_atom

            if letter not in [letter for letter, _ in self.pairings_dists]:

                dist1 = np.linalg.norm(r_atom1.center[0] - r_atom1.coord)
                dist2 = np.linalg.norm(r_atom2.center[0] - r_atom2.coord)

                self.pairings_dists.append((letter, dist1+dist2))


######################################################################################################### RUN

    def run(self):
        '''
        '''
        global scramble

        try:

            if not self.options.let:
                assert self.candidates < 1e8, ('ATTENTION! This calculation is probably going to be very big. To ignore this message' +
                                               ' and proceed, add the LET keyword to the input file.')

            head = ''
            for i, mol in enumerate(self.objects):
                s = [atom.symbol+'('+str(atom)+f', {round(np.linalg.norm(atom.center[0]-atom.coord), 3)} A)' for atom in mol.reactive_atoms_classes_dict.values()]
                t = ', '.join([f'{a} -> {b}' for a,b in zip(mol.reactive_indexes, s)])
                head += f'    {i+1}. {mol.name}: {t}\n'

            self.log('--> Input structures, reactive indexes and reactive atoms TSCODE type and orbital dimensions:\n' + head)
            self.log(f'--> Calculation options used were:')
            for line in str(self.options).split('\n'):
                if self.embed == 'string':
                    if line.split()[0] in ('rotation_range', 'rigid', 'suprafacial'):
                        continue
                self.log(f'    - {line}')

            t_start_run = time.time()

            if self.embed == 'cyclical' or self.embed == 'chelotropic':
                embedded_structures = self.cyclical_embed()

                if embedded_structures == []:
                    s = ('\n--> Cyclical embed did not find any suitable disposition of molecules.\n' +
                         '    This is probably because one molecule has two reactive centers at a great distance,\n' +
                         '    preventing the other two molecules from forming a closed, cyclical structure.')
                    self.log(s, p=False)
                    raise ZeroCandidatesError(s)

            else:
                embedded_structures = self.string_embed()


            atomnos = np.concatenate([molecule.atomnos for molecule in self.objects])
            # cumulative list of atomic numbers associated with coordinates

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
            self.log(f'Generated {len(self.structures)} transition state candidates ({round(t_end-t_start_run, 2)} s)\n')

            if not self.options.bypass:
                try:
                    ################################################# COMPENETRATION CHECK
                    
                    self.log('--> Checking structures for compenetrations')

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
                        self.log(f'Discarded {len([b for b in mask if b == False])} candidates for compenetration ({len([b for b in mask if b == True])} left, {round(t_end-t_start, 2)} s)')
                    self.log()
                    # Performing a sanity check for excessive compenetration on generated structures, discarding the ones that look too bad

                    ################################################# PRUNING: SIMILARITY

                    if len(self.structures) == 0:
                        raise ZeroCandidatesError()

                    self.log('--> Similarity Processing')
                    t_start = time.time()

                    before = len(self.structures)
                    for k in (5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2):
                        if 5*k < len(self.structures):
                            t_start_int = time.time()
                            self.structures, mask = prune_conformers(self.structures, atomnos, max_rmsd=self.options.pruning_thresh, k=k)
                            self.constrained_indexes = self.constrained_indexes[mask]
                            t_end_int = time.time()
                            self.log(f'    - similarity pre-processing   (k={k}) - {round(t_end_int-t_start_int, 2)} s - kept {len([b for b in mask if b == True])}/{len(mask)}')
                    
                    t_start_int = time.time()
                    self.structures, mask = prune_conformers(self.structures, atomnos, max_rmsd=self.options.pruning_thresh)
                    t_end = time.time()
                    self.log(f'    - similarity final processing (k=1) - {round(t_end-t_start_int, 2)} s - kept {len([b for b in mask if b == True])}/{len(mask)}')

                    self.constrained_indexes = self.constrained_indexes[mask]

                    if np.any(mask == False):
                        self.log(f'Discarded {before - len(np.where(mask == True)[0])} candidates for similarity ({len([b for b in mask if b == True])} left, {round(t_end-t_start, 2)} s)')
                    self.log()

                    ################################################# CHECKPOINT SAVE BEFORE OPTIMIZATION

                    outname = f'TSCoDe_checkpoint_{self.stamp}.xyz'
                    with open(outname, 'w') as f:        
                        for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):
                            write_xyz(structure, atomnos, f, title=f'TS candidate {i+1} - Checkpoint before optimization')
                    t_end_run = time.time()
                    self.log(f'--> Checkpoint output - Wrote {len(self.structures)} TS structures to {outname} file before optimization.\n')

                    ################################################# GEOMETRY OPTIMIZATION


                    if len(self.structures) == 0:
                        raise ZeroCandidatesError()

                    self.energies = np.zeros(len(self.structures))
                    self.exit_status = np.zeros(len(self.structures), dtype=bool)

                    t_start = time.time()
                    if self.options.optimization:

                        self.log(f'--> Structure optimization ({self.options.mopac_level} level)')

                        for i, structure in enumerate(deepcopy(self.structures)):
                            loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')
                            try:
                                t_start_opt = time.time()
                                self.structures[i], self.energies[i], self.exit_status[i] = optimize(structure,
                                                                                                     atomnos,
                                                                                                     graphs,
                                                                                                     self.constrained_indexes[i],
                                                                                                     method=f'{self.options.mopac_level} GEO-OK CYCLES=500',
                                                                                                     max_newbonds=self.options.max_newbonds)


                                exit_str = 'CONVERGED' if self.exit_status[i] else 'SCRAMBLED'

                            except MopacReadError:
                                # ase will throw a ValueError if the output lacks a space in the "FINAL POINTS AND DERIVATIVES" table.
                                # This occurs when one or more of them is not defined, that is when the calculation did not end well.
                                # The easiest solution is to reject the structure and go on.
                                self.structures[i] = None
                                self.energies[i] = np.inf
                                self.exit_status[i] = False
                                exit_str = 'FAILED TO READ FILE'

                            except Exception as e:
                                raise e

                            t_end_opt = time.time()
                            self.log(f'    - Mopac {self.options.mopac_level} optimization: Structure {i+1} {exit_str} - took {round(t_end_opt-t_start_opt, 2)} s', p=False)

                        loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')
                        t_end = time.time()
                        self.log(f'Mopac {self.options.mopac_level} optimization took {round(t_end-t_start, 2)} s (~{round((t_end-t_start)/len(self.structures), 2)} s per structure)')

                        ################################################# PRUNING: EXIT STATUS

                        mask = self.exit_status
                        self.structures = self.structures[mask]
                        self.energies = self.energies[mask]
                        self.constrained_indexes = self.constrained_indexes[mask]
                        self.exit_status = self.exit_status[mask]

                        if np.any(mask == False):
                            self.log(f'Discarded {len(np.where(mask == False)[0])} candidates because optimizations failed or scrambled some atoms ({len([b for b in mask if b == True])} left)')
                        

                        ################################################# PRUNING: ENERGY

                        if THRESHOLD_KCAL != None:
                    
                            if len(self.structures) == 0:
                                raise ZeroCandidatesError()

                            mask = (self.energies - np.min(self.energies)) < THRESHOLD_KCAL
                            self.structures = self.structures[mask]
                            self.energies = self.energies[mask]
                            self.exit_status = self.exit_status[mask]

                            _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
                            self.energies = scramble(self.energies, sequence)
                            self.structures = scramble(self.structures, sequence)
                            self.constrained_indexes = scramble(self.constrained_indexes, sequence)
                            # sorting structures based on energy

                            if np.any(mask == False):
                                self.log(f'Discarded {len(np.where(mask == False)[0])} candidates for energy (Threshold set to {THRESHOLD_KCAL} kcal/mol)')

                        ################################################# PRUNING: SIMILARITY (POST PARTIAL OPT)

                        if len(self.structures) == 0:
                            raise ZeroCandidatesError()

                        t_start = time.time()
                        self.structures, mask = prune_conformers(self.structures, atomnos, max_rmsd=self.options.pruning_thresh)
                        self.energies = self.energies[mask]
                        self.exit_status = self.exit_status[mask]
                        t_end = time.time()
                        
                        if np.any(mask == False):
                            self.log(f'Discarded {len(np.where(mask == False)[0])} candidates for similarity ({len([b for b in mask if b == True])} left, {round(t_end-t_start, 2)} s)')
                        self.log()

                        ################################################# REFINING: BONDING DISTANCES

                        self.log(f'--> Refining bonding distances for TSs ({self.options.mopac_level} level)')

                        # backing up structures before refinement
                        outname = f'TSCoDe_TSs_guesses_unrefined_{self.stamp}.xyz'
                        with open(outname, 'w') as f:        
                            for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):
                                write_xyz(structure, atomnos, f, title=f'Structure {i+1} - NOT REFINED')

                        os.remove(f'TSCoDe_checkpoint_{self.stamp}.xyz')
                        # We don't need the pre-optimized structures anymore

                        if not hasattr(self, 'pairings_dists') or len(self.pairings) > len(self.pairings_dists):
                        # if user did not specify all (or any) of the distances
                        # between imposed pairings, default values will be used
                            self._set_default_distances()

                        for i, structure in enumerate(deepcopy(self.structures)):
                            loadbar(i, len(self.structures), prefix=f'Refining structure {i+1}/{len(self.structures)} ')
                            try:
                                t_start_opt = time.time()
                                new_structure, new_energy, self.exit_status[i] = ase_adjust_spacings(self, structure, atomnos, graphs)

                                if self.exit_status[i]:
                                    self.structures[i] = new_structure
                                    self.energies[i] = new_energy
                                    exit_str = 'REFINED'
                                else:
                                    exit_str = 'SCRAMBLED'
                                                                                                                                        
                            except ValueError as e:
                                # ase will throw a ValueError if the output lacks a space in the "FINAL POINTS AND DERIVATIVES" table.
                                # This occurs when one or more of them is not defined, that is when the calculation did not end well.
                                # The easiest solution is to reject the structure and go on. TODO-check
                                self.log(e)
                                self.log(f'Failed to read MOPAC file for Structure {i+1}, skipping distance refinement', p=False)                                    

                            finally:
                                t_end_opt = time.time()
                                self.log(f'    - Mopac {self.options.mopac_level} refinement: Structure {i+1} {exit_str} - took {round(t_end_opt-t_start_opt, 2)} s', p=False)
                        
                        loadbar(1, 1, prefix=f'Refining structure {i+1}/{len(self.structures)} ')
                        t_end = time.time()
                        self.log(f'Mopac {self.options.mopac_level} refinement took {round(t_end-t_start, 2)} s (~{round((t_end-t_start)/len(self.structures), 2)} s per structure)')

                        before = len(self.structures)
                        if self.options.only_refined:
                            mask = self.exit_status
                            self.structures = self.structures[mask]
                            self.energies = self.energies[mask]
                            self.exit_status = self.exit_status[mask]
                            s = f'Discarded {len([i for i in mask if i == False])} unrefined structures'

                        else:
                            s = 'Non-refined ones will not be discarded.'


                        self.log(f'Successfully refined {len([i for i in self.exit_status if i == True])}/{before} structures. {s}')

                        ################################################# PRUNING: ENERGY, AFTER REFINEMENT
                    
                        if THRESHOLD_KCAL != None:

                            mask = (self.energies - np.min(self.energies)) < THRESHOLD_KCAL
                            self.structures = self.structures[mask]
                            self.energies = self.energies[mask]

                            _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
                            self.energies = scramble(self.energies, sequence)
                            self.structures = scramble(self.structures, sequence)
                            self.constrained_indexes = scramble(self.constrained_indexes, sequence)
                            # sorting structures based on energy

                            if np.any(mask == False):
                                self.log(f'Discarded {len(np.where(mask == False)[0])} candidates for energy (Threshold set to {THRESHOLD_KCAL} kcal/mol)')

                        ################################################# PRUNING: SIMILARITY (POST REFINEMENT)

                        if len(self.structures) == 0:
                            raise ZeroCandidatesError()

                        t_start = time.time()
                        self.structures, mask = prune_conformers(self.structures, atomnos, max_rmsd=self.options.pruning_thresh)
                        self.energies = self.energies[mask]
                        t_end = time.time()
                        
                        if np.any(mask == False):
                            self.log(f'Discarded {len(np.where(mask == False)[0])} candidates for similarity ({len([b for b in mask if b == True])} left, {round(t_end-t_start, 2)} s)')
                        self.log()


                except ZeroCandidatesError:
                    t_end_run = time.time()
                    s = ('Sorry, the program did not find any reasonable TS structure. Are you sure the input indexes were correct? If so, try these tips:\n' + 
                         '    - Enlarging the search space by specifying a larger "steps" value with the keyword STEPS=n\n' +
                         '    - Imposing less strict rejection criteria with the DEEP or CLASHES keyword.\n')
                    self.log(f'\n--> Program termination: No candidates found - Total time {round(t_end_run-t_start_run, 2)} s')
                    self.log(s)
                    clean_directory()
                    raise ZeroCandidatesError(s)

            else:
                self.energies = np.zeros(len(self.structures))

            ################################################# XYZ GUESSES OUTPUT 

            if self.options.optimization:

                self.energies = self.energies - np.min(self.energies)
                _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
                self.energies = scramble(self.energies, sequence)
                self.structures = scramble(self.structures, sequence)
                self.constrained_indexes = scramble(self.constrained_indexes, sequence)
                # sorting structures based on energy


                outname = f'TSCoDe_TSs_guesses_{self.stamp}.xyz'
                with open(outname, 'w') as f:        
                    for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):

                        kind = 'REFINED - ' if self.exit_status[i] else 'NOT REFINED - '

                        write_xyz(structure, atomnos, f, title=f'Structure {i+1} - {kind}Rel. E. = {round(self.energies[i], 3)} kcal/mol')

                os.remove(f'TSCoDe_TSs_guesses_unrefined_{self.stamp}.xyz')
                # since we have the refined structures, we can get rid of the unrefined ones

                t_end_run = time.time()
                self.log(f'--> Output: Wrote {len(self.structures)} rough TS structures to {outname} file - Total time {round(t_end_run-t_start_run, 2)} s\n')

            ################################################# TS SEEKING: IRC + NEB

                if self.options.neb:
                
                    self.log(f'--> HyperNEB optimization ({self.options.mopac_level} level)')
                    t_start = time.time()

                    for i, structure in enumerate(self.structures):

                        loadbar(i, len(self.structures), prefix=f'Performing NEB {i+1}/{len(self.structures)} ')

                        t_start_opt = time.time()

                        try:

                            self.structures[i], self.energies[i] = hyperNEB(structure,
                                                                            atomnos,
                                                                            self.ids,
                                                                            self.constrained_indexes[i],
                                                                            reag_prod_method=f'{self.options.mopac_level}',
                                                                            NEB_method=f'{self.options.mopac_level} GEO-OK',
                                                                            title=f'structure_{i+1}')

                            exit_str = 'COMPLETED'
                            self.exit_status[i] = True

                        except (MopacReadError, ValueError):
                            # Both are thrown if a MOPAC file read fails, but the former occurs when an internal (TSCoDe)
                            # read fails (getting reagent or product), the latter when an ASE read fails (during NEB)
                            exit_str = 'CRASHED'
                            self.exit_status[i] = False

                        t_end_opt = time.time()

                        self.log(f'    - Mopac {self.options.mopac_level} NEB optimization: Structure {i+1} - {exit_str} - ({round(t_end_opt-t_start_opt, 2)} s)', p=False)

                    loadbar(1, 1, prefix=f'Performing NEB {len(self.structures)}/{len(self.structures)} ')
                    t_end = time.time()
                    self.log(f'Mopac {self.options.mopac_level} NEB optimization took {round(t_end-t_start, 2)} s ({round((t_end-t_start)/len(self.structures), 2)} s per structure)')
                    self.log(f'NEB converged for {len([i for i in self.exit_status if i == True])}/{len(self.structures)} structures\n')

                    mask = self.exit_status
                    self.structures = self.structures[mask]
                    self.energies = self.energies[mask]
                    self.exit_status = self.exit_status[mask]

                    ################################################# PRUNING: SIMILARITY (POST NEB)

                    if len(self.structures) != 0:

                        t_start = time.time()
                        self.structures, mask = prune_conformers(self.structures, atomnos, max_rmsd=self.options.pruning_thresh)
                        self.energies = self.energies[mask]
                        t_end = time.time()
                        
                        if np.any(mask == False):
                            self.log(f'Discarded {len(np.where(mask == False)[0])} candidates for similarity ({len([b for b in mask if b == True])} left, {round(t_end-t_start, 2)} s)')
                        self.log()

                    ################################################# TS CHECK - FREQUENCY CALCULATION


                        self.log(f'--> TS frequency calculation ({self.options.mopac_level} level)')
                        t_start = time.time()

                        for i, structure in enumerate(self.structures):

                            loadbar(i, len(self.structures), prefix=f'Performing frequency calculation {i+1}/{len(self.structures)} ')

                            mopac_opt(structure,
                                    atomnos,
                                    method=f'{self.options.mopac_level} FORCE',
                                    title=f'TS_{i+1}_FREQ',
                                    read_output=False)

                        loadbar(1, 1, prefix=f'Performing frequency calculation {i+1}/{len(self.structures)} ')
                        t_end = time.time()
                        self.log(f'Mopac {self.options.mopac_level} frequency calculation took {round(t_end-t_start, 2)} s (~{round((t_end-t_start)/len(self.structures), 2)} s per structure)\n')

                    ################################################# NEB XYZ OUTPUT

                        self.energies -= np.min(self.energies)
                        _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
                        self.energies = scramble(self.energies, sequence)
                        self.structures = scramble(self.structures, sequence)
                        self.constrained_indexes = scramble(self.constrained_indexes, sequence)
                        # sorting structures based on energy

                        outname = f'TSCoDe_NEB_TSs_{self.stamp}.xyz'
                        with open(outname, 'w') as f:        
                            for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):
                                write_xyz(structure, atomnos, f, title=f'Structure {i+1} - TS - Rel. E. = {round(self.energies[i], 3)} kcal/mol')

                        self.log(f'--> Output: Wrote {len(self.structures)} final TS structures to {outname} file')

            ################################################# OPTIONAL: PRINT NON-COVALENT INTERACTION GUESSES

            if self.options.nci and self.options.optimization:

                self.log('--> Non-covalent interactions spotting')
                self.nci = []

                for i, structure in enumerate(self.structures):

                    nci, print_list = get_nci(structure, atomnos, self.constrained_indexes[i], self.ids)
                    self.nci.append(nci)

                    if nci != []:
                        self.log(f'Structure {i+1}: {len(nci)} interactions')

                        for p in print_list:
                            self.log('    '+p)
                        self.log()
                
                if len([l for l in self.nci if l != []]) == 0:
                    self.log('No particular NCIs spotted for these structures\n')

                else:
                    unshared_nci = []
                    for i, nci_list in enumerate(self.nci):
                        for nci in nci_list:
                        # for each interaction of each structure

                            if not nci in [n[0] for n in unshared_nci]:
                            # if we have not already done it

                                if not all([nci in structure_nci for structure_nci in self.nci]):
                                # if the interaction is not shared by all structures, take note

                                    shared_by = [i for i, structure_nci in enumerate(self.nci) if nci in structure_nci]
                                    unshared_nci.append((nci, shared_by))

                    if unshared_nci != []:
                        self.log(f'--> Differential NCIs found - these might be the source of selectivity:')
                        for nci, shared_by in unshared_nci:
                            nci_type, i1, i2 = nci
                            self.log(f'    {nci_type} between indexes {i1}/{i2} is present in {len(shared_by)}/{len(self.structures)} structures {tuple([i+1 for i in shared_by])}')
                        self.log()

            ################################################# VMD OUTPUT

            vmd_name = outname.split('.')[0] + '.vmd'
            path = os.path.join(os.getcwd(), vmd_name)
            with open(path, 'w') as f:
                s = ('display resetview\n' +
                    'mol new {%s.xyz}\n' % (path.strip('.vmd')) +
                    'mol selection index %s\n' % (' '.join([str(i) for i in self.constrained_indexes[0].ravel()])) +
                    'mol representation CPK 0.7 0.5 50 50\n' +
                    'mol color ColorID 7\n' +
                    'mol material Transparent\n' +
                    'mol addrep top\n')

                for a, b in self.pairings:
                    s += f'label add Bonds 0/{a} 0/{b}\n'

                f.write(s)

            self.log(f'--> Output: Wrote VMD {vmd_name} file\n')
            
            ################################################# END: CLEAN ALL TEMP FILES AND CLOSE LOG

            clean_directory()

            t_end_run = time.time()
            total_time = t_end_run - t_start_run

            timestring = ''
            if total_time > 3600:
                h = total_time // 3600
                timestring += f'{int(h)} hours '
                total_time %= 3600
            if total_time > 60:
                m = total_time // 60
                timestring += f'{int(m)} minutes '
                total_time %= 60
            timestring += f'{round(total_time, 3)} seconds '

            self.log(f'--> TSCoDe normal termination: total time {timestring}')

            self.logfile.close()

            #### EXTRA
            
            path = os.path.join(os.getcwd(), vmd_name)
            print('Opening VMD...')
            with suppress_stdout_stderr():
                os.system(f'vmd -e {path}')

            ################################################

        except KeyboardInterrupt:
            print('KeyboardInterrupt requested by user. Quitting.')
            quit()


if __name__ == '__main__':

    import sys

    usage = '\n\tTSCODE correct usage:\n\n\tpython tscode.py input.txt\n\n\tSee documentation for input formatting.\n'

    if len(sys.argv) != 2:

        # filename = 'DA.txt'
        filename = 'input.txt'
        # os.chdir('Resources/bend')
        os.chdir('Resources/test')

        # print(usage)
        # quit()
    
    else:
        if len(sys.argv[1].split('.')) == 1:
            print(usage)
            quit()


        filename = os.path.realpath(sys.argv[1])
        os.chdir(os.path.dirname(filename))

    docker = Docker(filename)
    # initialize docker from input file

    docker.run()
    # run the program

