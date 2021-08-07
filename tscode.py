# coding=utf-8
'''

TSCoDe: Transition State Conformational Docker
Copyright (C) 2021 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Version 0.00 - Pre-release

https://github.com/ntampellini/TSCoDe

Nicolo' Tampellini - nicolo.tampellini@yale.edu

'''

import os
import re
import time
from copy import deepcopy
from itertools import groupby

import numpy as np

from settings import (
                      CALCULATOR,
                      DEFAULT_LEVELS,
                      OPENBABEL_OPT_BOOL,
                      PROCS,
                      )

if OPENBABEL_OPT_BOOL:
    from optimization_methods import openbabel_opt

from operators import operate
from parameters import orb_dim_dict
from embeds import monomolecular_embed, string_embed, cyclical_embed, dihedral_embed
from hypermolecule_class import Hypermolecule, align_structures
from optimization_methods import (
                                  ase_adjust_spacings,
                                  get_nci,
                                  hyperNEB,
                                  mopac_opt,
                                  MopacReadError,
                                  optimize,
                                  prune_enantiomers,
                                  scramble,
                                  xtb_metadyn_augmentation,
                                  )
from utils import (
                   ase_view,
                   cartesian_product,
                   clean_directory,
                   InputError,
                   loadbar,
                   time_to_string,
                   write_xyz,
                   ZeroCandidatesError
                   )


from python_functions import prune_conformers, compenetration_check
# These could in the future be boosted via Cython or Julia, but they
# are the bottleneck of the program only for cyclical trimolecular
# embeds with a lot of conformers, that need to intersect in a 
# precise and specific way. Rare case, indeed.

def calc_positioned_conformers(self):
    self.positioned_conformers = np.array([[self.rotation @ v + self.position for v in conformer] for conformer in self.atomcoords])

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

class Options:

    __keywords__ = [
                    'BYPASS',         # Debug keyword. Used to skip all pruning steps and
                                      # directly output all the embedded geometries.

                    'CALC',           # Manually overrides the calculator in "settings.py"
                    
                    'CHECK',          # Visualize the input molecules through the ASE GUI,
                                      # to check orbital positions or reading faults.
                                       
                    'CLASHES',        # Manually specify the max number of clashes and/or
                                      # the distance threshold at which two atoms are considered
                                      # clashing. The more forgiving, the more structures will reach
                                      # the geometry optimization step. Syntax: `CLASHES(num=3,dist=1.2)`
                    
                    'DEEP',           # Performs a deeper search, retaining more starting points
                                      # for calculations and smaller turning angles. Equivalent to
                                      # `THRESH=0.3 STEPS=12 CLASHES=(num=5,dist=1)`

                    'DEBUG',          # DEBUG KEYWORD
                                      
                    'DIST',           # Manually imposed distance between specified atom pairs,
                                      # in Angstroms. Syntax uses parenthesis and commas:
                                      # `DIST(a=2.345,b=3.67,c=2.1)`

                    'ENANTIOMERS',    # Do not discard enantiomeric structures.

                    'EZPROT',         # Double bond protection

                    'KCAL',           # Trim output structures to a given value of relative energy.
                                      # Syntax: `KCAL=n`, where n can be an integer or float.
                                       
                    'LET',            # Overrides safety checks that prevent the
                                      # program from running too large calculations,
                                      # removes the limit of 5 conformers for cyclical embeds

                    'LEVEL',          # Manually set the MOPAC theory level to be used,
                                      # default is PM7. Syntax: `LEVEL=PM7`
                                       
                    'MMFF',           # Use the Merck Molecular Force Field during the
                                      # OpenBabel pre-optimization (default is UFF).

                    'MTD',            # Run conformational augmentation through metadynamic sampling (XTB)

                    'NCI',            # Estimate and print non-covalent interactions present in the generated poses.

                    'NEB',            # Perform an automatical climbing image nudged elastic band (CI-NEB)
                                      # TS search after the partial optimization step, inferring reagents
                                      # and products for each generated TS pose. These are guessed by
                                      # approaching the reactive atoms until they are at the right distance,
                                      # and then partially constrained (reagents) or free (products) optimizations
                                      # are carried out to get the start and end points for a CI-NEB TS search.
                                      # For trimolecular transition states, only the first imposed pairing (a) 
                                      # is approached - i.e. the C-C reactive distance in the example above.
                                      # This NEB option is only really usable for those reactions in which two
                                      # (or three) molecules are bound together (or strongly interacting) after
                                      # the TS, with no additional species involved. For example, cycloaddition
                                      # reactions are great candidates while atom transfer reactions
                                      # (i.e. epoxidations) are not. Of course this implementation is not
                                      # always reliable, and it is provided more as an experimenting tool
                                      # than a definitive feature.

                    'NEWBONDS',       # Manually specify the maximum number of "new bonds" that a
                                      # TS structure can have to be retained and not to be considered
                                      # scrambled. Default is 1. Syntax: `NEWBONDS=1`

                    'NOOPT',          # Skip the optimization steps, directly writing structures to file.

                    'ONLYREFINED',    # Discard structures that do not successfully refine bonding distances.

                    'PROCS',          # Set the number of parallel cores to be used by ORCA

                    'RIGID',          # Does not apply to "string" embeds. Avoid
                                      # bending structures to better build TSs.

                    'ROTRANGE',       # Does not apply to "string" embeds. Manually specify the rotation
                                      # range to be explored around the structure pivot.
                                      # Default is 120. Syntax: `ROTRANGE=120`

                    'SHRINK',         # Exaggerate orbital dimensions during embed, scaling them by a factor
                                      # of one and a half. This makes it easier to perform the embed without
                                      # having molecules clashing one another. Then, the correct distance between
                                      # reactive atom pairs is achieved as for standard runs by spring constraints
                                      # during MOPAC/ORCA optimization.

                    'STEPS',          # Manually specify the number of steps to be taken in scanning rotations.
                                      # For string embeds, the range to be explored is the full 360°, and the
                                      # default `STEPS=24` will perform 15° turns. For cyclical and chelotropic
                                      # embeds, the rotation range to be explored is +-`ROTRANGE` degrees.
                                      # Therefore, the default value of `ROTRANGE=120 STEPS=12` will perform
                                      # twelve 20 degrees turns.

                    'SUPRAFAC',       # Only retain suprafacial orbital configurations in cyclical TSs.
                                      # Thought for Diels-Alder and other cycloaddition reactions.

                    'THRESH',         # RMSD threshold (Angstroms) for structure pruning. The smaller,
                                      # the more retained structures. Default is 0.5 A.
                                      # Syntax: `THRESH=n`, where n is a number.
                    ]
                    
    # list of keyword names to be used in the first line of program input

    rotation_range = 90
    rotation_steps = None # Set later by the _setup() function, based on embed type
    pruning_thresh = None # Set later by the _setup() function, based on embed type/atom number
    rigid = False
    
    max_clashes = 0
    clash_thresh = 1.3

    max_newbonds = 0

    optimization = True
    calculator = CALCULATOR
    theory_level = None        # set later in _calculator_setup()
    procs = None               # set later in _calculator_setup()
    openbabel_opt = OPENBABEL_OPT_BOOL
    openbabel_level = 'UFF'

    neb = False
    nci = False
    shrink = False
    metadynamics = False
    suprafacial = False
    only_refined = False
    keep_enantiomers = False
    double_bond_protection = False

    fix_angles_in_deformation = False
    # Not possible to set manually through a keyword.
    # Monomolecular embeds have it on to prevent
    # scrambling, but better to leave it off for
    # less severe deformations, since convergence
    # is faster

    kcal_thresh = None
    bypass = False
    debug = False
    let = False
    check_structures = False
    # Default values, updated if _parse_input
    # finds keywords and calls _set_options

    def __repr__(self):
        d = {var:self.__getattribute__(var) for var in dir(self) if var[0:2] != '__'}
        
        do_not_repr = (
            'bypass',
            'check_structures',
            'debug',
            'let',
            'metadynamics',
            'nci',
            'neb',
        )
        
        for name in do_not_repr:
            d.pop(name)

        if self.kcal_thresh is None:
            d.pop('kcal_thresh')

        if not OPENBABEL_OPT_BOOL:
            d.pop('openbabel_level')

        if self.procs == 1 or self.calculator not in ('ORCA', ' GAUSSIAN'):
            d.pop('procs')

        padding = 1 + max([len(var) for var in d])

        return '\n'.join([f'{var}{" "*(padding-len(var))}: {d[var]}' for var in d])

class Docker:
    def __init__(self, filename, stamp=None):
        '''
        Initialize the Docker object by reading the input filename (.txt).
        Sets the Option dataclass properties to default and then updates them
        with the user-requested keywords, if there are any.

        '''
        if stamp is None:
            self.stamp = time.ctime().replace(' ','_').replace(':','-')[4:-8]
            # replaced ctime yields 'Sun_May_23_18-53-47_2021', only keeping 'May_23_18-53'

        else:
            self.stamp = stamp

        try:
            os.remove(f'TSCoDe_{self.stamp}.log')

        except FileNotFoundError:
            pass

        self.logfile = open(f'TSCoDe_{self.stamp}.log', 'a', buffering=1)


        s ='\n*************************************************************\n'
        s += '*      TSCoDe: Transition State Conformational Docker       *\n'
        s += '*************************************************************\n'
        s += '*                 Version 0.00 - Test pre-release           *\n'
        s += "*       Nicolo' Tampellini - nicolo.tampellini@yale.edu     *\n"
        s += '*************************************************************\n'

        self.log(s)

        self.options = Options()

        inp = self._parse_input(filename)
        self.objects = [Hypermolecule(name, c_ids) for name, c_ids in inp]

        self.ids = [len(mol.atomnos) for mol in self.objects]
        # used to divide molecules in TSs

        for i, mol in enumerate(self.objects):
            if mol.hyper:
                for r_atom in mol.reactive_atoms_classes_dict.values():
                    if i == 0:
                        r_atom.cumnum = r_atom.index
                    else:
                        r_atom.cumnum = r_atom.index + sum(self.ids[:i])
        # appending to each reactive atom the cumulative
        # number indexing in the TS context

        self._read_pairings(filename)
        self._set_options(filename)
        self._calculator_setup()

        if self.options.debug:
            for mol in self.objects:
                if mol.hyper:
                    mol.write_hypermolecule()

        if self.options.check_structures:
            self._inspect_structures()

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
            assert len(lines) < 5
            # (optional) keyword line + 1, 2 or 3 lines for molecules

            keywords = [l.split('=')[0] if not '(' in l else l.split('(')[0] for l in lines[0].split()]
            if any(k in self.options.__keywords__ for k in keywords):
                lines = lines[1:]

            inp = []
            for line in lines:
                filename, *reactive_atoms = line.split()

                if len(reactive_atoms) > 4:
                    s = f'Too many reactive atoms specified for {filename} ({len(reactive_atoms)}).'
                    raise SyntaxError(s)

                reactive_indexes = tuple([int(re.sub('[^0-9]', '', i)) for i in reactive_atoms])

                if '>' in filename:
                    filename = operate(filename,
                                        self.options.calculator,
                                        self.options.theory_level,
                                        procs=self.options.procs,
                                        logfunction=self.log)

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
                    for k in keywords:
                        if k not in self.options.__keywords__:
                            raise SyntaxError(f'Keyword {k} was not understood. Please check your syntax.')

                keywords_list = [word.upper() for word in lines[0].split()]

                if 'SUPRAFAC' in keywords_list:
                    self.options.suprafacial = True

                if 'DEEP' in keywords_list:
                    self.options.pruning_thresh = 0.3
                    self.options.rotation_steps = 24
                    self.options.max_clashes = 3
                    self.options.clash_thresh = 1.2

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
                    self.options.theory_level = kw.split('=')[1].upper().replace('_', ' ')

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

                if 'MMFF' in keywords_list:
                    self.options.openbabel_level = 'MMFF'

                if 'KCAL' in [k.split('=')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('KCAL')]
                    self.options.kcal_thresh = float(kw.split('=')[1])

                if 'SHRINK' in keywords_list:
                    self.options.shrink = True
                    self.options.shrink_multiplier = 1.5

                elif 'SHRINK' in [k.split('=')[0] for k in keywords_list]:
                    self.options.shrink = True
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('SHRINK')]
                    self.options.shrink_multiplier = float(kw.split('=')[1])

                if 'ENANTIOMERS' in keywords_list:
                    self.options.keep_enantiomers = True

                if 'DEBUG' in keywords_list:
                    self.options.debug = True

                if 'PROCS' in [k.split('=')[0] for k in keywords_list]:
                    
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('PROCS')]
                    self.options.procs = int(kw.split('=')[1])

                if 'EZPROT' in keywords_list:
                    self.options.double_bond_protection = True

                if 'CALC' in [k.split('=')[0] for k in keywords_list]:
                    kw = keywords_list[[k.split('=')[0] for k in keywords_list].index('CALC')]
                    self.options.calculator = kw.split('=')[1]

                if 'MTD' in keywords_list:
                    if self.options.calculator != 'XTB':
                        raise InputError(('Metadynamics augmentation can only be run with the XTB calculator.\n'
                                          'Change it in settings.py or use the CALC=XTB keyword.'))
                    self.options.metadynamics = True

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
                            raise SyntaxError('The only letters allowed for pairings are "a", "b" and "c".')

                        pairings.append([int(index), letters[0]])

                    else:
                        for l in letters:

                            if l not in ('a', 'b', 'c'):
                                raise SyntaxError('The only letters allowed for pairings are "a", "b" and "c".')

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
            else:
                if unlabeled != []:
                    for z in unlabeled:
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

        letters = tuple(self.pairings_table.keys())

        if len(letters) == 1 and tuple(sorted(letters)) != ('a',):
            raise SyntaxError('The pairing letters specified are invalid. To only specify one label, use letter \'a\'.')

        if len(letters) == 2 and tuple(sorted(letters)) != ('a', 'b'):
            raise SyntaxError('The pairing letters specified are invalid. To specify two labels, use letters \'a\' and \'b\'.')

        if len(letters) == 3 and tuple(sorted(letters)) != ('a', 'b', 'c'):
            raise SyntaxError('The pairing letters specified are invalid. To only three labels, use letters \'a\', \'b\' and \'c\'.')


        for letter, ids in self.pairings_table.items():

            if len(ids) == 1:
                raise SyntaxError(f'Letter \'{letter}\' is only specified once. Please flag the second reactive atom.')

            if len(ids) > 2:
                raise SyntaxError(f'Letter \'{letter}\' is specified more than two times. Please remove the unwanted letters.')

        if self.pairings == []:
            if all([len(mol.reactive_indexes) == 2 for mol in self.objects]):
                self.log('--> No atom pairings imposed. Computing all possible dispositions.\n')
                # if there is multiple regioisomers to be computed
        else:
            self.log(f'--> Atom pairings imposed are {len(self.pairings)}: {self.pairings} (Cumulative index numbering)\n')
        
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

        if mol.vicinal:
            pivots_lengths = [np.linalg.norm(pivot.pivot) for pivot in mol.pivots]
            shortest_length = min(pivots_lengths)
            mask = np.array([i - shortest_length < 1e-5 for i in pivots_lengths])
            mol.pivots = mol.pivots[mask]
        # if mol is vicinal (two connected reactive centers, reacting with sigma bond)
        # then remove all pivots that are not the shortest. This ensures a sort of 
        # "suprafaciality" to the pivots used, preventing the embed of structures that
        # would surely compenetrate

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

        if len(self.objects) == 1:

            if len(self.objects[0].reactive_indexes) == 4:
                self.embed = 'dihedral'
                if self.options.kcal_thresh is None:
                # set to 5 if user did not specify a value
                    self.options.kcal_thresh = 5

                if self.options.pruning_thresh is None:
                # set to 0.2 if user did not specify a value
                    self.options.pruning_thresh = 0.2

                return

            self.embed = 'monomolecular'
            self._set_pivots(self.objects[0])

            self.options.only_refined = True
            self.options.fix_angles_in_deformation = True
            # These are required: otherwise, extreme bending could scramble molecules
            
            self.candidates = int(len(self.objects[0].atomcoords))
            self.candidates *= len(self.objects[0].pivots)

        elif len(self.objects) in (2,3):
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
                    # be generated a tad too close to each other for how the cyclical embed works          

                self.options.rotation_steps = 9

                if hasattr(self.options, 'custom_rotation_steps'):
                # if user specified a custom value, use it.
                    self.options.rotation_steps = self.options.custom_rotation_steps

                self.systematic_angles = cartesian_product(*[range(self.options.rotation_steps+1) for _ in self.objects]) \
                                         * 2*self.options.rotation_range/self.options.rotation_steps - self.options.rotation_range

                self.candidates = len(self.systematic_angles)*np.prod([len(mol.atomcoords) for mol in self.objects])
                
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
                raise InputError(('Bad input - The only molecular configurations accepted are:\n' 
                                  '1) One molecule with two reactive centers (monomolecular embed)\n'
                                  '2) One molecule with four indexes (dihedral embed)\n'
                                  '3) Two or three molecules with two reactive centers each (cyclical embed)\n'
                                  '4) Two molecules with one reactive center each (string embed)\n'
                                  '5) Two molecules, one with a single reactive center and the other with two (chelotropic embed)'))
        else:
            raise InputError('Bad input - too many molecules specified (three at most).')

        if self.options.shrink:
            for molecule in self.objects:
                molecule._scale_orbs(self.options.shrink_multiplier)
                self._set_pivots(molecule)
            self.options.only_refined = True
        # SHRINK - scale orbitals and rebuild pivots

        if self.options.pruning_thresh is None:
            self.options.pruning_thresh = 1

            if sum(self.ids) < 50:
                self.options.pruning_thresh = 0.5
            # small molecules need smaller RMSD threshold

        self.log(f'--> Setup performed correctly. {self.candidates} candidates will be generated.\n')

    def _calculator_setup(self):
        '''
        Set up the calculator to be used with default theory levels.
        '''
        # Checking that calculator is specified correctly
        if self.options.calculator not in ('MOPAC', 'ORCA', 'GAUSSIAN','XTB'):
            raise SyntaxError(f'\'{self.options.calculator}\' is not a valid calculator. Change its value from the parameters.py file or with the CALC keyword.')

        # Setting default theory level if user did not specify it
        if self.options.theory_level is None:
            self.options.theory_level = DEFAULT_LEVELS[CALCULATOR]

        # Setting up procs number from settings if user did not specify another value
        if self.options.procs is None:
            self.options.procs = PROCS

            if self.options.theory_level in ('MNDO','AM1','PM3','HF-3c','HF MINIX D3BJ GCP(HF/MINIX) PATOM') and self.options.PROCS != 1:
                raise Exception(('ORCA does not support parallelization for Semiempirical Methods. '
                                 'Please change the value of PROCS to 1 in parameters.py or '
                                 'change the theory level.'))

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
            if swaps[n][i]:
                return list(reversed(ids))
            return ids

        if len(self.objects) == 2:

            swaps = [(0,0),
                     (0,1)]

            oriented = [orient(i,ids,n) for i, ids in enumerate(cumulative_pivots_ids)]
            couples = [[oriented[0][0], oriented[1][0]], [oriented[0][1], oriented[1][1]]]

            return couples

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

    def _set_target_distances(self):
        '''
        Called before TS refinement to compute all
        target bonding distances.

        (This function could be written better, I know. But it works.)
        '''
        self.target_distances = {}

        r_atoms = np.array([list(mol.reactive_atoms_classes_dict.values()) for mol in self.objects]).ravel()
        pairings = self.constrained_indexes.ravel()
        pairings = pairings.reshape(int(pairings.shape[0]/2), 2)
        pairings = {tuple(sorted((a,b))) for a, b in pairings}

        for index1, index2 in pairings:

            if [index1, index2] in self.pairings_table.values() and hasattr(self, 'pairings_dists'):
                letter = list(self.pairings_table.keys())[list(self.pairings_table.values()).index([index1, index2])]
                if letter in [l for l, _ in self.pairings_dists]:
                    target_dist = self.pairings_dists[[l for l, _ in self.pairings_dists].index(letter)][1]
                    self.target_distances[(index1, index2)] = target_dist
                    continue
            # if target distance has been specified by user, read that, otherwise compute it

            for r_atom in r_atoms:

                if index1 == r_atom.cumnum:
                    r_atom1 = r_atom

                if index2 == r_atom.cumnum:
                    r_atom2 = r_atom

            try:
                dist1 = orb_dim_dict[r_atom1.symbol + ' ' + str(r_atom1)]
            except KeyError:
                dist1 = orb_dim_dict['Fallback']

            try:
                dist2 = orb_dim_dict[r_atom2.symbol + ' ' + str(r_atom2)]
            except KeyError:
                dist2 = orb_dim_dict['Fallback']

            target_dist = dist1 + dist2

            self.target_distances[(index1, index2)] = target_dist

    def _inspect_structures(self):
        '''
        '''

        self.log('--> Structures check requested. Shutting down after last window is closed.\n')

        for mol in self.objects:
            ase_view(mol)
        
        self.logfile.close()
        os.remove(f'TSCoDe_{self.stamp}.log')

        quit()

    def apply_mask(self, attributes, mask):
        '''
        Applies in-place masking of Docker attributes
        '''
        for attr in attributes:
            if hasattr(self, attr):
                new_attr = getattr(self, attr)[mask]
                setattr(self, attr, new_attr)

    def scramble(self, array, sequence):
        return np.array([array[s] for s in sequence])

    ######################################################################################################### RUN METHODS

    def zero_candidates_check(self):
        '''
        Asserts that not all structures are being rejected.
        '''
        if len(self.structures) == 0:
            raise ZeroCandidatesError()

    def generate_candidates(self):
        '''
        Generate a series of candidate structures by the proper embed algorithm.
        '''

        if self.embed == 'dihedral':
            dihedral_embed(self)
            self.atomnos = self.objects[0].atomnos

            if len(self.structures) == 0:
                s = ('\n--> Dihedral embed did not find any suitable maxima above the set threshold\n'
                    f'    ({self.options.kcal_thresh} kcal/mol) during the scan procedure. Observe the\n'
                     '    generated energy plot and try lowering the threshold value (KCAL keyword).')
                self.log(s)
                raise ZeroCandidatesError()


        elif self.embed == 'monomolecular':
            monomolecular_embed(self)
            self.atomnos = self.objects[0].atomnos
            self.energies = np.zeros(len(self.structures))
            self.exit_status = np.zeros(len(self.structures), dtype=bool)
            self.graphs = [self.objects[0].graph]

            self.constrained_indexes = np.array([[self.objects[0].reactive_indexes] for _ in range(self.candidates)])

        else:

            if self.embed in ('cyclical', 'chelotropic'):
                embedded_structures = cyclical_embed(self)

                if embedded_structures == []:
                    s = ('\n--> Cyclical embed did not find any suitable disposition of molecules.\n' +
                           '    This is probably because one molecule has two reactive centers at a great distance,\n' +
                           '    preventing the other two molecules from forming a closed, cyclical structure.')
                    self.log(s, p=False)
                    raise ZeroCandidatesError(s)

            else: # self.embed == 'string'
                embedded_structures = string_embed(self)


            clean_directory() # removing temporary files from ase_bend

            self.atomnos = np.concatenate([molecule.atomnos for molecule in self.objects])
            # cumulative list of atomic numbers associated with coordinates

            conf_number = [len(molecule.atomcoords) for molecule in self.objects]
            conf_indexes = cartesian_product(*[np.array(range(i)) for i in conf_number])
            # first index of each vector is the conformer number of the first molecule and so on

            self.structures = np.zeros((int(len(conf_indexes)*int(len(embedded_structures))), len(self.atomnos), 3)) # like atomcoords property, but containing multimolecular arrangements
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
        self.log(f'Generated {len(self.structures)} transition state candidates ({time_to_string(t_end-self.t_start_run)})\n')

    def compenetration_refining(self):
        '''
        Performing a sanity check for excessive compenetration
        on generated structures, discarding the ones that look too bad.
        '''
        self.log('--> Checking structures for compenetrations')

        self.graphs = [mol.graph for mol in self.objects]

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

        if False in mask:
            self.log(f'Discarded {len([b for b in mask if not b])} candidates for compenetration ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
        self.log()

        self.zero_candidates_check()

        self.energies = np.zeros(len(self.structures))
        self.exit_status = np.zeros(len(self.structures), dtype=bool)

    def similarity_refining(self, verbose=False):
        '''
        Remove structures that are too similar to each other (RMSD-based).
        If not self.optimization.keep_enantiomers, removes duplicates
        (principal moments of inertia-based).
        '''

        if verbose:
            self.log('--> Similarity Processing')

        t_start = time.time()

        before = len(self.structures)
        attr = ('constrained_indexes', 'energies', 'exit_status')

        for k in (5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2):
            if 5*k < len(self.structures):
                t_start_int = time.time()
                self.structures, mask = prune_conformers(self.structures, self.atomnos, max_rmsd=self.options.pruning_thresh, k=k)

                self.apply_mask(attr, mask)
                t_end_int = time.time()

                if verbose:
                    self.log(f'    - similarity pre-processing   (k={k}) - {time_to_string(t_end_int-t_start_int)} - kept {len([b for b in mask if b])}/{len(mask)}')
        
        t_start_int = time.time()
        self.structures, mask = prune_conformers(self.structures, self.atomnos, max_rmsd=self.options.pruning_thresh)
        t_end = time.time()

        if verbose:
            self.log(f'    - similarity final processing (k=1) - {time_to_string(t_end-t_start_int)} - kept {len([b for b in mask if b])}/{len(mask)}')

        self.apply_mask(attr, mask)

        if False in mask:
            self.log(f'Discarded {int(before - len([b for b in mask if b]))} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')

        if not self.options.keep_enantiomers:

            t_start = time.time()
            self.structures, mask = prune_enantiomers(self.structures, self.atomnos)
            
            self.apply_mask(attr, mask)

            t_end = time.time()
            
            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} enantiomeric structures ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
            self.log()

    def force_field_refining(self):
        '''
        Performs structural optimizations with the UFF or MMFF force field,
        through the OpenBabel package. Only structures that do not scramble during
        MM optimization are updated, while others are kept as they are.
        '''

        ################################################# CHECKPOINT BEFORE MM OPTIMIZATION

        self.outname = f'TSCoDe_checkpoint_{self.stamp}.xyz'
        with open(self.outname, 'w') as f:        
            for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):
                write_xyz(structure, self.atomnos, f, title=f'TS candidate {i+1} - Checkpoint before MM optimization')
        self.log(f'--> Checkpoint output - Wrote {len(self.structures)} TS structures to {self.outname} file before MM optimization.\n')

        ################################################# GEOMETRY OPTIMIZATION - FORCE FIELD

        self.log(f'--> Structure optimization ({self.options.openbabel_level} level)')
        t_start = time.time()

        for i, structure in enumerate(deepcopy(self.structures)):
            loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')
            try:
                new_structure, self.exit_status[i] = openbabel_opt(structure, self.atomnos, self.constrained_indexes[i], self.graphs, method=self.options.openbabel_level)

                if self.exit_status[i]:
                    self.structures[i] = new_structure

            except Exception as e:
                raise e

        loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')
        t_end = time.time()
        self.log(f'Openbabel {self.options.openbabel_level} optimization took {time_to_string(t_end-t_start)} (~{time_to_string((t_end-t_start)/len(self.structures))} per structure)')
        
        ################################################# EXIT STATUS

        self.log(f'Successfully pre-refined {len([b for b in self.exit_status if b])}/{len(self.structures)} candidates at UFF level.')
        
        ################################################# PRUNING: SIMILARITY (POST FORCE FIELD OPT)

        self.zero_candidates_check()
        self.similarity_refining()

        ################################################# CHECKPOINT BEFORE MOPAC/ORCA OPTIMIZATION

        with open(self.outname, 'w') as f:        
            for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):
                exit_str = f'{self.options.openbabel_level} REFINED' if self.exit_status[i] else 'RAW'
                write_xyz(structure, self.atomnos, f, title=f'TS candidate {i+1} - {exit_str} - Checkpoint before {self.options.calculator} optimization')
        self.log(f'--> Checkpoint output - Updated {len(self.structures)} TS structures to {self.outname} file before {self.options.calculator} optimization.\n')
                        
    def optimization_refining(self):
        '''
        Refines structures by constrained optimizations with the MOPAC calculator,
        discarding similar ones and scrambled ones.
        '''

        t_start = time.time()

        self.log(f'--> Structure optimization ({self.options.theory_level} level)')

        if self.options.calculator == 'MOPAC':
            method = f'{self.options.theory_level} GEO-OK CYCLES=500'

        else:
            method = f'{self.options.theory_level}'

        for i, structure in enumerate(deepcopy(self.structures)):
            loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')
            try:
                t_start_opt = time.time()
                new_structure, self.energies[i], self.exit_status[i] = optimize(self.options.calculator,
                                                                                structure,
                                                                                self.atomnos,
                                                                                self.graphs,
                                                                                self.constrained_indexes[i],
                                                                                method=method,
                                                                                procs=self.options.procs,
                                                                                max_newbonds=self.options.max_newbonds)

                if self.exit_status[i]:
                    self.structures[i] = new_structure

                exit_str = 'CONVERGED' if self.exit_status[i] else 'SCRAMBLED'

            except MopacReadError:
                # ase will throw a ValueError if the output lacks a space in the "FINAL POINTS AND DERIVATIVES" table.
                # This occurs when one or more of them is not defined, that is when the calculation did not end well.
                # The easiest solution is to reject the structure and go on.
                self.energies[i] = np.inf
                self.exit_status[i] = False
                exit_str = 'FAILED TO READ FILE'

            except Exception as e:
                raise e

            self.log((f'    - {self.options.calculator} {self.options.theory_level} optimization: Structure {i+1} {exit_str} - '
                      f'took {time_to_string(time.time()-t_start_opt)}'), p=False)

        loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')
        
        self.log(f'Successfully optimized {len([b for b in self.exit_status if b])}/{len(self.structures)} structures. Non-optimized ones will not be discarded.')

        self.log((f'{self.options.calculator} {self.options.theory_level} optimization took '
                  f'{time_to_string(time.time()-t_start)} (~{time_to_string((time.time()-t_start)/len(self.structures))} per structure)'))

        ################################################# PRUNING: SIMILARITY (POST SEMIEMPIRICAL OPT)

        self.zero_candidates_check()
        self.similarity_refining()

        ################################################# REFINING: BONDING DISTANCES

        self.write_structures('TS_guesses_unrefined', energies=False, p=False)
        self.log(f'--> Checkpoint output - Updated {len(self.structures)} TS structures before distance refinement.\n')

        self.log(f'--> Refining bonding distances for TSs ({self.options.theory_level} level)')

        if self.options.openbabel_opt:
            try:
                os.remove(f'TSCoDe_checkpoint_{self.stamp}.xyz')
                # We don't need the pre-optimized structures anymore
            except FileNotFoundError:
                pass

        self._set_target_distances()

        for i, structure in enumerate(deepcopy(self.structures)):
            loadbar(i, len(self.structures), prefix=f'Refining structure {i+1}/{len(self.structures)} ')
            try:

                traj = f'refine_{i}.traj' if self.options.debug else None

                new_structure, new_energy, self.exit_status[i] = ase_adjust_spacings(self,
                                                                                        structure,
                                                                                        self.atomnos,
                                                                                        self.constrained_indexes[i],
                                                                                        self.graphs,
                                                                                        title=i,
                                                                                        max_newbonds=self.options.max_newbonds,
                                                                                        traj=traj
                                                                                        )

                if self.exit_status[i]:
                    self.structures[i] = new_structure
                    self.energies[i] = new_energy
                                                                                                                        
            except ValueError as e:
                # ase will throw a ValueError if the output lacks a space in the "FINAL POINTS AND DERIVATIVES" table.
                # This occurs when one or more of them is not defined, that is when the calculation did not end well.
                # The easiest solution is to reject the structure and go on. TODO-check
                self.log(e)
                self.log(f'Failed to read MOPAC file for Structure {i+1}, skipping distance refinement', p=False)                                    
        
        loadbar(1, 1, prefix=f'Refining structure {i+1}/{len(self.structures)} ')
        t_end = time.time()
        self.log(f'{self.options.calculator} {self.options.theory_level} refinement took {time_to_string(t_end-t_start)} (~{time_to_string((t_end-t_start)/len(self.structures))} per structure)')

        before = len(self.structures)
        if self.options.only_refined:
            mask = self.exit_status
            self.structures = self.structures[mask]
            self.energies = self.energies[mask]
            self.exit_status = self.exit_status[mask]
            s = f'Discarded {len([i for i in mask if not i])} unrefined structures'

        else:
            s = 'Non-refined ones will not be discarded.'


        self.log(f'Successfully refined {len([i for i in self.exit_status if i])}/{before} structures. {s}')

        ################################################# PRUNING: SIMILARITY (POST REFINEMENT)

        self.zero_candidates_check()
        self.similarity_refining()

        ################################################# PRUNING: ENERGY (POST REFINEMENT)

        self.energies = self.energies - np.min(self.energies)
        _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
        self.energies = self.scramble(self.energies, sequence)
        self.structures = self.scramble(self.structures, sequence)
        self.constrained_indexes = self.scramble(self.constrained_indexes, sequence)
        # sorting structures based on energy

        if self.options.kcal_thresh is not None:
    
            mask = (self.energies - np.min(self.energies)) < self.options.kcal_thresh
            self.structures = self.structures[mask]
            self.energies = self.energies[mask]
            self.exit_status = self.exit_status[mask]

            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for energy (Threshold set to {self.options.kcal_thresh} kcal/mol)')


        ################################################# XYZ GUESSES OUTPUT 

        self.outname = f'TSCoDe_TS_guesses_{self.stamp}.xyz'
        with open(self.outname, 'w') as f:        
            for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):

                kind = 'REFINED - ' if self.exit_status[i] else 'NOT REFINED - '

                write_xyz(structure, self.atomnos, f, title=f'Structure {i+1} - {kind}Rel. E. = {round(self.energies[i], 3)} kcal/mol')

        os.remove(f'TSCoDe_TS_guesses_unrefined_{self.stamp}.xyz')
        # since we have the refined structures, we can get rid of the unrefined ones

        self.log(f'Wrote {len(self.structures)} rough TS structures to {self.outname} file.\n')

    def metadynamics_augmentation(self):
        '''
        Runs a metadynamics simulation (MTD) through
        the XTB program for each structure in self.structure.
        New structures are obtained from the simulations, minimized
        in energy and added to self. structures.
        '''

        self.log(f'--> Performing XTB Metadynamic Augmentation of TS candidates')

        before = len(self.structures)
        t_start_run = time.time()

        for s, (structure, constrained_indexes) in enumerate(zip(deepcopy(self.structures), deepcopy(self.constrained_indexes))):

            loadbar(s+1, before, f'Running MTD {s+1}/{before} ')
            t_start = time.time()

            new_structures = xtb_metadyn_augmentation(structure,
                                                      self.atomnos,
                                                      constrained_indexes=constrained_indexes,
                                                      new_structures=5,
                                                      title=s)

            self.structures = np.concatenate((self.structures, new_structures))
            self.energies = np.concatenate((self.energies, [0 for _ in range(len(new_structures))]))
            self.constrained_indexes = np.concatenate((self.constrained_indexes, [constrained_indexes for _ in range(len(new_structures))]))
        
            self.log(f'   - Structure {s+1} - {len(new_structures)} new conformers ({time_to_string(time.time()-t_start)})', p=False)

        self.exit_status = np.array([True for _ in range(len(self.structures))], dtype=bool)

        self.log(f'Metadynamics Augmentation completed - found {len(self.structures)-before} new conformers ({time_to_string(time.time()-t_start_run)})\n')

    def hyperneb(self):
        '''
        Performs a clibing-image NEB calculation inferring reagents and products for each structure.
        '''
        self.log(f'--> HyperNEB optimization ({self.options.theory_level} level)')
        t_start = time.time()

        for i, structure in enumerate(self.structures):

            loadbar(i, len(self.structures), prefix=f'Performing NEB {i+1}/{len(self.structures)} ')

            t_start_opt = time.time()

            try:

                self.structures[i], self.energies[i] = hyperNEB(self,
                                                                structure,
                                                                self.atomnos,
                                                                self.ids,
                                                                self.constrained_indexes[i],
                                                                title=f'structure_{i+1}')

                exit_str = 'COMPLETED'
                self.exit_status[i] = True

            except (MopacReadError, ValueError):
                # Both are thrown if a MOPAC file read fails, but the former occurs when an internal (TSCoDe)
                # read fails (getting reagent or product), the latter when an ASE read fails (during NEB)
                exit_str = 'CRASHED'
                self.exit_status[i] = False

            t_end_opt = time.time()

            self.log(f'    - {self.options.calculator} {self.options.theory_level} NEB optimization: Structure {i+1} - {exit_str} - ({time_to_string(t_end_opt-t_start_opt)})', p=False)

        loadbar(1, 1, prefix=f'Performing NEB {len(self.structures)}/{len(self.structures)} ')
        t_end = time.time()
        self.log(f'{self.options.calculator} {self.options.theory_level} NEB optimization took {time_to_string(t_end-t_start)} ({time_to_string((t_end-t_start)/len(self.structures))} per structure)')
        self.log(f'NEB converged for {len([i for i in self.exit_status if i])}/{len(self.structures)} structures\n')

        mask = self.exit_status
        self.structures = self.structures[mask]
        self.energies = self.energies[mask]
        self.exit_status = self.exit_status[mask]

        ################################################# PRUNING: SIMILARITY (POST NEB)

        if len(self.structures) != 0:

            t_start = time.time()
            self.structures, mask = prune_conformers(self.structures, self.atomnos, max_rmsd=self.options.pruning_thresh)
            self.energies = self.energies[mask]
            t_end = time.time()
            
            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
            self.log()

        ################################################# TS CHECK - FREQUENCY CALCULATION


            self.log(f'--> TS frequency calculation ({self.options.theory_level} level)')
            t_start = time.time()

            for i, structure in enumerate(self.structures):

                loadbar(i, len(self.structures), prefix=f'Performing frequency calculation {i+1}/{len(self.structures)} ')

                mopac_opt(structure,
                          self.atomnos,
                          method=f'{self.options.theory_level} FORCE',
                          title=f'TS_{i+1}_FREQ',
                          read_output=False)

            loadbar(1, 1, prefix=f'Performing frequency calculation {i+1}/{len(self.structures)} ')
            t_end = time.time()
            self.log(f'Mopac {self.options.theory_level} frequency calculation took {time_to_string(t_end-t_start)} (~{time_to_string((t_end-t_start)/len(self.structures))} per structure)\n')

        ################################################# NEB XYZ OUTPUT

            self.energies -= np.min(self.energies)
            _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
            self.energies = scramble(self.energies, sequence)
            self.structures = scramble(self.structures, sequence)
            self.constrained_indexes = scramble(self.constrained_indexes, sequence)
            # sorting structures based on energy

            self.outname = f'TSCoDe_NEB_TSs_{self.stamp}.xyz'
            with open(self.outname, 'w') as f:        
                for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):
                    write_xyz(structure, self.atomnos, f, title=f'Structure {i+1} - TS - Rel. E. = {round(self.energies[i], 3)} kcal/mol')

            self.log(f'Wrote {len(self.structures)} final TS structures to {self.outname} file\n')

    def print_nci(self):
        '''
        Prints the non-covalent interactions guesses for final structures.
        '''
        self.log('--> Non-covalent interactions spotting')
        self.nci = []

        for i, structure in enumerate(self.structures):

            nci, print_list = get_nci(structure, self.atomnos, self.constrained_indexes[i], self.ids)
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

    def print_header(self):
        '''
        Writes information about the TSCoDe parameters used in the calculation.
        '''

        if self.embed != 'dihedral':

            head = ''
            for i, mol in enumerate(self.objects):
                descs = [atom.symbol+'('+str(atom)+f', {round(np.linalg.norm(atom.center[0]-atom.coord), 3)} A, {len(atom.center)} centers)' for atom in mol.reactive_atoms_classes_dict.values()]
                t = '\n        '.join([(str(index) + ' ' if len(str(index)) == 1 else str(index)) + ' -> ' + desc for index, desc in zip(mol.reactive_indexes, descs)])
               
                pivot_line = ''
                if hasattr(mol, 'pivots'):
                    pivot_line += f' -> {len(mol.pivots)} pivots'

                    if mol.vicinal:
                        pivot_line += ', vicinal'

                    if mol.sigmatropic:
                        pivot_line += ', sigmatropic'

                head += f'\n    {i+1}. {mol.name}{pivot_line}\n        {t}\n'

            self.log('--> Input structures & reactive indexes data:\n' + head)

        self.log(f'--> Calculation options used were:')
        for line in str(self.options).split('\n'):

            if self.embed in ('monomolecular','string') and line.split()[0] in ('rotation_range', 'rigid', 'suprafacial'):
                continue

            if self.embed == 'dihedral' and line.split()[0] not in ('optimization', 'calculator','theory_level', 'kcal_thresh', 'debug', 'pruning_thresh'):
                continue
            
            self.log(f'    - {line}')

    def write_vmd(self, indexes=None):
        '''
        Write VMD file with bonds and reactive atoms highlighted.
        '''

        if indexes is None:
            indexes = self.constrained_indexes[0].ravel()

        self.vmd_name = self.outname.split('.')[0] + '.vmd'
        path = os.path.join(os.getcwd(), self.vmd_name)
        with open(path, 'w') as f:
            s = ('display resetview\n' +
                'mol new {%s.xyz}\n' % (path.strip('.vmd')) +
                'mol selection index %s\n' % (' '.join([str(i) for i in indexes])) +
                'mol representation CPK 0.7 0.5 50 50\n' +
                'mol color ColorID 7\n' +
                'mol material Transparent\n' +
                'mol addrep top\n')

            for a, b in self.pairings:
                s += f'label add Bonds 0/{a} 0/{b}\n'

            f.write(s)

        self.log(f'Wrote VMD {self.vmd_name} file\n')

    def write_structures(self, tag, indexes=None, energies=True, p=True):
        '''
        '''

        if indexes is None:
            indexes = self.constrained_indexes[0]

        if energies:
            rel_e = self.energies - np.min(self.energies)

        self.outname = f'TSCoDe_{tag}_{self.stamp}.xyz'
        with open(self.outname, 'w') as f:        
            for i, structure in enumerate(align_structures(self.structures, indexes)):
                title = f'TS candidate {i+1} - {tag}'

                if energies:
                    title += f' - Rel. E. = {round(rel_e[i], 3)} kcal/mol'

                write_xyz(structure, self.atomnos, f, title=title)

        if p:
            self.log(f'Wrote {len(self.structures)} {tag} TS structures to {self.outname} file.\n')

    def run(self):
        '''
        Run the TSCoDe program.
        '''

        if not self.options.let and hasattr(self, 'candidates'):
            assert self.candidates < 1e8, ('ATTENTION! This calculation is probably going to be very big. To ignore this message'
                                           ' and proceed, add the LET keyword to the input file.')

        self.print_header()
        self.t_start_run = time.time()

        try: # except KeyboardInterrupt

            try: # except ZeroCandidatesError()

                self.generate_candidates()

                if self.embed == 'dihedral':
                    self.similarity_refining()
                    self.write_structures('TS_guesses', indexes=self.objects[0].reactive_indexes)

                else:

                    if not self.options.bypass:

                        if not self.embed == 'monomolecular':
                            self.compenetration_refining()

                        self.similarity_refining(verbose=True)

                    if self.options.optimization:

                        if self.options.openbabel_opt:
                            self.force_field_refining()

                        self.optimization_refining()

                    else:
                        self.write_structures('unoptimized', energies=False)


            except ZeroCandidatesError:
                t_end_run = time.time()
                s = ('    Sorry, the program did not find any reasonable TS structure. Are you sure the input indexes and pairings were correct? If so, try these tips:\n'
                     '    - If no structure passes the compenetration check, the SHRINK keyword may help (see documentation).\n'
                     '    - Similarly, enlarging the spacing between atom pairs with the DIST keyword facilitates the embed.\n'
                     '    - Impose less strict compenetration rejection criteria with the CLASHES keyword.\n'
                     '    - Generate more structures with higher STEPS and ROTRANGE values.\n'
                )

                self.log(f'\n--> Program termination: No candidates found - Total time {time_to_string(t_end_run-self.t_start_run)}')
                self.log(s)
                clean_directory()
                quit()

            ##################### AUGMENTATION - METADYNAMICS

            if self.options.metadynamics:

                self.metadynamics_augmentation()
                self.optimization_refining()
                self.similarity_refining()

            ##################### POST TSCODE - NEB, NCI, VMD

            if not self.options.bypass:
                indexes = self.objects[0].reactive_indexes if self.embed == 'dihedral' else None
                self.write_vmd(indexes=indexes)

            if self.embed != 'dihedral':

                if self.options.neb:
                    self.hyperneb()
                    
                if self.options.nci and self.options.optimization:
                    self.print_nci()
            
            clean_directory()
            t_end_run = time.time()

            self.log(f'--> TSCoDe normal termination: total time {time_to_string(t_end_run - self.t_start_run, verbose=True)}.')
            self.logfile.close()

            #### EXTRA
            
            # if self.options.debug:
            #     path = os.path.join(os.getcwd(), self.vmd_name)
            #     check_call(f'vmd -e {path}'.split())

            ################################################

        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt requested by user. Quitting.')
            quit()

if __name__ == '__main__':

    import sys

    if len(sys.argv) < 2 or len(sys.argv[1].split('.')) == 1:

        print('\n\tTSCoDe correct usage:\n\n\tpython tscode.py input.txt\n\n\tSee documentation for input formatting.\n')
        quit()

    filename = os.path.realpath(sys.argv[1])
    
    os.chdir(os.path.dirname(filename))

    if len(sys.argv) > 2:
        stamp = sys.argv[2]

    else:
        stamp = None

    docker = Docker(filename, stamp)
    # initialize docker from input file

    docker.run()
    # run the program