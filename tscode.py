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
from subprocess import run

from parameters import OPENBABEL_OPT_BOOL, orb_dim_dict
from embeds import string_embed, cyclical_embed
from hypermolecule_class import Hypermolecule, align_structures
from optimization_methods import (
                                  ase_adjust_spacings,
                                  get_nci,
                                  hyperNEB,
                                  mopac_opt,
                                  MopacReadError,
                                  openbabel_opt,
                                  optimize,
                                  suppress_stdout_stderr,
                                  write_xyz
                                  )
from utils import (
                   ase_view,
                   cartesian_product,
                   clean_directory,
                   InputError,
                   loadbar,
                   time_to_string,
                   ZeroCandidatesError
                   )


# try: 
#     from compenetration import compenetration_check
# except ImportError:
#     from _fallback import compenetration_check

# try:
#     from prune import prune_conformers
# except ImportError:
#     from _fallback import prune_conformers
# If cython libraries are not present, load pure python ones.

# TODO - eventually I could re-write cython libraries, but for now pure python seem fast enough

from _fallback import prune_conformers, compenetration_check


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
                    
                    'CHECK',          # Visualize the input molecules through the ASE GUI,
                                      # to check orbital positions or reading faults.
                                       
                    'CLASHES',        # Manually specify the max number of clashes and/or
                                      # the distance threshold at which two atoms are considered
                                      # clashing. The more forgiving, the more structures will reach
                                      # the geometry optimization step. Syntax: `CLASHES(num=3,dist=1.2)`
                    
                    'DEEP',           # Performs a deeper search, retaining more starting points
                                      # for calculations and smaller turning angles. Equivalent to
                                      # `THRESH=0.3 STEPS=12 CLASHES=(num=5,dist=1)`
                                      
                    'DIST',           # Manually imposed distance between specified atom pairs,
                                      # in Angstroms. Syntax uses parenthesis and commas:
                                      # `DIST(a=2.345,b=3.67,c=2.1)`

                    'KCAL',           # Trim output structures to a given value of relative energy.
                                      # Syntax: `KCAL=n`, where n can be an integer or float.
                                       
                    'LET',            # Overrides safety checks that prevent the
                                      # program from running too large calculations.

                    'LEVEL',          # Manually set the MOPAC theory level to be used,
                                      # default is PM7. Syntax: `LEVEL=PM7`
                                       
                    'MMFF'            # Use the Merck Molecular Force Field during the
                                      # OpenBabel pre-optimization (default is UFF).

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

                    'RIGID',          # Does not apply to "string" embeds. Avoid
                                      # bending structures to better build TSs.

                    'ROTRANGE',       # Does not apply to "string" embeds. Manually specify the rotation
                                      # range to be explored around the structure pivot.
                                      # Default is 120. Syntax: `ROTRANGE=120`

                    'SHRINK',         # Exaggerate orbital dimensions during embed, scaling them by a factor
                                      # of one and a half. This makes it easier to perform the embed without
                                      # having molecules clashing one another. Then, the correct distance between
                                      # reactive atom pairs is achieved as for standard runs by spring constraints
                                      # during MOPAC optimization.

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
    rotation_steps = None # This is set later by the _setup() function, based on embed type
    pruning_thresh = 1
    rigid = False
    
    max_clashes = 0
    clash_thresh = 1.3

    max_newbonds = 0

    optimization = True
    neb = False
    mopac_level = 'PM7'
    openbabel_level = 'UFF'
    suprafacial = False
    nci = False
    only_refined = False
    shrink = False
    mopac_opt = OPENBABEL_OPT_BOOL

    kcal_thresh = None
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

        if self.kcal_thresh == None:
            d.pop('kcal_thresh')

        if not OPENBABEL_OPT_BOOL:
            d.pop('openbabel_level')

        return '\n'.join([f'{var}{" "*(18-len(var))}: {d[var]}' for var in d])

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

                if len(reactive_atoms) > 2:
                    raise SyntaxError(f'Too many reactive atoms specified for {filename} ({len(reactive_atoms)})')

                reactive_indexes = tuple([int(re.sub('[^0-9]', '', i)) for i in reactive_atoms])

                if filename[0:5] == 'conf>':
                    filename = filename[5:]
                    confname = filename[:-4] + '_confs.xyz'

                    with suppress_stdout_stderr():
                        run(f'obabel {filename} -O {confname} --confab --rcutoff 1 --original', shell=True, check=True)

                    filename = confname
                    # redirect the input file to the new one with the conformers

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

                self.options.rotation_steps = 9

                if hasattr(self.options, 'custom_rotation_steps'):
                # if user specified a custom value, use it.
                    self.options.rotation_steps = self.options.custom_rotation_steps

                self.systematic_angles = cartesian_product(*[range(self.options.rotation_steps+1) for _ in self.objects]) * 2*self.options.rotation_range/self.options.rotation_steps - self.options.rotation_range

                self.candidates = len(self.systematic_angles)*np.prod([len(mol.atomcoords) for mol in self.objects])
                
                if len(self.objects) == 3:
                    self.candidates *= 8
                else:
                    self.candidates *= 2
                # The number 8 is the number of different triangles originated from three oriented vectors,
                # while 2 is the disposition of two vectors (parallel, antiparallel). This ends here if
                # no parings are to be respected. If there are any, each one reduces the number of
                # candidates to be computed, and we divide self.candidates number in the next section.

                if self.options.shrink:
                    for molecule in self.objects:
                        molecule._scale_orbs(self.options.shrink_multiplier)
                    self.options.only_refined = True

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

        Is this function badly written? Yup.
        Should i do something more elegant? Yup.
        Am I satisfied with it working as it is? Probably yup.
        TODO
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

######################################################################################################### RUN

    def run(self):
        '''
        '''

        def scramble(array, sequence):
            return np.array([array[s] for s in sequence])

        try:

            if not self.options.let:
                assert self.candidates < 1e8, ('ATTENTION! This calculation is probably going to be very big. To ignore this message' +
                                               ' and proceed, add the LET keyword to the input file.')

            head = ''
            for i, mol in enumerate(self.objects):
                descs = [atom.symbol+'('+str(atom)+f', {round(np.linalg.norm(atom.center[0]-atom.coord), 3)} A)' for atom in mol.reactive_atoms_classes_dict.values()]
                t = '\n        '.join([(str(index) + ' ' if len(str(index)) == 1 else str(index)) + ' -> ' + desc for index, desc in zip(mol.reactive_indexes, descs)])
                head += f'    {i+1}. {mol.name}:\n        {t}\n'

            self.log('--> Input structures, reactive indexes and reactive atoms TSCoDe type and orbital dimensions:\n' + head)
            self.log(f'--> Calculation options used were:')
            for line in str(self.options).split('\n'):
                if self.embed == 'string':
                    if line.split()[0] in ('rotation_range', 'rigid', 'suprafacial'):
                        continue
                self.log(f'    - {line}')

            t_start_run = time.time()

            if self.embed == 'cyclical' or self.embed == 'chelotropic':
                embedded_structures = cyclical_embed(self)

                if embedded_structures == []:
                    s = ('\n--> Cyclical embed did not find any suitable disposition of molecules.\n' +
                         '    This is probably because one molecule has two reactive centers at a great distance,\n' +
                         '    preventing the other two molecules from forming a closed, cyclical structure.')
                    self.log(s, p=False)
                    raise ZeroCandidatesError(s)

            else:
                embedded_structures = string_embed(self)


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
            self.log(f'Generated {len(self.structures)} transition state candidates ({time_to_string(t_end-t_start_run)})\n')

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

                    if False in mask:
                        self.log(f'Discarded {len([b for b in mask if not b])} candidates for compenetration ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
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
                            self.log(f'    - similarity pre-processing   (k={k}) - {time_to_string(t_end_int-t_start_int)} - kept {len([b for b in mask if b])}/{len(mask)}')
                    
                    t_start_int = time.time()
                    self.structures, mask = prune_conformers(self.structures, atomnos, max_rmsd=self.options.pruning_thresh)
                    t_end = time.time()
                    self.log(f'    - similarity final processing (k=1) - {time_to_string(t_end-t_start_int)} - kept {len([b for b in mask if b])}/{len(mask)}')

                    self.constrained_indexes = self.constrained_indexes[mask]

                    if False in mask:
                        self.log(f'Discarded {int(before - len([b for b in mask if b]))} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
                    self.log()

                    ################################################# CHECKPOINT BEFORE MM OPTIMIZATION

                    outname = f'TSCoDe_checkpoint_{self.stamp}.xyz'
                    with open(outname, 'w') as f:        
                        for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):
                            write_xyz(structure, atomnos, f, title=f'TS candidate {i+1} - Checkpoint before MM optimization')
                    self.log(f'--> Checkpoint output - Wrote {len(self.structures)} TS structures to {outname} file before MM optimization.\n')

                    ################################################# GEOMETRY OPTIMIZATION - FORCE FIELD

                    if len(self.structures) == 0:
                        raise ZeroCandidatesError()

                    if self.options.optimization:

                        if self.options.mopac_opt:

                            self.log(f'--> Structure optimization ({self.options.openbabel_level} level)')
                            self.exit_status = np.zeros(len(self.structures), dtype=bool)
                            t_start = time.time()

                            for i, structure in enumerate(deepcopy(self.structures)):
                                loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')
                                try:
                                    new_structure, self.exit_status[i] = openbabel_opt(structure, atomnos, self.constrained_indexes[i], graphs, method=self.options.openbabel_level)

                                    if self.exit_status[i]:
                                        self.structures[i] = new_structure

                                except Exception as e:
                                    raise e

                            loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')
                            t_end = time.time()
                            self.log(f'Openbabel {self.options.openbabel_level} optimization took {time_to_string(t_end-t_start)} (~{time_to_string((t_end-t_start)/len(self.structures))} per structure)')
                            
                            ################################################# DIFFERENTIATING: EXIT STATUS

                            # mask = self.exit_status
                            # self.structures = self.structures[mask]
                            # self.constrained_indexes = self.constrained_indexes[mask]
                            # self.exit_status = self.exit_status[mask]

                            if False in mask:
                                self.log(f'Successfully refined {len([b for b in self.exit_status if b])}/{len(self.structures)} candidates at UFF level. Non-refined structures are kept anyway.')
                            
                            ################################################# PRUNING: SIMILARITY (POST FORCE FIELD OPT)

                            if len(self.structures) == 0:
                                raise ZeroCandidatesError()

                            t_start = time.time()
                            self.structures, mask = prune_conformers(self.structures, atomnos, max_rmsd=self.options.pruning_thresh)
                            self.exit_status = self.exit_status[mask]
                            t_end = time.time()
                            
                            if False in mask:
                                self.log(f'Discarded {len([b for b in mask if not b])} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
                            self.log()

                            ################################################# CHECKPOINT BEFORE MOPAC OPTIMIZATION

                            with open(outname, 'w') as f:        
                                for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):
                                    exit_str = f'{self.options.openbabel_level} REFINED' if self.exit_status[i] else 'RAW'
                                    write_xyz(structure, atomnos, f, title=f'TS candidate {i+1} - {exit_str} - Checkpoint before MOPAC optimization')
                            self.log(f'--> Checkpoint output - Updated {len(self.structures)} TS structures to {outname} file before MOPAC optimization.\n')
                        
                        ################################################# GEOMETRY OPTIMIZATION - SEMIEMPIRICAL

                        if len(self.structures) == 0:
                            raise ZeroCandidatesError()

                        self.energies = np.zeros(len(self.structures))

                        t_start = time.time()

                        self.log(f'--> Structure optimization ({self.options.mopac_level} level)')

                        for i, structure in enumerate(deepcopy(self.structures)):
                            loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')
                            try:
                                t_start_opt = time.time()
                                new_structure, self.energies[i], self.exit_status[i] = optimize(structure,
                                                                                                atomnos,
                                                                                                graphs,
                                                                                                self.constrained_indexes[i],
                                                                                                method=f'{self.options.mopac_level} GEO-OK CYCLES=500',
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

                            t_end_opt = time.time()
                            self.log(f'    - Mopac {self.options.mopac_level} optimization: Structure {i+1} {exit_str} - took {time_to_string(t_end_opt-t_start_opt)}', p=False)

                        loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')
                        t_end = time.time()
                        self.log(f'Mopac {self.options.mopac_level} optimization took {time_to_string(t_end-t_start)} (~{time_to_string((t_end-t_start)/len(self.structures))} per structure)')

                        ################################################# PRUNING: SIMILARITY (POST SEMIEMPIRICAL OPT)

                        if len(self.structures) == 0:
                            raise ZeroCandidatesError()

                        t_start = time.time()
                        self.structures, mask = prune_conformers(self.structures, atomnos, max_rmsd=self.options.pruning_thresh)
                        self.energies = self.energies[mask]
                        self.exit_status = self.exit_status[mask]
                        t_end = time.time()
                        
                        if False in mask:
                            self.log(f'Discarded {len([b for b in mask if not b])} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
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

                        self._set_target_distances()

                        for i, structure in enumerate(deepcopy(self.structures)):
                            loadbar(i, len(self.structures), prefix=f'Refining structure {i+1}/{len(self.structures)} ')
                            try:
                                t_start_opt = time.time()
                                new_structure, new_energy, self.exit_status[i] = ase_adjust_spacings(self,
                                                                                                        structure,
                                                                                                        atomnos,
                                                                                                        self.constrained_indexes[i],
                                                                                                        graphs,
                                                                                                        method=self.options.mopac_level,
                                                                                                        max_newbonds=self.options.max_newbonds
                                                                                                    #  traj=f'adjust_{i}.traj'
                                                                                                        )

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
                                self.log(f'    - Mopac {self.options.mopac_level} refinement: Structure {i+1} {exit_str} - took {time_to_string(t_end_opt-t_start_opt)}', p=False)
                        
                        loadbar(1, 1, prefix=f'Refining structure {i+1}/{len(self.structures)} ')
                        t_end = time.time()
                        self.log(f'Mopac {self.options.mopac_level} refinement took {time_to_string(t_end-t_start)} (~{time_to_string((t_end-t_start)/len(self.structures))} per structure)')

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

                        if len(self.structures) == 0:
                            raise ZeroCandidatesError()

                        t_start = time.time()
                        self.structures, mask = prune_conformers(self.structures, atomnos, max_rmsd=self.options.pruning_thresh)
                        self.energies = self.energies[mask]
                        t_end = time.time()
                        
                        if False in mask:
                            self.log(f'Discarded {len([b for b in mask if not b])} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
                        self.log()


                except ZeroCandidatesError:
                    t_end_run = time.time()
                    s = ('Sorry, the program did not find any reasonable TS structure. Are you sure the input indexes were correct? If so, try these tips:\n' + 
                         '    - Enlarging the spacing between atom pairs with the DIST keyword\n' +
                         '    - Imposing less strict rejection criteria with the DEEP or CLASHES keyword.\n' +
                         '    - If the transition state is trimolecular, the SHRINK keyword may help (see documentation).\n')

                    self.log(f'\n--> Program termination: No candidates found - Total time {time_to_string(t_end_run-t_start_run)}')
                    self.log(s)
                    clean_directory()
                    raise ZeroCandidatesError(s)

            else:
                self.energies = np.zeros(len(self.structures))

            ################################################# PRUNING: ENERGY

            if self.options.optimization:

                self.energies = self.energies - np.min(self.energies)
                _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
                self.energies = scramble(self.energies, sequence)
                self.structures = scramble(self.structures, sequence)
                self.constrained_indexes = scramble(self.constrained_indexes, sequence)
                # sorting structures based on energy

                if self.options.kcal_thresh is not None:
            
                    mask = (self.energies - np.min(self.energies)) < self.options.kcal_thresh
                    self.structures = self.structures[mask]
                    self.energies = self.energies[mask]
                    self.exit_status = self.exit_status[mask]

                    if False in mask:
                        self.log(f'Discarded {len([b for b in mask if not b])} candidates for energy (Threshold set to {self.options.kcal_thresh} kcal/mol)')


            ################################################# XYZ GUESSES OUTPUT 

                outname = f'TSCoDe_TSs_guesses_{self.stamp}.xyz'
                with open(outname, 'w') as f:        
                    for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):

                        kind = 'REFINED - ' if self.exit_status[i] else 'NOT REFINED - '

                        write_xyz(structure, atomnos, f, title=f'Structure {i+1} - {kind}Rel. E. = {round(self.energies[i], 3)} kcal/mol')

                os.remove(f'TSCoDe_TSs_guesses_unrefined_{self.stamp}.xyz')
                # since we have the refined structures, we can get rid of the unrefined ones

                t_end_run = time.time()
                self.log(f'--> Output: Wrote {len(self.structures)} rough TS structures to {outname} file - Total time {time_to_string(t_end_run-t_start_run)}\n')

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

                        self.log(f'    - Mopac {self.options.mopac_level} NEB optimization: Structure {i+1} - {exit_str} - ({time_to_string(t_end_opt-t_start_opt)})', p=False)

                    loadbar(1, 1, prefix=f'Performing NEB {len(self.structures)}/{len(self.structures)} ')
                    t_end = time.time()
                    self.log(f'Mopac {self.options.mopac_level} NEB optimization took {time_to_string(t_end-t_start)} ({time_to_string((t_end-t_start)/len(self.structures))} per structure)')
                    self.log(f'NEB converged for {len([i for i in self.exit_status if i])}/{len(self.structures)} structures\n')

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
                        
                        if False in mask:
                            self.log(f'Discarded {len([b for b in mask if not b])} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
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
                        self.log(f'Mopac {self.options.mopac_level} frequency calculation took {time_to_string(t_end-t_start)} (~{time_to_string((t_end-t_start)/len(self.structures))} per structure)\n')

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

            ################################################# NON-COVALENT INTERACTION GUESSES

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

            if not self.options.bypass:

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

            self.log(f'--> TSCoDe normal termination: total time {time_to_string(t_end_run - t_start_run, verbose=True)}.')

            self.logfile.close()

            #### EXTRA
            
            # path = os.path.join(os.getcwd(), vmd_name)
            # os.system(f'vmd -e {path}')

            ################################################

        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt requested by user. Quitting.')
            quit()


if __name__ == '__main__':

    import sys

    usage = '\n\tTSCoDe correct usage:\n\n\tpython tscode.py input.txt\n\n\tSee documentation for input formatting.\n'

    if len(sys.argv) < 2 or len(sys.argv[1].split('.')) == 1:

        print(usage)
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

