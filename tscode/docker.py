# coding=utf-8
'''

TSCODE: Transition State Conformational Docker
Copyright (C) 2021 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''
import os
import re
import time
from itertools import groupby

import numpy as np

from tscode.docker_options import Options, OptionSetter, keywords_list
from tscode.errors import InputError
from tscode.fast_algebra import norm_of
from tscode.hypermolecule_class import Hypermolecule, Pivot
from tscode.operators import operate
from tscode.settings import CALCULATOR, DEFAULT_LEVELS, PROCS
from tscode.utils import ase_view, cartesian_product


class Docker:
    '''
    Docker class, containing all methods to set attributes,
    options and initialize the calculation
    '''

    def __init__(self, filename, stamp=None):
        '''
        Initialize the Docker object by reading the input filename (.txt).
        Sets the Option dataclass properties to default and then updates them
        with the user-requested keywords, if there are any.

        '''
        os.chdir(os.path.dirname(filename))

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
        s += '*                Version 0.0.4 - Public Beta                *\n'
        s += "*       Nicolo' Tampellini - nicolo.tampellini@yale.edu     *\n"
        s += '*************************************************************\n'

        self.log(s)

        self.options = Options()

        inp = self._parse_input(filename)
        self.objects = [Hypermolecule(name, c_ids) for name, c_ids in inp]

        self.ids = [len(mol.atomnos) for mol in self.objects]
        # Used to divide molecules in TSs

        self.graphs = [mol.graph for mol in self.objects]
        # Store molecular graphs

        self._set_reactive_atoms_cumnums()
        # appending to each reactive atom the cumulative
        # number indexing in the TS context

        self._read_pairings()
        self._set_options(filename)
        self._calculator_setup()
        self._apply_operators()

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

        lines = [line for line in lines if line[0] not in ('#', '\n')]
        
        try:
            assert len(lines) < 5
            # (optional) keyword line + 1, 2 or 3 lines for molecules

            keywords = [l.split('=')[0] if not '(' in l else l.split('(')[0] for l in lines[0].split()]
            if any(k in keywords_list for k in keywords):
                self.kw_line, *self.mol_lines = lines
            else:
                self.mol_lines = lines

            inp = []
            for line in self.mol_lines:
                filename, *reactive_atoms = line.split()

                if len(reactive_atoms) > 4:
                    s = f'Too many reactive atoms specified for {filename} ({len(reactive_atoms)}).'
                    raise SyntaxError(s)

                reactive_indexes = tuple([int(re.sub('[^0-9]', '', i)) for i in reactive_atoms])

                if '>' in filename:
                    self.options.operators.append(filename)
                    filename = filename.split('>')[-1]
                    # record that we will need to perform these operations before the run

                inp.append((filename, reactive_indexes))

            return inp
            
        except Exception as e:
            print(e)
            raise InputError(f'Error in reading molecule input for {filename}. Please check your syntax.')

    def _set_options(self, filename):
        '''
        Set the options dataclass parameters through the OptionSetter class,
        from a list of given keywords. These will be used during the run to
        vary the search depth and/or output.
        '''
      
        if hasattr(self, 'kw_line'):
            try:
                option_setter = OptionSetter(self.kw_line, self)
                option_setter.set_options()

            except SyntaxError as e:
                raise e

            except Exception as e:
                print(e)
                raise InputError(f'Error in reading keywords from {filename}. Please check your syntax.')

    def _set_reactive_atoms_cumnums(self):

        for i, mol in enumerate(self.objects):
            if mol.hyper:
                for c, _ in enumerate(mol.atomcoords):
                    for r_atom in mol.reactive_atoms_classes_dict[c].values():
                        r_atom.cumnum = r_atom.index
                        if i > 0:
                            r_atom.cumnum += sum(self.ids[:i])

    def _read_pairings(self):
        '''
        Reads atomic pairings to be respected from the input file, if any are present.
        '''

        parsed = []
        unlabeled_list = []
        self.pairings_dict = {i:{} for i, _ in enumerate(self.objects)}

        for i, line in enumerate(self.mol_lines):
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

                    for l in letters:
                        if l not in ('a', 'b', 'c', 'x', 'y', 'z'):
                            raise SyntaxError(f'Letter \'{l}\' not accepted. Please use only these letters to specify pairings:\n' +
                                               '    reacting atoms: "a", "b" and "c"\n    interactions: "x", "y" and "z"\n')

                        pairings.append([int(index), l])

            for index, letter in pairings:
                self.pairings_dict[i][letter] = index
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

        pairings = sorted(list(links.items()), key=lambda x: x[0])
        # sorting values so that 'a' is the first pairing

        self.pairings_table = {i[0]:sorted(i[1]) for i in pairings}
        # {'a':[3,45]}

        letters = tuple(self.pairings_table.keys())

        for letter, ids in self.pairings_table.items():

            if len(ids) == 1:
                raise SyntaxError(f'Letter \'{letter}\' is only specified once. Please flag the second reactive atom.')

            if len(ids) > 2:
                raise SyntaxError(f'Letter \'{letter}\' is specified more than two times. Please remove the unwanted letters.')

        if not self.pairings_table:
            if all([len(mol.reactive_indexes) == 2 for mol in self.objects]):
                self.log('--> No atom pairings imposed. Computing all possible dispositions.\n')
                # only print the no pairings statements if there are multiple regioisomers to be computed
        else:
            self.log(f'--> Atom pairings imposed are {len(self.pairings_table)}: {list(self.pairings_table.values())} (Cumulative index numbering)\n')
        
        if len(self.mol_lines) == 3:
        # adding third pairing if we have three molecules and user specified two pairings
        # (used to adjust distances for trimolecular TSs)
            if len(unlabeled_list) == 2:
                third_constraint = list(sorted(unlabeled_list))
                self.pairings_table['?'] = third_constraint
        
        elif len(self.mol_lines) == 2:
        # adding second pairing if we have two molecules and user specified one pairing
        # (used to adjust distances for bimolecular TSs)
            if len(unlabeled_list) == 2:
                second_constraint = list(sorted(unlabeled_list))
                self.pairings_table['?'] = second_constraint

    def _set_custom_orbs(self, orb_string):
        '''
        Update the reactive_atoms classes with the user-specified orbital distances.
        :param orb_string: string that looks like 'a=2.345,b=3.456,c=2.22'

        '''
        self.pairings_dists = {piece.split('=')[0] : float(piece.split('=')[1]) for piece in orb_string.split(',')}

        for letter, dist in self.pairings_dists.items():

            if letter not in self.pairings_table:
                raise SyntaxError(f'Letter \'{letter}\' is specified in DIST but not present in molecules string.')

            for i, mol in enumerate(self.objects):
                for c, _ in enumerate(mol.atomcoords):

                    r_index = self.pairings_dict[i].get(letter)
                    if r_index is None:
                        continue

                    r_atom = mol.reactive_atoms_classes_dict[c][r_index]
                    r_atom.init(mol, r_index, update=True, orb_dim=dist/2)
                                
            # Set the new orbital center with imposed distance from the reactive atom. The imposed distance is half the 
            # user-specified one, as the final atomic distances will be given by two halves of this length.

    def _set_pivots(self, mol):
        '''
        params mol: Hypermolecule class
        (Cyclical embed) Function that sets the mol.pivots attribute, that is a list
        containing each vector connecting two orbitals on different atoms or on the
        same atom (for single-reactive atom molecules in chelotropic embedding)
        '''
        mol.pivots = self._get_pivots(mol)

        for c, _ in enumerate(mol.atomcoords):

            if len(mol.pivots[c]) == 2:
            # reactive atoms have one and two centers,
            # respectively. Apply bridging carboxylic acid correction.
                symbols = [atom.symbol for atom in mol.reactive_atoms_classes_dict[c].values()]
                if 'H' in symbols:
                    if ('O' in symbols) or ('S' in symbols):
                        if max([norm_of(p.pivot)/self.options.shrink_multiplier for p in mol.pivots[c]]) < 4.5:
                            class_types = [str(atom) for atom in mol.reactive_atoms_classes_dict[c].values()]
                            if 'Single Bond' in class_types and 'Ketone' in class_types:
                            # if we have a bridging acid, remove the longest of the two pivots,
                            # as it would lead to weird structures
                                norms = np.linalg.norm([p.pivot for p in mol.pivots[c]], axis=1)
                                for sample in norms:
                                    to_keep = [i for i in norms if sample >= i]
                                    if len(to_keep) == 1:
                                        mask = np.array([i in to_keep for i in norms])
                                        mol.pivots[c] = mol.pivots[c][mask]
                                        break

            if self.options.suprafacial:
                if len(mol.pivots[c]) == 4:
                # reactive atoms have two centers each.
                # Applying suprafacial correction, only keeping
                # the shorter two, as they should be the suprafacial ones
                    norms = np.linalg.norm([p.pivot for p in mol.pivots[c]], axis=1)
                    for sample in norms:
                        to_keep = [i for i in norms if sample >= i]
                        if len(to_keep) == 2:
                            mask = np.array([i in to_keep for i in norms])
                            mol.pivots[c] = mol.pivots[c][mask]
                            break

            if mol.sp3_sigmastar:
                pivots_lengths = [norm_of(pivot.pivot) for pivot in mol.pivots[c]]
                shortest_length = min(pivots_lengths)
                mask = np.array([(i - shortest_length) < 1e-5 for i in pivots_lengths])
                mol.pivots[c] = mol.pivots[c][mask]
            # if mol is reacting with a sigmastar orbital (two connected reactive Sp3/Single
            # Bond centers) then remove all pivots that are not the shortest. This ensures
            # the "suprafaciality" to the pivots used, preventing the embed of
            # impossible bonding structures

    def _get_pivots(self, mol):
        '''
        params mol: Hypermolecule class
        (Cyclical embed) Function that yields the molecule pivots. Called by _set_pivots
        and in pre-conditioning (deforming, bending) the molecules in ase_bend.
        '''
        pivots_list = [[] for _ in mol.atomcoords]

        for c, _ in enumerate(mol.atomcoords):

            if len(mol.reactive_atoms_classes_dict[c]) == 2:
            # most molecules: dienes and alkenes for Diels-Alder, conjugated ketones for acid-bridged additions

                indexes = cartesian_product(*[range(len(atom.center)) for atom in mol.reactive_atoms_classes_dict[c].values()])
                # indexes of vectors in reactive_atom.center. Reactive atoms are 2 and so for one center on atom 0 and 
                # 2 centers on atom 2 we get [[0,0], [0,1], [1,0], [1,1]]

                for i,j in indexes:
                    a1, a2 = mol.get_r_atoms(c)
                   
                    c1 = a1.center[i]
                    c2 = a2.center[j]

                    pivots_list[c].append(Pivot(c1, c2, a1, a2, i, j))

            elif len(mol.reactive_atoms_classes_dict[c]) == 1:
            # carbenes, oxygen atom in Prilezhaev reaction, SO2 in classic chelotropic reactions

                indexes = cartesian_product(*[range(len(mol.get_r_atoms(c)[0].center)) for _ in range(2)])
                indexes = [i for i in indexes if i[0] != i[1] and (sorted(i) == i).all()]
                # indexes of vectors in reactive_atom.center. Reactive atoms is just one, that builds pivots with itself. 
                # pivots with the same index or inverse order are discarded. 2 centers on one atom 2 yield just [[0,1]]
                
                for i,j in indexes:
                    a1 = mol.get_r_atoms(c)[0]
                    # chelotropic embeds have pivots that start/end on the same atom

                    c1 = a1.center[i]
                    c2 = a1.center[j]

                    pivots_list[c].append(Pivot(c1, c2, a1, a1, i, j))

        return [np.array(l) for l in pivots_list]

    def _setup(self, p=True):
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
            
        elif len(self.objects) in (2,3):
        # Setting embed type and calculating the number of conformation combinations based on embed type

            cyclical = all([len(molecule.reactive_atoms_classes_dict[0]) == 2 for molecule in self.objects])
            chelotropic = sorted([len(molecule.reactive_atoms_classes_dict[0]) for molecule in self.objects]) == [1,2]
            string = all([len(molecule.reactive_atoms_classes_dict[0]) == 1 for molecule in self.objects]) and len(self.objects) == 2

            if cyclical or chelotropic:

                if cyclical:
                    self.embed = 'cyclical'
                else:
                    self.embed = 'chelotropic'
                    for mol in self.objects:
                        for c, _ in enumerate(mol.atomcoords):
                            for index, atom in mol.reactive_atoms_classes_dict[c].items():
                                orb_dim = norm_of(atom.center[0]-atom.coord)
                                atom.init(mol, index, update=True, orb_dim=orb_dim + 0.2)
                    # Slightly enlarging orbitals for chelotropic embeds, or they will
                    # be generated a tad too close to each other for how the cyclical embed works          

                self.options.rotation_steps = 9

                if hasattr(self.options, 'custom_rotation_steps'):
                # if user specified a custom value, use it.
                    self.options.rotation_steps = self.options.custom_rotation_steps

                self.systematic_angles = cartesian_product(*[range(self.options.rotation_steps+1) for _ in self.objects]) \
                            * 2*self.options.rotation_range/self.options.rotation_steps - self.options.rotation_range

                for molecule in self.objects:
                    self._set_pivots(molecule)

            elif string:
                
                self.embed = 'string'
                self.options.rotation_steps = 24

                if hasattr(self.options, 'custom_rotation_steps'):
                # if user specified a custom value, use it.
                    self.options.rotation_steps = self.options.custom_rotation_steps

                self.systematic_angles = [n * 360 / self.options.rotation_steps for n in range(self.options.rotation_steps)]
   
            else:
                raise InputError(('Bad input - The only molecular configurations accepted are:\n' 
                                  '1) One molecule with two reactive centers (monomolecular embed)\n'
                                  '2) One molecule with four indexes (dihedral embed)\n'
                                  '3) Two or three molecules with two reactive centers each (cyclical embed)\n'
                                  '4) Two molecules with one reactive center each (string embed)\n'
                                  '5) Two molecules, one with a single reactive center and the other with two (chelotropic embed)'))
        else:
            raise InputError('Bad input - too many/few molecules specified (one to three required).')

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

        self.candidates = self._get_number_of_candidates()

        if p:
            self.log(f'--> Setup performed correctly. {self.candidates} candidates will be generated.\n')

    def _get_number_of_candidates(self):
        '''
        Get the number of structures that will be generated in the run.
        '''
        l = len(self.objects)
        if l == 1:
            return int(sum([len(self.objects[0].pivots[c])
                            for c, _ in enumerate(self.objects[0].atomcoords)]))

        if self.embed == 'string':
            return int(self.options.rotation_steps*(
                       np.prod([sum([len(mol.get_r_atoms(conf)[0].center)
                                     for conf, _ in enumerate(mol.atomcoords)]) 
                                for mol in self.objects]))
                      )

        candidates = 2*len(self.systematic_angles)*np.prod([len(mol.atomcoords) if self.options.let else min(len(mol.atomcoords), 10)
                                                            for mol in self.objects])
        
        if l == 3:
            candidates *= 4
        # Trimolecular there are 8 different triangles originated from three oriented vectors,
        # while only 2 disposition of two vectors (parallel, antiparallel).

        if self.pairings_table:
        # If there is any pairing to be respected, each one reduces the number of
        # candidates to be computed.

            if self.embed == 'cyclical':
                if len(self.objects) == 2:
                # Diels-Alder-like, if we have one (two) pairing(s) only half
                # of the total arrangements are to be checked
                    candidates /= 2

                else: # trimolecular
                    if len(self.pairings_table) == 1:
                        candidates /= 4
                    else: # trimolecular, 2 (3) pairings imposed
                        candidates /= 8

        candidates *= np.prod([len(mol.pivots[0]) for mol in self.objects]) # add sum over len(mol.pivots[c])?
        # The more atomic pivots, the more candidates

        return int(candidates)

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

            if self.options.theory_level in ('MNDO','AM1','PM3','HF-3c','HF MINIX D3BJ GCP(HF/MINIX) PATOM') and self.options.procs != 1:
                raise Exception(('ORCA does not support parallelization for Semiempirical Methods. '
                                 'Please change the value of PROCS to 1 or change the theory level '
                                 'from the setup command.'))

        # if self.options.solvent is not None:
        #     from solvents import get_solvent_line
        #     self.options.theory_level = get_solvent_line(self.options.solvent,
        #                                                  self.options.calculator,
        #                                                  self.options.theory_level)

    def _apply_operators(self):
        '''
        Replace molecules in self.objects with
        their post-operators ones.
        '''
        for input_string in self.options.operators:

            outname = operate(input_string, self)

            names = [mol.name for mol in self.objects]
            filename = input_string.split('>')[-1]
            index = names.index(filename)
            reactive_indexes = self.objects[index].reactive_indexes
            hyper = self.objects[index].hyper

            self.objects[index] = Hypermolecule(outname, reactive_indexes, hyper)
            # replacing the old molecule with the one post-operators

            self._set_reactive_atoms_cumnums()
            # updating the orbital cumnums

    def _inspect_structures(self):
        '''
        '''

        self.log('--> Structures check requested. Shutting down after last window is closed.\n')

        for mol in self.objects:
            ase_view(mol)
        
        self.logfile.close()
        os.remove(f'TSCoDe_{self.stamp}.log')

        quit()

    def scramble(self, array, sequence):
        return np.array([array[s] for s in sequence])
