# coding=utf-8
'''

TSCODE: Transition State Conformational Docker
Copyright (C) 2021-2023 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''
import logging
import os
import random
import re
import sys
import time
from getpass import getuser
from itertools import groupby

import numpy as np

from tscode.__main__ import __version__
from tscode.algebra import norm_of
from tscode.embedder_options import Options, OptionSetter, keywords_list
from tscode.embeds import _get_monomolecular_reactive_indices
from tscode.errors import InputError, NoOrbitalError
from tscode.graph_manipulations import get_sum_graph
from tscode.hypermolecule_class import Hypermolecule, Pivot
from tscode.operators import operate
from tscode.run import RunEmbedding
from tscode.settings import CALCULATOR, DEFAULT_LEVELS, PROCS, THREADS
from tscode.utils import (ase_view, auto_newline, cartesian_product, graphize,
                          time_to_string)


class Embedder:
    '''
    Embedder class, containing all methods to set attributes,
    options and initialize the calculation
    '''

    def __init__(self, filename, stamp=None, procs=None, threads=None):
        '''
        Initialize the Embedder object by reading the input filename (.txt).
        Sets the Option dataclass properties to default and then updates them
        with the user-requested keywords, if there are any.

        '''

        self.t_start_run = time.perf_counter()
        os.chdir(os.path.dirname(filename))

        if stamp is None:
            self.stamp = time.ctime().replace(' ','_').replace(':','-')[4:-8]
            # replaced ctime yields 'Sun_May_23_18-53-47_2021', only keeping 'May_23_18-53'

        else:
            self.stamp = stamp

        self.avail_cpus = len(os.sched_getaffinity(0))
        self.threads = int(threads) if threads is not None else THREADS or self.avail_cpus//4 or 1
        self.procs = int(procs) if procs is not None else PROCS or 4

        try:
            os.remove(f'TSCoDe_{self.stamp}.log')

        except FileNotFoundError:
            pass

        log_filename = f'TSCoDe_{self.stamp}.log'
        self.logfile = open(log_filename, 'a', buffering=1, encoding="utf-8")
        logging.basicConfig(filename=log_filename, filemode='a')

        try:

            self.write_banner()
            # Write banner to log file

            self.options = Options()
            # initialize option subclass

            self.embed = None
            # initialize embed type variable, to be modified later

            inp = self._parse_input(filename)
            # collect information about molecule files

            self.objects = [Hypermolecule(name, c_ids) for name, c_ids in inp]
            # load designated molecular files

            # self.objects.sort(key=lambda obj: len(obj.atomcoords[0]), reverse=True)
            # sort them in descending number of atoms (not for now - messes up pairings)

            self.ids = np.array([len(mol.atomnos) for mol in self.objects])
            # Compute length of each molecule coordinates. Used to divide molecules in TSs

            self.graphs = [mol.graph for mol in self.objects]
            # Store molecular graphs

            self._read_pairings()
            # read imposed pairings from input file [i.e. mol1(6)<->mol2(45)]

            self._set_options(filename)
            # read the keywords line and set the relative options
            # then read the operators and store them 

            self._calculator_setup()
            # initialize default or specified calculator

            self._apply_operators()
            # execute the operators, replacing the self.objects molecule

            self._setup()
            # setting embed type and getting ready to embed (if needed)

            if self.options.debug:
                for mol in self.objects:
                    if hasattr(mol, 'reactive_atoms_classes_dict'):
                        if len(mol.reactive_atoms_classes_dict[0]) > 0:
                            mol.write_hypermolecule()
                self.log(f'--> DEBUG: written hypermolecule files ({len(self.objects)})\n')

            if self.options.check_structures:
                self._inspect_structures()

        except Exception as e:
            logging.exception(e)
            raise e

    def log(self, string='', p=True):
        if p:
            print(string)
        string += '\n'
        self.logfile.write(string)

    def write_banner(self):
        '''
        Write banner to log file, containing program and run info
        '''
        banner = '''
       +   .     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ .     .    
    *    .   .. ╱────────────────────────────────────╲   *     .  
 .     ..   +  ╱▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒ ╲ .   .   +  
   +       ▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒ . ..   .  
     .  ▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒  .   *  
       ▒░████████╗░██████╗░█████╗░░█████╗░██████╗░███████╗░░░▒  .  .  
   +   ▒░╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝░░░▒   ..  
 ..  . ▒░░░░██║░░░╚█████╗░██║░░╚═╝██║░░██║██║░░██║█████╗░░░░░▒ *    +   
   .   ▒░░░░██║░░░░╚═══██╗██║░░██╗██║░░██║██║░░██║██╔══╝░░░░░▒   .   .
.       ▒░░░██║░░░██████╔╝╚█████╔╝╚█████╔╝██████╔╝███████╗░░░▒ ..   +   
 *  .  ╱ ▒░░╚═╝░░░╚═════╝░░╚════╝░░╚════╝░╚═════╝░╚══════╝░░▒ ╲ .  ..  
  ..  ╱   ▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒   ╲   .   
.    ╱    ▒░░╔══════════════════════════════════════════╗░░▒    ╲ +    
    ╱      ▒░║  Transition State Conformational Docker  ║░▒      ╲ ..  
 +  ╲╲     ▒░║        nicolo.tampellini@yale.edu        ║░▒     ╱╱    .  
     ╲╲    ▒░║                                          ║░▒    ╱╱  .       
 ..   ╲╲   ▒░║     Version    >{0:^25}║░▒   ╱╱ .  *                                    
   .   ╲╲  ▒░║      User      >{1:^25}║░▒  ╱╱   .                                     
        ╲╲ ▒░║      Time      >{2:^25}║░▒ ╱╱ *   .                                                      
 ..   *  ╲╲▒░║      Procs     >{3:^25}║░▒╱╱   ..            
    .     ╲▒░║     Threads    >{4:^25}║░▒╱  +              
      .    ▒░║    Avail CPUs  >{5:^25}║░▒ .   ..                            
  +  .. .  ▒░╚══════════════════════════════════════════╝░▒  .. .   
    .       ▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒     .     
 .     *  +   ╲╲ ▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒ ╱╱  .      .
     .      .  ╲╲▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁╱╱ .   .    
           '''.format(__version__,
                      getuser(),
                      time.ctime()[0:-8],
                      self.procs,
                      self.threads,
                      self.avail_cpus)

        # ⏣█▓▒░ banner art adapted from https://fsymbols.com/generators/tarty/

        self.log(banner)

        if self.procs * self.threads > self.avail_cpus:
            self.log(f'--> ATTENTION: Excessive hyperthreading - {self.avail_cpus} CPUs (w/Hyperthreading) detected, {self.procs}*{self.threads} will be used')

    def _parse_input(self, filename):
        '''
        Reads a textfile and sets the Embedder properties for the run.
        Keywords are read from the first non-comment(#), non-blank line
        if there are any, and molecules are read afterward.

        '''

        with open(filename, 'r') as f:
            lines = f.readlines()

        lines = [line.replace(', ',',') for line in lines if line[0] not in ('#', '\n')]
        
        def _remove_internal_constraints(string):
            numbers = [int(re.sub('[^0-9]', '', i)) for i in string]
            letters = [re.sub('[^a-z]', '', i) for i in string]
            count = [letters.count(l) if (l != '') else 1 for l in letters]
            return tuple([n for n, c in zip(numbers, count) if c == 1])

        try:

            keywords = [l.split('=')[0] if not '(' in l else l.split('(')[0] for l in lines[0].split()]
            if any(k.upper() in keywords_list for k in keywords):
                self.kw_line, *self.mol_lines = lines
            else:
                self.mol_lines = lines

            inp = []
            for _l, line in enumerate(self.mol_lines):

                if '>' in line:
                    self.options.operators_dict[_l] = [op.rstrip().lstrip() for op in reversed(line.rstrip('\n').split('>')[:-1])]
                    self.options.operators.append(line.rstrip('\n'))
                    line = line.split('>')[-1].lstrip()
                    # record that we will need to perform these operations before the run

                filename, *reactive_atoms = line.split()

                if reactive_atoms:
                    reactive_indices = _remove_internal_constraints(reactive_atoms)
                else:
                    reactive_indices = None

                inp.append((filename, reactive_indices))

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
      
        try:
            option_setter = OptionSetter(self)
            option_setter.set_options()

        except SyntaxError as e:
            raise e

        except Exception as e:
            print(e)
            raise InputError(f'Error in reading keywords from {filename}. Please check your syntax.')

    def _set_reactive_atoms_cumnums(self):

        if self.embed in ('cyclical', 'chelotropic', 'string'):
            for i, mol in enumerate(self.objects):

                if not hasattr(mol, 'reactive_atoms_classes_dict'):
                    mol.compute_orbitals()

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

            fragments = line.split('>')[-1].split()[1:]
            # remove operators (if present) and the molecule name, keeping pairs only ['2a','5b']

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

            # appending pairing to dict before
            # calculating their cumulative index
            # If a pairing is already present, add the number
            # (refine>/REFINE runs)
            for index, letter in pairings:

                if self.pairings_dict[i].get(letter) is not None:
                    prev = self.pairings_dict[i][letter]
                    self.pairings_dict[i][letter] = (prev, index)

                else:
                    self.pairings_dict[i][letter] = index

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
        # cumulative indices for TSs

        links = {j:[] for j in set([i[1] for i in parsed])}
        for index, tag in parsed:
            links[tag].append(index)
        # storing couples into a dictionary

        pairings = sorted(list(links.items()), key=lambda x: x[0])
        # sorting values so that 'a' is the first pairing

        self.pairings_table = {i[0]:sorted(i[1]) for i in pairings}
        # cumulative, looks like {'a':[3,45]}

        letters = tuple(self.pairings_table.keys())

        for letter, ids in self.pairings_table.items():

            if len(ids) == 1:
                raise SyntaxError(f'Letter \'{letter}\' is only specified once. Please flag the second reactive atom.')

            if len(ids) > 2:
                raise SyntaxError(f'Letter \'{letter}\' is specified more than two times. Please remove the unwanted letters.')
        
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

        # Now record the internal constraints, that is the intramolecular
        # distances to freeze and later enforce to the imposed spacings
        self.internal_constraints = []

        # making sure we set the kw_line attribute
        self.kw_line = self.kw_line if hasattr(self, 'kw_line') else ''
        
        for letter, pair in self.pairings_table.items():
            for mol_id in self.pairings_dict:
                if isinstance(self.pairings_dict[mol_id].get(letter), tuple):

                    # They are internal constraints only if we have a distance 
                    # to impose later on. We are checking this way because the
                    # set_options function is still to be called at this stage
                    if f'{letter}=' in self.kw_line:
                        self.internal_constraints.append([pair])
        self.internal_constraints = np.concatenate(self.internal_constraints) if self.internal_constraints else []

    def _set_custom_orbs(self, orb_string):
        '''
        Update the reactive_atoms classes with the user-specified orbital distances.
        :param orb_string: string that looks like 'a=2.345,b=3.456,c=2.22'

        '''
        for mol in self.objects:
            if not hasattr(mol, 'reactive_atoms_classes_dict'):
                mol.compute_orbitals()

        self.pairing_dists = {piece.split('=')[0] : float(piece.split('=')[1]) for piece in orb_string.split(',')}

        # Set the new orbital center with imposed distance from the reactive atom. The imposed distance is half the 
        # user-specified one, as the final atomic distances will be given by two halves of this length.
        for letter, dist in self.pairing_dists.items():

            if letter not in self.pairings_table:
                raise SyntaxError(f'Letter \'{letter}\' is specified in DIST but not present in molecules string.')

            for i, mol in enumerate(self.objects):
                for c, _ in enumerate(mol.atomcoords):

                    r_index = self.pairings_dict[i].get(letter)
                    if r_index is None:
                        continue
                    
                    if isinstance(r_index, int):
                        r_atom = mol.reactive_atoms_classes_dict[c][r_index]
                        r_atom.init(mol, r_index, update=True, orb_dim=dist/2, conf=c)
                    
                    else:
                        for r_i in r_index:
                            r_atom = mol.reactive_atoms_classes_dict[c].get(r_i)
                            if r_atom:
                                r_atom.init(mol, r_i, update=True, orb_dim=dist/2, conf=c)

        # saves the last orb_string executed so that operators can
        # keep the imposed orbital spacings when replacing molecules
        self.orb_string = orb_string
        # self.log(f'DEBUG ---> Updated orb string -> {orb_string}')

    def _set_pivots(self, mol):
        '''
        params mol: Hypermolecule class
        (Cyclical embed) Function that sets the mol.pivots attribute, that is a list
        containing each vector connecting two orbitals on different atoms or on the
        same atom (for single-reactive atom molecules in chelotropic embedding)
        '''
        mol.pivots = self._get_pivots(mol)

        for c, _ in enumerate(mol.atomcoords):
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

            # if mol is reacting with a sigmastar orbital (two connected reactive Sp3/Single
            # Bond centers) then remove all pivots that are not the shortest. This ensures
            # the "suprafaciality" to the pivots used, preventing the embed of
            # impossible bonding structures
            if hasattr(mol, 'sp3_sigmastar') and mol.sp3_sigmastar:
                pivots_lengths = [norm_of(pivot.pivot) for pivot in mol.pivots[c]]
                shortest_length = min(pivots_lengths)
                mask = np.array([(i - shortest_length) < 1e-5 for i in pivots_lengths])
                mol.pivots[c] = mol.pivots[c][mask]

    def _get_pivots(self, mol):
        '''
        params mol: Hypermolecule class
        (Cyclical embed) Function that yields the molecule pivots. Called by _set_pivots
        and in pre-conditioning (deforming, bending) the molecules in ase_bend.
        '''

        if not hasattr(mol, 'reactive_atoms_classes_dict'):
            return []

        pivots_list = [[] for _ in mol.atomcoords]

        for c, _ in enumerate(mol.atomcoords):

            if len(mol.reactive_atoms_classes_dict[c]) == 2:
            # most molecules: dienes and alkenes for Diels-Alder, conjugated ketones for acid-bridged additions

                indices = cartesian_product(*[range(len(atom.center)) for atom in mol.reactive_atoms_classes_dict[c].values()])
                # indices of vectors in reactive_atom.center. Reactive atoms are 2 and so for one center on atom 0 and 
                # 2 centers on atom 2 we get [[0,0], [0,1], [1,0], [1,1]]

                for i,j in indices:
                    a1, a2 = mol.get_r_atoms(c)
                   
                    c1 = a1.center[i]
                    c2 = a2.center[j]

                    pivots_list[c].append(Pivot(c1, c2, a1, a2, i, j))

            elif len(mol.reactive_atoms_classes_dict[c]) == 1:
            # carbenes, oxygen atom in Prilezhaev reaction, SO2 in classic chelotropic reactions

                indices = cartesian_product(*[range(len(mol.get_r_atoms(c)[0].center)) for _ in range(2)])
                indices = [i for i in indices if i[0] != i[1] and (sorted(i) == i).all()]
                # indices of vectors in reactive_atom.center. Reactive atoms is just one, that builds pivots with itself. 
                # pivots with the same index or inverse order are discarded. 2 centers on one atom 2 yield just [[0,1]]
                
                for i,j in indices:
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

        if any('pka>'      in op for op in self.options.operators) or (
           any('scan>' in op for op in self.options.operators)
        ):
            self.embed = 'data'
            # If a pka or scan operator is requested, the embed is skipped
            # and data is shown instead
            return

        if any('refine>' in op for op in self.options.operators) or self.options.noembed:
            self.embed = 'refine'

            # If the run is a refine>/REFINE one, the self.embed
            # attribute is set in advance by the self._set_options
            # function through the OptionSetter class
            return

        for mol in self.objects:
            if self.options.max_confs < len(mol.atomcoords):
                self.log(f'--> {mol.name} - kept {self.options.max_confs}/{len(mol.atomcoords)} conformations for the embed (override with CONFS=n)\n')
                mol.atomcoords = mol.atomcoords[0:self.options.max_confs]
        # remove conformers if there are too many

        if all([len(mol.reactive_indices) == 0 for mol in self.objects]):
            self.embed = None
            # Flag the embed type as None if no reactive indices are
            # provided (and the run is not a refine> one)
            return

        if len(self.objects) == 1:
        # embed must be either monomolecular or dihedral

            mol = self.objects[0]

            if len(mol.reactive_indices) == 4:
                self.embed = 'dihedral'
                if 'kcal' not in self.kw_line.lower():
                # set to 5 if user did not specify a value
                    self.options.kcal_thresh = 5

                if 'rmsd' not in self.kw_line.lower():
                # set to 0.2 if user did not specify a value
                    self.options.rmsd = 0.2

                return

            if len(mol.reactive_indices) == 2:

                self.embed = 'monomolecular'
                mol.compute_orbitals()
                self._set_pivots(mol)

                self.options.only_refined = True
                self.options.fix_angles_in_deformation = True
                # These are required: otherwise, extreme bending could scramble molecules

            else:
                self.embed = 'error'
                # if none of the previous, the program had trouble recognizing the embed to carry.

                return
            
        elif len(self.objects) in (2,3):
        # Setting embed type and calculating the number of conformation combinations based on embed type

            cyclical = all(len(molecule.reactive_indices) == 2 for molecule in self.objects)

            # chelotropic embed should check that the two atoms on one molecule are bonded
            chelotropic = sorted(len(molecule.reactive_indices) for molecule in self.objects) == [1,2]

            string = all(len(molecule.reactive_indices) == 1 for molecule in self.objects) and len(self.objects) == 2

            multiembed = (len(self.objects) == 2 and
                          all(len(molecule.reactive_indices) >= 2 for molecule in self.objects) and 
                          not cyclical)

            if cyclical or chelotropic or multiembed:

                if cyclical:
                    self.embed = 'cyclical'
                elif multiembed:
                    self.embed = 'multiembed'
                else:
                    self.embed = 'chelotropic'
                    for mol in self.objects:
                        mol.compute_orbitals()
                        for c, _ in enumerate(mol.atomcoords):
                            for index, atom in mol.reactive_atoms_classes_dict[c].items():
                                orb_dim = norm_of(atom.center[0]-atom.coord)
                                atom.init(mol, index, update=True, orb_dim=orb_dim + 0.2, conf=c)
                    # Slightly enlarging orbitals for chelotropic embeds, or they will
                    # be generated a tad too close to each other for how the cyclical embed works          

                self.options.rotation_steps = 9

                if hasattr(self.options, 'custom_rotation_steps'):
                # if user specified a custom value, use it.
                    self.options.rotation_steps = self.options.custom_rotation_steps

                self.systematic_angles = cartesian_product(*[range(self.options.rotation_steps+1) for _ in self.objects]) \
                            * 2*self.options.rotation_range/self.options.rotation_steps - self.options.rotation_range

                if p:
                # avoid calculating pivots if this is an early call 
                    for molecule in self.objects:
                        self._set_pivots(molecule)

            elif string:
                
                self.embed = 'string'
                self.options.rotation_steps = 36

                for mol in self.objects:
                    if not hasattr(mol, 'reactive_atoms_classes_dict'):
                        mol.compute_orbitals()

                if hasattr(self.options, 'custom_rotation_steps'):
                # if user specified a custom value, use it.
                    self.options.rotation_steps = self.options.custom_rotation_steps

                self.systematic_angles = [n * 360 / self.options.rotation_steps for n in range(self.options.rotation_steps)]

            elif multiembed:
                # Complex, unspecified embed type - will explore many possibilities concurrently
                self.embed = 'multiembed' 
   
            else:
                raise InputError(('Bad input - The only molecular configurations accepted are:\n' 
                                  '1) One molecule with two reactive centers (monomolecular embed)\n'
                                  '2) One molecule with four indices(dihedral embed)\n'
                                  '3) Two or three molecules with two reactive centers each (cyclical embed)\n'
                                  '4) Two molecules with one reactive center each (string embed)\n'
                                  '5) Two molecules, one with a single reactive center and the other with two (chelotropic embed)\n'
                                  '6) Two molecules with at least two reactive centers each'))
            
            if p:
            # avoid calculating this if this is an early call 

                self._set_reactive_atoms_cumnums()
                # appending to each reactive atom the cumulative
                # number indexing in the TS context

        else:
            raise InputError('Bad input - could not set up an appropriate embed type (too many structures specified?)')

        if self.options.shrink:
            for molecule in self.objects:
                molecule._scale_orbs(self.options.shrink_multiplier)
                self._set_pivots(molecule)
            self.options.only_refined = True
        # SHRINK - scale orbitals and rebuild pivots

        if self.options.rmsd is None:
            self.options.rmsd = 0.25

        if p:
            self.candidates = self._get_number_of_candidates()
            _s = self.candidates or 'Many'
            self.log(f'--> Setup performed correctly. {_s} candidates will be generated.\n')

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
        
        if self.embed == 'multiembed':
            return 0

        candidates = 2*len(self.systematic_angles)*np.prod([len(mol.atomcoords) for mol in self.objects])
        
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

    def _set_embedder_structures_from_mol(self):
        '''
        Intended for REFINE runs, set the self.structures variable
        (and related) to the confomers of a specific molecuele.
        '''
        self.structures = self.objects[0].atomcoords
        self.atomnos = self.objects[0].atomnos
        self.constrained_indices = _get_monomolecular_reactive_indices(self)
        self.ids = None
        self.energies = np.array([0 for _ in self.structures])
        self.exit_status = np.ones(self.structures.shape[0], dtype=bool)
        self.embed_graph = get_sum_graph([graphize(self.structures[0], self.atomnos)], self.constrained_indices[0])

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

    def _apply_operators(self):
        '''
        Replace molecules in self.objects with
        their post-operator ones.
        '''

        # early call to get the self.embed attribute
        self._setup(p=False)

        # for input_string in self.options.operators:
        for index, operators in self.options.operators_dict.items():

            for operator in operators: 

                input_string = f'{operator}> {self.objects[index].name}'
                outname = operate(input_string, self)
                # operator = input_string.split('>')[0]

                if operator == 'refine':
                    self._set_embedder_structures_from_mol()

                # these operators do not need molecule substitution
                elif operator not in ('pka', 'scan'):

                    # names = [mol.name for mol in self.objects]
                    # filename = self._extract_filename(input_string)
                    # index = names.index(filename)
                    reactive_indices = self.objects[index].reactive_indices

                    # replacing the old molecule with the one post-operators
                    self.objects[index] = Hypermolecule(outname, reactive_indices)

                    # calculating where the new orbitals are
                    self.objects[index].compute_orbitals()

                    # updating orbital size if not default
                    if hasattr(self, 'orb_string'):
                        self._set_custom_orbs(self.orb_string)

                    # updating global docker if necessary
                    if operator in ('rsearch', 'csearch') and self.options.noembed and len(self.objects) == 1:
                        self.structures = self.objects[0].atomcoords
                        self.atomnos = self.objects[0].atomnos
                        self.constrained_indices = _get_monomolecular_reactive_indices(self)
                        self.ids = None
                        self.energies = np.array([0 for _ in self.structures])
                        self.exit_status = np.ones(self.structures.shape[0], dtype=bool)
                        self.embed_graph = get_sum_graph([graphize(self.structures[0], self.atomnos)], self.constrained_indices[0])

        # updating the orbital cumnums for 
        # all the molecules in the run
        self._set_reactive_atoms_cumnums()

        # resetting the attribute
        self.embed = None

    def _extract_filename(self, input_string):
        '''
        Input: 'refine> TSCoDe_unoptimized_comp_check.xyz 5a 36a 0b 43b 33c 60c'
        Output: 'TSCoDe_unoptimized_comp_check.xyz'
        '''
        input_string = input_string.split('>')[-1].lstrip()
        # remove operator and whitespaces after it

        input_string = input_string.split()[0]
        # remove pairing numbers/letters and newline chars

        return input_string

    def _inspect_structures(self):
        '''
        '''

        self.log('--> Structures check requested. Shutting down after last window is closed.\n')

        for mol in self.objects:
            ase_view(mol)
        
        self.logfile.close()
        os.remove(f'TSCoDe_{self.stamp}.log')

        sys.exit()

    def scramble(self, array, sequence):
        return np.array([array[s] for s in sequence])

    def get_pairing_dist_from_letter(self, letter):
        '''
        Get constrained distance between paired reactive
        atoms, accessed via the associated constraint letter
        '''

        if hasattr(self, 'pairing_dists') and self.pairing_dists.get(letter) is not None:
            # if self.options.shrink:
            #     return self.pairing_dists[letter] * self.options.shrink_multiplier
            return self.pairing_dists[letter]

        d = 0
        try:
            for mol_index, mol_pairing_dict in self.pairings_dict.items():
                if r_atom_index := mol_pairing_dict.get(letter):

                    # for refine embeds, one letter corresponds to two indices
                    # on the same molecule
                    if isinstance(r_atom_index, tuple):
                        i1, i2 = r_atom_index
                        return (self.objects[mol_index].get_orbital_length(i1) +
                                self.objects[mol_index].get_orbital_length(i2))

                    # for other runs, it is just one atom per molecule per letter
                    d += self.objects[mol_index].get_orbital_length(r_atom_index)

            return d

        # If no orbitals were built, return None
        except NoOrbitalError:
            return None

    def get_pairing_dists_from_constrained_indices(self, constrained_pair):
        '''
        Returns the constrained distance
        for a specific constrained pair of indices
        '''
        try:
            letter = next(lett for lett, pair in self.pairings_table.items() if (pair[0] == constrained_pair[0] and      
                                                                                 pair[1] == constrained_pair[1]))
            return self.get_pairing_dist_from_letter(letter)

        except StopIteration:
            return None

    def get_pairing_dists(self, conf):
        '''
        Returns a list with the constrained distances for each embedder constraint
        '''
        if self.constrained_indices[conf].size == 0:
            return None

        constraints = np.concatenate([self.constrained_indices[conf], self.internal_constraints]) if len(self.internal_constraints) > 0 else self.constrained_indices[conf]
        return [self.get_pairing_dists_from_constrained_indices(pair) for pair in constraints]

    def write_quote(self):
        '''
        Reads the quote file and writes one in the logfile
        '''
        from tscode.quotes import quotes
        quote, author = random.choice(quotes).values()

        self.log('\n' + auto_newline(quote))

        if author:
            self.log(f'    - {author}\n')

    def run(self):
        '''
        Run the embedding.
        '''
        try:
            RunEmbedding(self).run()

        except Exception as _e:
            logging.exception(_e)
            raise _e
        
    def normal_termination(self):
        '''
        Terminate the run, printing the total time and a quote.

        '''
        self.log(f'\n--> TSCoDe normal termination: total time {time_to_string(time.perf_counter() - self.t_start_run, verbose=True)}.')
        
        self.write_quote()
        self.logfile.close()
        sys.exit()