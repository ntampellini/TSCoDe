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
import time
from copy import deepcopy

import numpy as np

from tscode.ase_manipulations import ase_adjust_spacings, ase_saddle
from tscode.calculators._xtb import xtb_metadyn_augmentation
from tscode.embeds import (cyclical_embed, dihedral_embed, monomolecular_embed,
                           string_embed)
from tscode.errors import MopacReadError, ZeroCandidatesError
from tscode.fast_algebra import norm_of
from tscode.hypermolecule_class import align_structures
from tscode.nci import get_nci
from tscode.optimization_methods import (fitness_check, hyperNEB, opt_iscans,
                                         optimize, prune_enantiomers)
from tscode.parameters import orb_dim_dict
from tscode.python_functions import (compenetration_check, prune_conformers,
                                     scramble)
from tscode.utils import clean_directory, loadbar, time_to_string, write_xyz


class RunEmbedding:
    '''
    Class for running embeds, containing all
    methods to embed and refine structures
    '''

    def __init__(self, docker):
        # for attr, value in vars(docker).items():
        for attr in dir(docker):
            if attr[0:2] != '__':
                setattr(self, attr, getattr(docker, attr))
        self.run()

    def apply_mask(self, attributes, mask):
        '''
        Applies in-place masking of Docker attributes
        '''
        for attr in attributes:
            if hasattr(self, attr):
                new_attr = getattr(self, attr)[mask]
                setattr(self, attr, new_attr)

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

        embed_functions = {
            'chelotropic' : cyclical_embed,
            'cyclical' : cyclical_embed,
            'dihedral' : dihedral_embed,
            'monomolecular' : monomolecular_embed,
            'string' : string_embed,
        }

        if self.options.noembed:
            return

        self.structures = embed_functions[self.embed](self)
        self.atomnos = np.concatenate([molecule.atomnos for molecule in self.objects])
        # cumulative list of atomic numbers associated with coordinates

        self.log(f'Generated {len(self.structures)} transition state candidates ({time_to_string(time.time()-self.t_start_run)})\n')

    def compenetration_refining(self):
        '''
        Performing a sanity check for excessive compenetration
        on generated structures, discarding the ones that look too bad.
        '''

        if self.embed not in ('cyclical', 'monomolecular'):
            
            self.log('--> Checking structures for compenetrations')

            t_start = time.time()
            mask = np.zeros(len(self.structures), dtype=bool)
            num = len(self.structures)
            for s, structure in enumerate(self.structures):
                # if num > 100 and num % 100 != 0 and s % (num % 100) == 99:
                #     loadbar(s, num, prefix=f'Checking structure {s+1}/{num} ')
                mask[s] = compenetration_check(structure, self.ids, max_clashes=self.options.max_clashes, thresh=self.options.clash_thresh)

            # loadbar(1, 1, prefix=f'Checking structure {len(self.structures)}/{len(self.structures)} ')

            self.apply_mask(('structures', 'constrained_indexes'), mask)
            t_end = time.time()

            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for compenetration ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
            else:
                self.log('All structures passed the compenetration check')
            self.log()

        self.zero_candidates_check()

        self.energies = np.zeros(len(self.structures))
        self.exit_status = np.zeros(len(self.structures), dtype=bool)
    
    def fitness_refining(self):
        '''
        Performing a distance check on generated structures, 
        discarding the ones that do not respect the imposed pairings.
        Most of the times, rejects structures that changed their NCIs
        from the imposed ones to other combinations.
        '''
        self.log('--> Fitness pruning - removing inaccurate structures')

        mask = np.zeros(len(self.structures), dtype=bool)
        
        for s, structure in enumerate(self.structures):
            mask[s] = fitness_check(self, structure)

        attr = (
            'structures',
            'energies',
            'constrained_indexes',
            'exit_status',
        )

        self.apply_mask(attr, mask)

        if False in mask:
            self.log(f'Discarded {len([b for b in mask if not b])} candidates for unfitness ({len([b for b in mask if b])} left)')
        else:
            self.log('All candidates meet the imposed criteria.')
        self.log()

        self.zero_candidates_check()

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

        if not self.options.keep_enantiomers:

            self.structures, mask = prune_enantiomers(self.structures, self.atomnos)
            
            self.apply_mask(attr, mask)
           
            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} enantiomeric structures ({len([b for b in mask if b])} left, {time_to_string(time.time()-t_start)})')
            self.log()

        t_start = time.time()
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

        if verbose:
            self.log(f'    - similarity final processing (k=1) - {time_to_string(time.time()-t_start_int)} - kept {len([b for b in mask if b])}/{len(mask)}')

        self.apply_mask(attr, mask)

        if before > len(self.structures):
            self.log(f'Discarded {int(before - len([b for b in mask if b]))} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(time.time()-t_start)})')

    def force_field_refining(self):
        '''
        Performs structural optimizations with the docker force field caculator.
        Only structures that do not scramble during FF optimization are updated,
        while the rest are kept as they are.
        '''

        ################################################# CHECKPOINT BEFORE MM OPTIMIZATION

        self.outname = f'TSCoDe_checkpoint_{self.stamp}.xyz'
        with open(self.outname, 'w') as f:        
            for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):
                write_xyz(structure, self.atomnos, f, title=f'TS candidate {i+1} - Checkpoint before MM optimization')
        self.log(f'--> Checkpoint output - Wrote {len(self.structures)} TS structures to {self.outname} file before MM optimization.\n')

        ################################################# GEOMETRY OPTIMIZATION - FORCE FIELD

        self.log(f'--> Structure optimization ({self.options.ff_level} level via {self.options.ff_calc})')
        t_start = time.time()

        for i, structure in enumerate(deepcopy(self.structures)):
            loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')
            try:
                new_structure, _, self.exit_status[i] = optimize(structure,
                                                                    self.atomnos,
                                                                    self.options.ff_calc,
                                                                    method=self.options.ff_level,
                                                                    constrained_indexes=self.constrained_indexes[i],
                                                                    mols_graphs=self.graphs)

                if self.exit_status[i]:
                    self.structures[i] = new_structure

            except Exception as e:
                raise e

        loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')
        t_end = time.time()
        self.log(f'Force Field {self.options.ff_level} optimization took {time_to_string(t_end-t_start)} (~{time_to_string((t_end-t_start)/len(self.structures))} per structure)')
        
        ################################################# EXIT STATUS

        self.log(f'Successfully pre-refined {len([b for b in self.exit_status if b])}/{len(self.structures)} candidates at {self.options.ff_level} level.')
        
        ################################################# PRUNING: SIMILARITY (POST FORCE FIELD OPT)

        self.zero_candidates_check()
        self.similarity_refining()

        ################################################# CHECKPOINT BEFORE MOPAC/ORCA OPTIMIZATION

        with open(self.outname, 'w') as f:        
            for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):
                exit_str = f'{self.options.ff_level} REFINED' if self.exit_status[i] else 'RAW'
                write_xyz(structure, self.atomnos, f, title=f'TS candidate {i+1} - {exit_str} - Checkpoint before {self.options.calculator} optimization')
        self.log(f'--> Checkpoint output - Updated {len(self.structures)} TS structures to {self.outname} file before {self.options.calculator} optimization.\n')

    def _set_target_distances(self):
        '''
        Called before TS refinement to compute all
        target bonding distances. These are only returned
        if that pairing is not a non-covalent interaction,
        that is if pairing was not specified with letters
        "x", "y" or "z".
        '''
        self.target_distances = {}

        r_atoms = {}
        for mol in self.objects:
            for letter, r_atom in mol.reactive_atoms_classes_dict[0].items():
                if letter not in ("x", "y", "z"):
                    r_atoms[r_atom.cumnum] = r_atom

        pairings = self.constrained_indexes.ravel()
        pairings = pairings.reshape(int(pairings.shape[0]/2), 2)
        pairings = {tuple(sorted((a,b))) for a, b in pairings}

        active_pairs = [indexes for letter, indexes in self.pairings_table.items() if letter not in ("x", "y", "z")]

        for index1, index2 in pairings:

            if [index1, index2] in active_pairs:

                if hasattr(self, 'pairings_dists'):
                    letter = list(self.pairings_table.keys())[active_pairs.index([index1, index2])]

                    if letter in self.pairings_dists:
                        self.target_distances[(index1, index2)] = self.pairings_dists[letter]
                        continue
                # if target distance has been specified by user, read that, otherwise compute it

                r_atom1 = r_atoms[index1]
                r_atom2 = r_atoms[index2]

                dist1 = orb_dim_dict.get(r_atom1.symbol + ' ' + str(r_atom1), orb_dim_dict['Fallback'])
                dist2 = orb_dim_dict.get(r_atom2.symbol + ' ' + str(r_atom2), orb_dim_dict['Fallback'])

                self.target_distances[(index1, index2)] = dist1 + dist2

    def optimization_refining(self):
        '''
        Refines structures by constrained optimizations with the active calculator,
        discarding similar ones and scrambled ones.
        '''

        t_start = time.time()

        self.log(f'--> Structure optimization ({self.options.theory_level} level via {self.options.calculator})')

        if self.options.calculator == 'MOPAC':
            method = f'{self.options.theory_level} GEO-OK CYCLES=500'

        else:
            method = f'{self.options.theory_level}'

        for i, structure in enumerate(deepcopy(self.structures)):
            loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')
            try:
                t_start_opt = time.time()
                new_structure, self.energies[i], self.exit_status[i] = optimize(structure,
                                                                                    self.atomnos,
                                                                                    self.options.calculator,
                                                                                    method=method,
                                                                                    constrained_indexes=self.constrained_indexes[i],
                                                                                    mols_graphs=self.graphs,
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

        if not self.options.noembed:

            self.write_structures('TS_guesses_unrefined', energies=False, p=False)
            self.log(f'--> Checkpoint output - Updated {len(self.structures)} TS structures before distance refinement.\n')

            self.log(f'--> Refining bonding distances for TSs ({self.options.theory_level} level)')

            if self.options.ff_opt:
                try:
                    os.remove(f'TSCoDe_checkpoint_{self.stamp}.xyz')
                    # We don't need the pre-optimized structures anymore
                except FileNotFoundError:
                    pass

            self._set_target_distances()
            t_start = time.time()

            for i, structure in enumerate(deepcopy(self.structures)):
                loadbar(i, len(self.structures), prefix=f'Refining structure {i+1}/{len(self.structures)} ')
                try:

                    traj = f'refine_{i}.traj' if self.options.debug else None

                    new_structure, new_energy, self.exit_status[i] = ase_adjust_spacings(self,
                                                                                            structure,
                                                                                            self.atomnos,
                                                                                            self.constrained_indexes[i],
                                                                                            title=i,
                                                                                            traj=traj
                                                                                            )

                    if self.exit_status[i]:
                        self.structures[i] = new_structure
                        self.energies[i] = new_energy
                                                                                                                            
                except ValueError as e:
                    # ase will throw a ValueError if the output lacks a space in the "FINAL POINTS AND DERIVATIVES" table.
                    # This occurs when one or more of them is not defined, that is when the calculation did not end well.
                    # The easiest solution is to reject the structure and go on.
                    self.log(repr(e))
                    self.log(f'Failed to read MOPAC file for Structure {i+1}, skipping distance refinement', p=False)                                    
            
            loadbar(1, 1, prefix=f'Refining structure {i+1}/{len(self.structures)} ')
            t_end = time.time()
            self.log(f'{self.options.calculator} {self.options.theory_level} refinement took {time_to_string(t_end-t_start)} (~{time_to_string((t_end-t_start)/len(self.structures))} per structure)')

            before = len(self.structures)
            if self.options.only_refined:

                mask = self.exit_status
                self.apply_mask(('structures', 'energies', 'exit_status', 'constrained_indexes'), mask)

                s = f'Discarded {len([i for i in mask if not i])} unrefined structures.'

            else:
                s = 'Non-refined ones will not be discarded.'


            self.log(f'Successfully refined {len([i for i in self.exit_status if i])}/{before} structures. {s}')

            ################################################# PRUNING: SIMILARITY (POST REFINEMENT)

            self.zero_candidates_check()
            self.similarity_refining()

            ################################################# PRUNING: FITNESS 

            self.fitness_refining()

        ################################################# PRUNING: ENERGY

        self.energies = self.energies - np.min(self.energies)
        _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
        self.energies = self.scramble(self.energies, sequence)
        self.structures = self.scramble(self.structures, sequence)
        self.constrained_indexes = self.scramble(self.constrained_indexes, sequence)
        # sorting structures based on energy

        if self.options.kcal_thresh is not None:
    
            mask = (self.energies - np.min(self.energies)) < self.options.kcal_thresh

            self.apply_mask(('structures', 'energies', 'exit_status'), mask)

            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for energy (Threshold set to {self.options.kcal_thresh} kcal/mol)')
        
        ################################################# XYZ GUESSES OUTPUT 

        self.outname = f'TSCoDe_TS_guesses_{self.stamp}.xyz'
        with open(self.outname, 'w') as f:        
            for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):

                kind = 'REFINED - ' if self.exit_status[i] else 'NOT REFINED - '

                write_xyz(structure, self.atomnos, f, title=f'Structure {i+1} - {kind}Rel. E. = {round(self.energies[i], 3)} kcal/mol')

        try:
            os.remove(f'TSCoDe_TS_guesses_unrefined_{self.stamp}.xyz')
            # since we have the refined structures, we can get rid of the unrefined ones
        except FileNotFoundError:
            pass

        self.log(f'Wrote {len(self.structures)} rough TS structures to {self.outname} file.\n')

    def metadynamics_augmentation(self):
        '''
        Runs a metadynamics simulation (MTD) through
        the XTB program for each structure in self.structure.
        New structures are obtained from the simulations, minimized
        in energy and added to self. structures.
        '''

        self.log(f'--> Performing XTB Metadynamic augmentation of TS candidates')

        before = len(self.structures)
        t_start_run = time.time()

        for s, (structure, constrained_indexes) in enumerate(zip(deepcopy(self.structures), deepcopy(self.constrained_indexes))):

            loadbar(s, before, f'Running MTD {s+1}/{before} ')
            t_start = time.time()

            new_structures = xtb_metadyn_augmentation(structure,
                                                      self.atomnos,
                                                      constrained_indexes=constrained_indexes,
                                                      new_structures=5,
                                                      title=s)

            self.structures = np.concatenate((self.structures, new_structures))
            self.energies = np.concatenate((self.energies, [0 for _ in new_structures]))
            self.constrained_indexes = np.concatenate((self.constrained_indexes, [constrained_indexes for _ in new_structures]))
        
            self.log(f'   - Structure {s+1} - {len(new_structures)} new conformers ({time_to_string(time.time()-t_start)})', p=False)

        loadbar(before, before, f'Running MTD {before}/{before} ')
        self.exit_status = np.array([True for _ in self.structures], dtype=bool)

        self.log(f'Metadynamics augmentation completed - found {len(self.structures)-before} new conformers ({time_to_string(time.time()-t_start_run)})\n')

    def hyperneb_refining(self):
        '''
        Performs a clibing-image NEB calculation inferring reagents and products for each structure.
        '''
        self.log(f'--> HyperNEB optimization ({self.options.theory_level} level)')
        t_start = time.time()

        for i, structure in enumerate(self.structures):

            loadbar(i, len(self.structures), prefix=f'Performing NEB {i+1}/{len(self.structures)} ')

            t_start_opt = time.time()

            try:

                self.structures[i], self.energies[i], self.exit_status[i] = hyperNEB(self,
                                                                                        structure,
                                                                                        self.atomnos,
                                                                                        self.ids,
                                                                                        self.constrained_indexes[i],
                                                                                        title=f'structure_{i+1}')

                exit_str = 'COMPLETED' if self.exit_status[i] else 'CRASHED'

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
        self.apply_mask(('structures', 'energies', 'exit_status'), mask)

        ################################################# PRUNING: SIMILARITY (POST NEB)

        if len(self.structures) != 0:

            t_start = time.time()
            self.structures, mask = prune_conformers(self.structures, self.atomnos, max_rmsd=self.options.pruning_thresh)
            self.energies = self.energies[mask]
            t_end = time.time()
            
            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
            self.log()

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

    def saddle_refining(self):
        '''
        Performs a first order saddle optimization for each structure.
        '''
        self.log(f'--> Saddle optimization ({self.options.theory_level} level)')
        t_start = time.time()

        for i, structure in enumerate(self.structures):

            loadbar(i, len(self.structures), prefix=f'Performing saddle opt {i+1}/{len(self.structures)} ')

            try:

                self.structures[i], self.energies[i], self.exit_status[i] = ase_saddle(self,
                                                                            structure,
                                                                            self.atomnos,
                                                                            self.constrained_indexes[i],
                                                                            mols_graphs=self.graphs if self.embed != 'monomolecular' else None,
                                                                            title=f'Saddle opt - Structure {i+1}',
                                                                            logfile=self.logfile,
                                                                            traj=f'Saddle_opt_{i+1}.traj',
                                                                            maxiterations=200)

            except ValueError:
                # Thrown when an ASE read fails (during saddle opt)
                self.exit_status[i] = False

        loadbar(1, 1, prefix=f'Performing saddle opt {len(self.structures)}/{len(self.structures)} ')
        t_end = time.time()
        self.log(f'{self.options.calculator} {self.options.theory_level} saddle optimization took {time_to_string(t_end-t_start)} ({time_to_string((t_end-t_start)/len(self.structures))} per structure)')
        self.log(f'Saddle opt completed for {len([i for i in self.exit_status if i])}/{len(self.structures)} structures')

        mask = self.exit_status

        self.apply_mask(('structures', 'energies', 'exit_status'), mask)

        ################################################# PRUNING: SIMILARITY (POST SADDLE OPT)

        if len(self.structures) != 0:

            t_start = time.time()
            self.structures, mask = prune_conformers(self.structures, self.atomnos, max_rmsd=self.options.pruning_thresh)
            self.apply_mask(('energies', 'exit_status'), mask)
            t_end = time.time()
            
            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
            self.log()

        ################################################# SADDLE OPT EXTRA XYZ OUTPUT

            self.energies -= np.min(self.energies)
            _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
            self.energies = scramble(self.energies, sequence)
            self.structures = scramble(self.structures, sequence)
            self.constrained_indexes = scramble(self.constrained_indexes, sequence)
            # sorting structures based on energy

            self.outname = f'TSCoDe_SADDLE_TSs_{self.stamp}.xyz'
            with open(self.outname, 'w') as f:        
                for i, structure in enumerate(align_structures(self.structures, self.constrained_indexes[0])):
                    write_xyz(structure, self.atomnos, f, title=f'Structure {i+1} - TS - Rel. E. = {round(self.energies[i], 3)} kcal/mol')

            self.log(f'Wrote {len(self.structures)} saddle-optimized structures to {self.outname} file\n')

        else:
            self.log()

    def independent_scans_refining(self):
        '''
        Performs independent scans optimization for each structure.
        '''
        self.log(f'--> Performing independent scans refinement ({self.options.theory_level} level)')
        t_start = time.time()

        for i, structure in enumerate(self.structures):

            loadbar(i, len(self.structures), prefix=f'Refining structure {i+1}/{len(self.structures)} ')

            try:

                self.structures[i], self.energies[i], self.exit_status[i] = opt_iscans(self,
                                                                                        structure,
                                                                                        self.atomnos,
                                                                                        title=f'Structure {i+1}',
                                                                                        logfile=self.logfile,
                                                                                        xyztraj=f'IScan_{i+1}.xyz' if self.options.debug else None
                                                                                        )

            except (ValueError, MopacReadError):
                # Thrown when an ASE or MOPAC read fails (during scan opt)
                self.exit_status[i] = False

        loadbar(1, 1, prefix=f'Refining structure {len(self.structures)}/{len(self.structures)} ')
        t_end = time.time()
        self.log(f'{self.options.calculator} {self.options.theory_level} independent scans took {time_to_string(t_end-t_start)} ({time_to_string((t_end-t_start)/len(self.structures))} per structure)')
        self.log(f'Independent scans refinement completed for {len([i for i in self.exit_status if i])}/{len(self.structures)} structures')

        mask = self.exit_status
        self.apply_mask(('structures', 'energies', 'exit_status'), mask)


        ################################################# PRUNING: SIMILARITY (POST ISCANS OPT)

        if len(self.structures) != 0:

            t_start = time.time()
            self.structures, mask = prune_conformers(self.structures, self.atomnos, max_rmsd=self.options.pruning_thresh)
            self.apply_mask(('energies', 'exit_status'), mask)
            t_end = time.time()
            
            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
            self.log()

        else:
            self.log('No candidates successfully refined with the independent scans approach. Terminating.\n')
            self.normal_termination()

    def print_nci(self):
        '''
        Prints and logs the non-covalent interactions guesses for final structures.
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
        
        if not [l for l in self.nci if l != []]:
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

    def write_header(self):
        '''
        Writes information about the TSCoDe parameters used in the calculation.
        '''

        if self.embed != 'dihedral':

            head = ''
            for i, mol in enumerate(self.objects):
                descs = [atom.symbol+'('+str(atom)+f', {round(norm_of(atom.center[0]-atom.coord), 3)} A, ' +
                        f'{len(atom.center)} center{"s" if len(atom.center) != 1 else ""})' for atom in mol.reactive_atoms_classes_dict[0].values()]

                t = '\n        '.join([(str(index) + ' ' if len(str(index)) == 1 else str(index)) + ' -> ' + desc for index, desc in zip(mol.reactive_indexes, descs)])
               
                mol_line = f' -> {len(mol.atomcoords[0])} atoms, {len(mol.atomcoords)} conformer{"s" if len(mol.atomcoords) != 1 else ""}'
                if hasattr(mol, 'pivots'):
                    mol_line += f', {len(mol.pivots[0])} pivot{"s" if len(mol.pivots[0]) != 1 else ""}'

                    if mol.sp3_sigmastar:
                        mol_line += ', sp3_sigmastar'

                    if any(mol.sigmatropic):
                        mol_line += ', sigmatropic'
                        if all(mol.sigmatropic):
                            mol_line += ' (all conformers)'
                        else:
                            mol_line += ' (some conformers)'

                head += f'\n    {i+1}. {mol.name}{mol_line}\n        {t}\n'

            self.log('--> Input structures & reactive indexes data:\n' + head)

        self.log(f'--> Calculation options used were:')
        for line in str(self.options).split('\n'):

            if self.embed in ('monomolecular','string') and line.split()[0] in ('rotation_range', 'rigid', 'suprafacial'):
                continue

            if self.embed == 'dihedral' and line.split()[0] not in ('optimization',
                                                                    'calculator',
                                                                    'theory_level',
                                                                    'kcal_thresh',
                                                                    'debug',
                                                                    'pruning_thresh',
                                                                    'neb',
                                                                    'saddle'):
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
                'mol new {%s.xyz}\n' % (os.path.join(os.getcwd() + self.vmd_name).rstrip('.vmd')) +
                'mol selection index %s\n' % (' '.join([str(i) for i in indexes])) +
                'mol representation CPK 0.7 0.5 50 50\n' +
                'mol color ColorID 7\n' +
                'mol material Transparent\n' +
                'mol addrep top\n')

            for a, b in self.pairings_table.values():
                s += f'label add Bonds 0/{a} 0/{b}\n'

            f.write(s)

        self.log(f'Wrote VMD {self.vmd_name} file\n')

    def run(self):
        '''
        Run the TSCoDe program.
        '''

        if not self.options.let and hasattr(self, 'candidates'):
            assert self.candidates < 1e8, ('ATTENTION! This calculation is probably going to be very big. To ignore this message'
                                           ' and proceed, add the LET keyword to the input file.')

        self.write_header()
        self.t_start_run = time.time()

        try: # except KeyboardInterrupt
            try: # except ZeroCandidatesError()
                self.generate_candidates()
                
                if self.options.bypass:
                    self.write_structures('unoptimized', energies=False)
                    self.normal_termination()

                self.compenetration_refining()
                self.similarity_refining(verbose=True)

                if self.options.optimization:

                    if self.options.ff_opt:
                        self.force_field_refining()

                    self.optimization_refining()

            except ZeroCandidatesError:
                t_end_run = time.time()
                s = ('    Sorry, the program did not find any reasonable TS structure. Are you sure the input indexes and pairings were correct? If so, try these tips:\n'
                     '    - If no structure passes the compenetration check, the SHRINK keyword may help (see documentation).\n'
                     '    - Similarly, enlarging the spacing between atom pairs with the DIST keyword facilitates the embed.\n'
                     '    - If no structure passes the fitness check, try adding a solvent with the SOLVENT keyword.\n'
                     '    - Impose less strict compenetration rejection criteria with the CLASHES keyword.\n'
                     '    - Generate more structures with higher STEPS and ROTRANGE values.\n'
                )

                self.log(f'\n--> Program termination: No candidates found - Total time {time_to_string(t_end_run-self.t_start_run)}')
                self.log(s)
                self.logfile.close()
                clean_directory()
                quit()

            ##################### AUGMENTATION - METADYNAMICS

            if self.options.metadynamics:

                self.metadynamics_augmentation()
                self.optimization_refining()
                self.similarity_refining()

            ##################### POST TSCODE - SADDLE, NEB, NCI, VMD

            if self.options.optimization and not self.options.bypass:
                self.write_vmd()

            if self.options.neb:
                self.hyperneb_refining()

            if self.options.saddle:
                self.saddle_refining()

            if self.options.ts:
                self.independent_scans_refining()
                self.saddle_refining()
                
            if self.options.nci and self.options.optimization:
                self.print_nci()
            
            self.normal_termination()

            #### EXTRA
            
            # if self.options.debug:
            #     path = os.path.join(os.getcwd(), self.vmd_name)
            #     check_call(f'vmd -e {path}'.split())

            ################################################

        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt requested by user. Quitting.')
            quit()
