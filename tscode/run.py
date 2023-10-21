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
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
from rmsd import kabsch_rmsd

from tscode.algebra import norm_of
from tscode.ase_manipulations import ase_saddle
from tscode.calculators._xtb import (xtb_metadyn_augmentation, xtb_opt,
                                     xtb_pre_opt)
from tscode.embeds import (cyclical_embed, dihedral_embed, monomolecular_embed,
                           string_embed)
from tscode.errors import (MopacReadError, NoOrbitalError, SegmentedGraphError,
                           ZeroCandidatesError)
from tscode.graph_manipulations import get_sum_graph
from tscode.hypermolecule_class import align_structures
from tscode.multiembed import multiembed_dispatcher
from tscode.nci import get_nci
from tscode.optimization_methods import (fitness_check, hyperNEB,
                                         opt_funcs_dict, opt_iscans,
                                         prune_by_moment_of_inertia)
from tscode.parameters import orb_dim_dict
from tscode.pt import pt
from tscode.python_functions import (compenetration_check,
                                     prune_conformers_rmsd,
                                     prune_conformers_tfd, scramble)
from tscode.torsion_module import (_get_quadruplets, csearch,
                                   prune_conformers_rmsd_rot_corr)
from tscode.utils import (clean_directory, loadbar, scramble_check,
                          time_to_string, timing_wrapper, write_xyz)


class RunEmbedding:
    '''
    Class for running embeds, containing all
    methods to embed and refine structures
    '''

    def __init__(self, embedder):
        '''
        Copying all functions and attributes 
        of the previous embedder.
        '''
        for attr in dir(embedder):
            if attr[0:2] != '__' and attr != 'run':
                setattr(self, attr, getattr(embedder, attr))

    def rel_energies(self):
        return self.energies - np.min(self.energies)

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
            'multiembed' : multiembed_dispatcher,
        }

        if self.embed == 'refine':
            self.log('\n')
            return

        # Embed structures and assign them to self.structures
        self.structures = embed_functions[self.embed](self)

        # cumulative list of atomic numbers associated with coordinates
        self.atomnos = np.concatenate([molecule.atomnos for molecule in self.objects])

        # Build the embed graph. This will be used as a future reference.
        # Note that the use of the first constrained_indices pair is irrelevant
        # for the torsion fingerprint outcome, but other future features might
        # rely on the embed_graph to be accurate if conformers have different
        # constrained indices.

        additional_bonds = self.constrained_indices[0]
        if len(self.internal_constraints) > 0:
            additional_bonds = np.concatenate((self.internal_constraints, additional_bonds))

        self.embed_graph = get_sum_graph(self.graphs, additional_bonds)
        
        self.log(f'Generated {len(self.structures)} transition state candidates ({time_to_string(time.perf_counter()-self.t_start_run)})\n')

        # if self.options.debug:
        self.write_structures('embedded', energies=False)

    def dump_status(self, outname):
        '''
        Writes structures and energies to [outname].xyz
        and [outname].dat to help debug the current run.
                
        '''
        with open(f'{outname}_energies.dat', 'w') as _f:
            for i, energy in enumerate(self.energies):
                print_energy = str(round(energy-np.min(self.energies), 2))+' kcal/mol' if energy != 1E10 else 'SCRAMBLED'
                _f.write('Candidate {:5} : {}\n'.format(i, print_energy))

        with open(f'{outname}_structures.xyz', 'w') as _f:        
            for i, (structure, status, energy) in enumerate(zip(align_structures(self.structures),
                                                                self.exit_status,
                                                                self.rel_energies())):

                kind = 'REFINED - ' if status else 'NOT REFINED - '
                write_xyz(structure, self.atomnos, _f, title=f'Structure {i+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({self.options.ff_level})')

        with open(f'{outname}_constraints.dat', 'w') as _f:
            for i, constraints in enumerate(self.constrained_indices):
                _f.write('Candidate {:5} : {}\n'.format(i, constraints))

        with open(f'{outname}_runembedding.pickle', 'wb') as _f:
            d = {
                'structures' : self.structures,
                'energies' : self.energies,
                'constrained_indices' : self.constrained_indices,
                'graphs' : self.graphs,
                'objects' : self.objects,
                'options' : self.options,
            }
            pickle.dump(d, _f)

    def compenetration_refining(self):
        '''
        Performing a sanity check for excessive compenetration
        on generated structures, discarding the ones that look too bad.
        '''

        if self.embed not in ('string', 'cyclical', 'monomolecular'):
        # these do not need compenetration refining: the 
        # algorithm checks for compenetrations when embedding
            
            self.log('--> Checking structures for compenetrations')

            t_start = time.perf_counter()
            mask = np.zeros(len(self.structures), dtype=bool)
            # num = len(self.structures)
            for s, structure in enumerate(self.structures):
                # if num > 100 and num % 100 != 0 and s % (num % 100) == 99:
                #     loadbar(s, num, prefix=f'Checking structure {s+1}/{num} ')
                mask[s] = compenetration_check(structure, self.ids, max_clashes=self.options.max_clashes, thresh=self.options.clash_thresh)

            # loadbar(1, 1, prefix=f'Checking structure {len(self.structures)}/{len(self.structures)} ')

            self.apply_mask(('structures', 'constrained_indices'), mask)
            t_end = time.perf_counter()

            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for compenetration ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
            else:
                self.log(f'All {len(mask)} structures passed the compenetration check')
            self.log()

            self.zero_candidates_check()

        # initialize embedder values for the active structures
        # that survived the compenetration check
        self.energies = np.full(len(self.structures), 1E10)
        self.exit_status = np.zeros(len(self.structures), dtype=bool)
    
    def fitness_refining(self, threshold=5, verbose=False):
        '''
        Performing a distance check on generated structures, 
        discarding the ones that do not respect the imposed pairings.
        Internal constraints are ignored.

        threshold : rejection happens when the sum of the deviations from the
        intended spacings is greater than threshold.

        '''
        if verbose:
            self.log(' \n--> Fitness pruning - removing inaccurate structures')

        mask = np.ones(len(self.structures), dtype=bool)
        
        for s, (structure, constraints) in enumerate(zip(self.structures, self.constrained_indices)):

            constrained_distances = tuple(self.get_pairing_dists_from_constrained_indices(_c) for _c in constraints)

            mask[s] = fitness_check(structure,
                                    constraints,
                                    constrained_distances,
                                    threshold=threshold)

        attr = (
            'structures',
            'energies',
            'constrained_indices',
            'exit_status',
        )

        self.apply_mask(attr, mask)

        if False in mask:
            self.log(f'Discarded {len([b for b in mask if not b])} candidates for unfitness ({len([b for b in mask if b])} left)')
        else:
            if verbose:
                self.log('All candidates meet the imposed criteria.')
        self.log()

        self.zero_candidates_check()

    def similarity_refining(self, tfd=True, moi=True, rmsd=True, verbose=False):
        '''
        If possible, removes structures with similar torsional profile (TFD-based).
        Removes structures that are too similar to each other (RMSD-based).
        '''

        if verbose:
            self.log('--> Similarity Processing')

        before = len(self.structures)
        attr = ('constrained_indices', 'energies', 'exit_status')

        if hasattr(self, 'embed_graph') and tfd:

            t_start = time.perf_counter()

            quadruplets = _get_quadruplets(self.embed_graph)
            if len(quadruplets) > 0:
                self.structures, mask = prune_conformers_tfd(self.structures, quadruplets, verbose=verbose)
                
            self.apply_mask(attr, mask)
            
            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} structures for TFD similarity ({len([b for b in mask if b])} left, {time_to_string(time.perf_counter()-t_start)})')

        if moi:

            if len(self.structures) <= 500:

                ### Now again, based on the moment of inertia

                before3 = len(self.structures)

                t_start = time.perf_counter()
                self.structures, mask = prune_by_moment_of_inertia(self.structures, self.atomnos)

                self.apply_mask(attr, mask)

                if before3 > len(self.structures):
                    self.log(f'Discarded {int(len([b for b in mask if not b]))} candidates for MOI similarity ({len([b for b in mask if b])} left, {time_to_string(time.perf_counter()-t_start)})')

        if rmsd and len(self.structures) <= 1E4:

            before1 = len(self.structures)

            t_start = time.perf_counter()
            self.structures, mask = prune_conformers_rmsd(self.structures, self.atomnos, max_rmsd=self.options.rmsd, verbose=verbose)

            self.apply_mask(attr, mask)

            if before1 > len(self.structures):
                self.log(f'Discarded {int(len([b for b in mask if not b]))} candidates for RMSD similarity ({len([b for b in mask if b])} left, {time_to_string(time.perf_counter()-t_start)})')

            ### Second step: again but symmetry-corrected (unless we have too many structures)

            if len(self.structures) <= 500:

                before2 = len(self.structures)

                t_start = time.perf_counter()
                self.structures, mask = prune_conformers_rmsd_rot_corr(self.structures, self.atomnos, self.embed_graph, max_rmsd=self.options.rmsd, verbose=verbose, logfunction=(self.log if verbose else None))

                self.apply_mask(attr, mask)

                if before2 > len(self.structures):
                    self.log(f'Discarded {int(len([b for b in mask if not b]))} candidates for symmetry-corrected RMSD similarity ({len([b for b in mask if b])} left, {time_to_string(time.perf_counter()-t_start)})')


        if verbose and len(self.structures) == before:
            self.log(f'All structures passed the similarity check.{" "*15}')

        self.log()

    def force_field_refining(self, conv_thr="tight", only_fixed_constraints=False, prevent_scrambling=False):
        '''
        Performs structural optimizations with the embedder force field caculator.
        Only structures that do not scramble during FF optimization are updated,
        while the rest are kept as they are.
        conv_thr: convergence threshold, passed to calculator
        only_fixed_constraints: only uses fixed (a, b, c) constraints in optimization
        prevent_scrambling: preserves molecular identities constraining bonds present in graphs (XTB only)
        '''

        ################################################# CHECKPOINT BEFORE FF OPTIMIZATION

        if not only_fixed_constraints:
            self.outname = f'TSCoDe_checkpoint_{self.stamp}.xyz'
            with open(self.outname, 'w') as f:        
                for i, structure in enumerate(align_structures(self.structures)):
                    write_xyz(structure, self.atomnos, f, title=f'TS candidate {i+1} - Checkpoint before FF optimization')
            self.log(f'\n--> Checkpoint output - Wrote {len(self.structures)} unoptimized structures to {self.outname} file before FF optimization.\n')

        ################################################# GEOMETRY OPTIMIZATION - FORCE FIELD

        task = 'Distance refining' if only_fixed_constraints else f'Structure {"pre-" if prevent_scrambling else ""}optimization'
        self.log(f'--> {task} ({self.options.ff_level}{f"/{self.options.solvent}" if self.options.solvent is not None else ""} level via {self.options.ff_calc}, {int(self.avail_cpus/2)} thread{"s" if int(self.avail_cpus/2)>1 else ""})')

        t_start_ff_opt = time.perf_counter()

        processes = []
        cum_time = 0

        opt_function = xtb_pre_opt if prevent_scrambling else xtb_opt

        # Running half as many threads as we have procs (2 proc/thread)
        # since FF does not parallelize well with more cores per struc
        with ProcessPoolExecutor(max_workers=int(self.avail_cpus/2)) as executor:

            for i, structure in enumerate(deepcopy(self.structures)):

                if only_fixed_constraints:
                    constraints = np.array([value for key, value in self.pairings_table.items() if key in ('a', 'b', 'c')])
                
                else:
                    constraints = np.concatenate([self.constrained_indices[i], self.internal_constraints]) if len(self.internal_constraints) > 0 else self.constrained_indices[i]

                pairing_dists = [self.get_pairing_dists_from_constrained_indices(_c) for _c in constraints]

                process = executor.submit(
                                            timing_wrapper,
                                            opt_function,
                                            structure,
                                            self.atomnos,
                                            graphs=self.graphs,
                                            calculator=self.options.ff_calc,
                                            method=self.options.ff_level,
                                            solvent=self.options.solvent,
                                            charge=self.options.charge,
                                            maxiter=None,
                                            conv_thr=conv_thr,
                                            constrained_indices=constraints,
                                            constrained_distances=pairing_dists,
                                            procs=2, # FF just needs two per structure
                                            title=f'Candidate_{i+1}',
                                            spring_constant=0.2 if prevent_scrambling else 1000,
                                        )
                processes.append(process)
          
            for i, process in enumerate(as_completed(processes)):
                        
                loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')

                (new_structure, new_energy, self.exit_status[i]), t_struct = process.result()
                
                # assert that the structure did not scramble during optimization
                if self.exit_status[i]:
                    constraints = (np.concatenate([self.constrained_indices[i], self.internal_constraints])
                                   if len(self.internal_constraints) > 0
                                   else self.constrained_indices[i])
                    self.exit_status[i] = scramble_check(new_structure,
                                                        self.atomnos,
                                                        excluded_atoms=constraints.ravel(),
                                                        mols_graphs=self.graphs,
                                                        max_newbonds=0)
                    
                cum_time += t_struct

                if self.options.debug:
                    exit_status = 'REFINED  ' if self.exit_status[i] else 'SCRAMBLED'
                    self.log(f'    - Candidate_{i+1} - {exit_status} {time_to_string(t_struct, digits=3)}', p=False)
                
                self.structures[i] = new_structure
                if self.exit_status[i] and new_energy is not None:
                    self.energies[i] = new_energy

                else:
                    self.energies[i] = 1E10

                ### Update checkpoint every 20 optimized structures, and give an estimate of the remaining time

                if i % 20 == 19:

                    with open(self.outname, 'w') as f:        
                        for j, (structure, status, energy) in enumerate(zip(align_structures(self.structures),
                                                                            self.exit_status,
                                                                            self.rel_energies())):

                            kind = 'REFINED - ' if status else 'NOT REFINED - '
                            write_xyz(structure, self.atomnos, f, title=f'Structure {j+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({self.options.ff_level})')

                    elapsed = time.perf_counter() - t_start_ff_opt
                    average = (elapsed)/(i+1)
                    time_left = time_to_string((average) * (len(self.structures)-i-1))
                    speedup = cum_time/elapsed
                    self.log(f'    - Optimized {i+1:>4}/{len(self.structures):>4} structures - updated checkpoint file (avg. {time_to_string(average)}/struc, {round(speedup, 1)}x speedup, est. {time_left} left)', p=False)
        
        loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')

        elapsed = time.perf_counter() - t_start_ff_opt
        average = (elapsed)/(len(self.structures))
        speedup = cum_time/elapsed

        self.log(f'{self.options.ff_calc}/{self.options.ff_level} optimization took {time_to_string(elapsed)} (~{time_to_string(average)} per structure, {round(speedup, 1)}x speedup)')
        
        ################################################# EXIT STATUS

        self.log(f'Successfully optimized {len([b for b in self.exit_status if b])}/{len(self.structures)} candidates at {self.options.ff_level} level.')
        
        ################################################# PRUNING: ENERGY

        _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
        self.energies = self.scramble(self.energies, sequence)
        self.structures = self.scramble(self.structures, sequence)
        self.constrained_indices = self.scramble(self.constrained_indices, sequence)
        # sorting structures based on energy

        if self.options.debug:
            self.dump_status(f'force_field_refining_{conv_thr}')

        if self.options.kcal_thresh is not None and only_fixed_constraints:
    
            mask = self.rel_energies() < 2*self.options.kcal_thresh

            self.apply_mask(('structures', 'constrained_indices', 'energies', 'exit_status'), mask)

            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for energy ({np.count_nonzero(mask)} left, threshold {2*self.options.kcal_thresh} kcal/mol)')

        ################################################# PRUNING: FITNESS (POST FORCE FIELD OPT)

        self.fitness_refining(threshold=2)

        ################################################# PRUNING: SIMILARITY (POST FORCE FIELD OPT)

        self.zero_candidates_check()
        self.similarity_refining()

        ################################################# CHECKPOINT AFTER FF OPTIMIZATION
        
        s = f'--> Checkpoint output - Updated {len(self.structures)} optimized structures to {self.outname} file'

        if self.options.optimization and (self.options.ff_level != self.options.theory_level) and conv_thr != "tight":
            s += f' before {self.options.calculator} optimization.'

        else:
            self.outname = f'TSCoDe_poses_{self.stamp}.xyz'
            # if the FF optimization was the last one, call the outfile accordingly


        self.log(s+'\n')

        with open(self.outname, 'w') as f:        
            for i, (structure, status, energy) in enumerate(zip(align_structures(self.structures),
                                                                self.exit_status,
                                                                self.rel_energies())):

                kind = 'REFINED - ' if status else 'NOT REFINED - '
                write_xyz(structure, self.atomnos, f, title=f'Structure {i+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({self.options.ff_level})')

        # do not retain energies for the next optimization step if optimization was not tight
        if not only_fixed_constraints:
            self.energies.fill(0)

    def _set_target_distances(self):
        '''
        Called before TS refinement to compute all
        target bonding distances. These are only returned
        if that pairing is not a non-covalent interaction,
        that is if pairing was not specified with letters
        "x", "y" or "z".
        '''
        self.target_distances = {}

        # grab the atoms we want to extract information from
        r_atoms = {}
        for mol in self.objects:
            for letter, r_atom in mol.reactive_atoms_classes_dict[0].items():
                cumnum = r_atom.cumnum if hasattr(r_atom, 'cumnum') else r_atom.index
                if letter not in ("x", "y", "z"):
                    r_atoms[cumnum] = r_atom

        pairings = self.constrained_indices.ravel()
        pairings = pairings.reshape(int(pairings.shape[0]/2), 2)
        pairings = {tuple(sorted((a,b))) for a, b in pairings}

        active_pairs = [indices for letter, indices in self.pairings_table.items() if letter not in ("x", "y", "z")]

        for index1, index2 in pairings:

            if [index1, index2] in active_pairs:

                if hasattr(self, 'pairing_dists'):
                    letter = list(self.pairings_table.keys())[active_pairs.index([index1, index2])]

                    if letter in self.pairing_dists:
                        self.target_distances[(index1, index2)] = self.pairing_dists[letter]
                        continue
                # if target distance has been specified by user, read that, otherwise compute it

                r_atom1 = r_atoms[index1]
                r_atom2 = r_atoms[index2]

                dist1 = orb_dim_dict.get(r_atom1.symbol + ' ' + str(r_atom1), orb_dim_dict['Fallback'])
                dist2 = orb_dim_dict.get(r_atom2.symbol + ' ' + str(r_atom2), orb_dim_dict['Fallback'])

                self.target_distances[(index1, index2)] = dist1 + dist2
 
    def get_pairing_dist_from_letter(self, letter):
        '''
        Get constrained distance between paired reactive
        atoms, accessed via the associated constraint letter
        '''

        if hasattr(self, 'pairing_dists') and self.pairing_dists.get(letter) is not None:
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

    def optimization_refining(self, maxiter=None, conv_thr='tight', only_fixed_constraints=False):
        '''
        Refines structures by constrained optimizations with the active calculator,
        discarding similar ones and scrambled ones.
        maxiter - int, number of max iterations for the optimization
        constraints - 'all' or 'fixed'. The latter only considers (a,b,c) and not (x,y,z)

        '''

        self.outname = f'TSCoDe_poses_{self.stamp}.xyz'


        task = 'Distance refining' if only_fixed_constraints else 'Structure optimization'
        self.log(f'--> {task} ({self.options.theory_level}{f"/{self.options.solvent}" if self.options.solvent is not None else ""} level via {self.options.calculator}, {self.threads} thread{"s" if self.threads>1 else ""})')

        self.energies.fill(0)
        # Resetting all energies since we changed theory level

        t_start = time.perf_counter()
        processes = []
        cum_time = 0

        with ProcessPoolExecutor(max_workers=self.threads) as executor:

            opt_func = opt_funcs_dict[self.options.calculator]

            for i, structure in enumerate(deepcopy(self.structures)):
                loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')

                if only_fixed_constraints:
                    constraints = np.array([value for key, value in self.pairings_table.items() if key in ('a', 'b', 'c')])
                
                else:
                    constraints = np.concatenate([self.constrained_indices[i], self.internal_constraints]) if len(self.internal_constraints) > 0 else self.constrained_indices[i]

                pairing_dists = [self.get_pairing_dists_from_constrained_indices(_c) for _c in constraints]

                process = executor.submit(
                                            timing_wrapper,
                                            opt_func,
                                            structure,
                                            self.atomnos,
                                            method=self.options.theory_level,
                                            solvent=self.options.solvent,
                                            charge=self.options.charge,
                                            maxiter=maxiter,
                                            conv_thr=conv_thr,
                                            constrained_indices=constraints,
                                            constrained_distances=pairing_dists,
                                            procs=self.procs,
                                            title=f'Candidate_{i+1}',
                                        )
                processes.append(process)

            for i, process in enumerate(as_completed(processes)):
                        
                loadbar(i, len(self.structures), prefix=f'Optimizing structure {i+1}/{len(self.structures)} ')

                (new_structure, new_energy, self.exit_status[i]), t_struct = process.result()
                
                # # assert that the structure did not scramble during optimization
                if self.exit_status[i]:
                    constraints = (np.concatenate([self.constrained_indices[i], self.internal_constraints])
                                   if len(self.internal_constraints) > 0
                                   else self.constrained_indices[i])
                    self.exit_status[i] = scramble_check(new_structure,
                                                        self.atomnos,
                                                        excluded_atoms=constraints.ravel(),
                                                        mols_graphs=self.graphs,
                                                        max_newbonds=0)
                    
                cum_time += t_struct

                if self.options.debug:
                    exit_status = 'REFINED  ' if self.exit_status[i] else 'SCRAMBLED'
                    self.log(f'    - Candidate_{i+1} - {exit_status} {time_to_string(t_struct, digits=3)}', p=False)
                
                self.structures[i] = new_structure
                if self.exit_status[i] and new_energy is not None:
                    self.energies[i] = new_energy

                else:
                    self.energies[i] = 1E10

                ### Update checkpoint every 20 optimized structures, and give an estimate of the remaining time

                if i % 20 == 19:

                    with open(self.outname, 'w') as f:        
                        for j, (structure, status, energy) in enumerate(zip(align_structures(self.structures),
                                                                            self.exit_status,
                                                                            self.rel_energies())):

                            kind = 'REFINED - ' if status else 'NOT REFINED - '
                            write_xyz(structure, self.atomnos, f, title=f'Structure {j+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({self.options.ff_level})')

                    elapsed = time.perf_counter() - t_start
                    average = (elapsed)/(i+1)
                    time_left = time_to_string((average) * (len(self.structures)-i-1))
                    speedup = cum_time/elapsed
                    self.log(f'    - Optimized {i+1:>4}/{len(self.structures):>4} structures - updated checkpoint file (avg. {time_to_string(average)}/struc, {round(speedup, 1)}x speedup, est. {time_left} left)', p=False)

            loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')
            
            elapsed = time.perf_counter() - t_start
            average = (elapsed)/(len(self.structures))
            speedup = cum_time/elapsed

            self.log((f'{self.options.calculator}/{self.options.theory_level} optimization took '
                    f'{time_to_string(elapsed)} (~{time_to_string(average)} per structure, {round(speedup, 1)}x speedup)'))

            ################################################# EXIT STATUS

            self.log(f'Successfully optimized {len([b for b in self.exit_status if b])}/{len(self.structures)} structures. Non-optimized ones will not be discarded.')

            ################################################# PRUNING: ENERGY

            _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
            self.energies = self.scramble(self.energies, sequence)
            self.structures = self.scramble(self.structures, sequence)
            self.constrained_indices = self.scramble(self.constrained_indices, sequence)
            # sorting structures based on energy

            if self.options.debug:
                self.dump_status(f'optimization_refining_{conv_thr}')

            if self.options.kcal_thresh is not None and only_fixed_constraints:
        
                mask = self.rel_energies() < self.options.kcal_thresh

                self.apply_mask(('structures', 'constrained_indices', 'energies', 'exit_status'), mask)

                if False in mask:
                    self.log(f'Discarded {len([b for b in mask if not b])} candidates for energy ({np.count_nonzero(mask)} left, threshold {self.options.kcal_thresh} kcal/mol)')
       
            ################################################# PRUNING: FITNESS (POST SEMIEMPIRICAL OPT)

            self.fitness_refining(threshold=2)

            ################################################# PRUNING: SIMILARITY (POST SEMIEMPIRICAL OPT)

            self.zero_candidates_check()
            self.similarity_refining()

            ################################################# CHECKPOINT AFTER SE OPTIMIZATION      

            with open(self.outname, 'w') as f:        
                for i, (structure, status, energy) in enumerate(zip(align_structures(self.structures),
                                                                    self.exit_status,
                                                                    self.rel_energies())):

                    kind = 'REFINED - ' if status else 'NOT REFINED - '
                    write_xyz(structure, self.atomnos, f, title=f'Structure {i+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({self.options.ff_level})')

            self.log(f'--> Wrote {len(self.structures)} optimized structures to {self.outname}')

            # do not retain energies for the next optimization step if optimization was not tight
            if not only_fixed_constraints:
                self.energies.fill(0)

    def metadynamics_augmentation(self):
        '''
        Runs a metadynamics simulation (MTD) through
        the XTB program for each structure in self.structure.
        New structures are obtained from the simulations, minimized
        in energy and added to self. structures.
        '''

        self.log(f'--> Performing XTB Metadynamic augmentation of TS candidates')

        before = len(self.structures)
        t_start_run = time.perf_counter()

        for s, (structure, constrained_indices) in enumerate(zip(deepcopy(self.structures), deepcopy(self.constrained_indices))):

            loadbar(s, before, f'Running MTD {s+1}/{before} ')
            t_start = time.perf_counter()

            new_structures = xtb_metadyn_augmentation(structure,
                                                      self.atomnos,
                                                      constrained_indices=constrained_indices,
                                                      new_structures=5,
                                                      title=s)

            self.structures = np.concatenate((self.structures, new_structures))
            self.energies = np.concatenate((self.energies, [0 for _ in new_structures]))
            self.constrained_indices = np.concatenate((self.constrained_indices, [constrained_indices for _ in new_structures]))
        
            self.log(f'   - Structure {s+1} - {len(new_structures)} new conformers ({time_to_string(time.perf_counter()-t_start)})', p=False)

        loadbar(before, before, f'Running MTD {before}/{before} ')
        self.exit_status = np.array([True for _ in self.structures], dtype=bool)

        self.log(f'Metadynamics augmentation completed - found {len(self.structures)-before} new conformers ({time_to_string(time.perf_counter()-t_start_run)})\n')

    def csearch_augmentation(self, text='', max_structs=1000):
        '''
        Runs a conformational search for each structure in self.structure,
        preserving the current reactive atoms pose and HB interactions.
        New structures geometries are optimized and added to self. structures.
        '''

        self.log(f'--> Performing conformational augmentation of TS candidates {text}')

        before = len(self.structures)
        t_start_run = time.perf_counter()
        n_out = 100 if len(self.structures)*100 < max_structs else round(max_structs/len(self.structures))
        n_out = max((1, n_out))

        for s, (structure, constrained_indices) in enumerate(zip(self.structures, self.constrained_indices)):

            loadbar(s, before, f'Performing CSearch {s+1}/{before} ', suffix=f'({len(self.structures)-before} generated)')
            t_start = time.perf_counter()

            if self.options.debug:
                dump = open(f'Candidate_{s+1}_csearch_log.txt', 'w', buffering=1)

            try:
                new_structures = csearch(
                                            structure,
                                            self.atomnos,
                                            constrained_indices=constrained_indices,
                                            keep_hb=True,
                                            mode=2,
                                            n_out=n_out,
                                            logfunction=lambda s: dump.write(s+'\n') if self.options.debug else None,
                                            title=f'Candidate_{s+1}',
                                            interactive_print=False,
                                            write_torsions=self.options.debug
                                        )

            # if CSearch cannot be performed, just go on
            except SegmentedGraphError:
                new_structures = []

            if self.options.debug:
                dump.close()

            if len(new_structures) != 0: # could be either array or list, so have to check this way
                self.structures = np.concatenate((self.structures, new_structures))
                self.energies = np.concatenate((self.energies, [1E10 for _ in new_structures]))
                self.constrained_indices = np.concatenate((self.constrained_indices, [constrained_indices for _ in new_structures]))
        
            self.log(f'   - Candidate {s+1} - {len(new_structures)} new conformers ({time_to_string(time.perf_counter()-t_start)})', p=False)

        loadbar(before, before, f'Performing CSearch {before}/{before} ', suffix=f'{" "*15}')
        self.exit_status = np.array([True for _ in self.structures], dtype=bool)

        self.similarity_refining(rmsd=False)

        self.log(f'Conformational augmentation completed - generated {len(self.structures)-before} new conformers ({time_to_string(time.perf_counter()-t_start_run)})\n')

    def csearch_augmentation_routine(self):
        '''
        '''

        if self.options.csearch_aug:

            csearch_func = self.csearch_augmentation if self.options.threads == 1 else self.csearch_augmentation_parallel

            null_runs = 0

            for i in range(3):

                min_e = np.min(self.energies)

                csearch_func(text=f'(step {i+1}/3)', max_structs=self.options.max_confs)
                self.force_field_refining()

                if np.min(self.energies) < min_e:
                    delta = min_e - np.min(self.energies)
                    self.log(f'--> Lower minima found: {round(delta, 2)} kcal/mol below previous best\n')

                    if self.options.debug:
                        with open(f'best_of_aug_run_{i}.xyz', 'w') as f:
                            e, s = zip(*sorted(zip(self.energies, self.structures), key=lambda x: x[0]))
                            write_xyz(s[0], self.atomnos, f, title=f'Energy = {round(e[0], 6)}')

                    min_e = np.min(self.energies)

                else:
                    self.log(f'--> No new minima found.\n')
                    null_runs += 1

                if null_runs == 2:
                    break

    def hyperneb_refining(self):
        '''
        Performs a clibing-image NEB calculation inferring reagents and products for each structure.
        '''
        self.log(f'--> HyperNEB optimization ({self.options.theory_level} level)')
        t_start = time.perf_counter()

        for i, structure in enumerate(self.structures):

            loadbar(i, len(self.structures), prefix=f'Performing NEB {i+1}/{len(self.structures)} ')

            t_start_opt = time.perf_counter()

            try:

                self.structures[i], self.energies[i], self.exit_status[i] = hyperNEB(self,
                                                                                        structure,
                                                                                        self.atomnos,
                                                                                        self.ids,
                                                                                        self.constrained_indices[i],
                                                                                        title=f'structure_{i+1}')

                exit_str = 'COMPLETED' if self.exit_status[i] else 'CRASHED'

            except (MopacReadError, ValueError):
                # Both are thrown if a MOPAC file read fails, but the former occurs when an internal (TSCoDe)
                # read fails (getting reagent or product), the latter when an ASE read fails (during NEB)
                exit_str = 'CRASHED'
                self.exit_status[i] = False

            t_end_opt = time.perf_counter()

            self.log(f'    - {self.options.calculator} {self.options.theory_level} NEB optimization: Structure {i+1} - {exit_str} - ({time_to_string(t_end_opt-t_start_opt)})', p=False)

        loadbar(1, 1, prefix=f'Performing NEB {len(self.structures)}/{len(self.structures)} ')
        t_end = time.perf_counter()
        self.log(f'{self.options.calculator} {self.options.theory_level} NEB optimization took {time_to_string(t_end-t_start)} ({time_to_string((t_end-t_start)/len(self.structures))} per structure)')
        self.log(f'NEB converged for {len([i for i in self.exit_status if i])}/{len(self.structures)} structures\n')

        mask = self.exit_status
        self.apply_mask(('structures', 'energies', 'exit_status'), mask)

        ################################################# PRUNING: SIMILARITY (POST NEB)

        if len(self.structures) != 0:

            t_start = time.perf_counter()
            self.structures, mask = prune_conformers_rmsd(self.structures, self.atomnos, max_rmsd=self.options.rmsd)
            self.energies = self.energies[mask]
            t_end = time.perf_counter()
            
            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
            self.log()

        ################################################# NEB XYZ OUTPUT

            self.energies -= np.min(self.energies)
            _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
            self.energies = scramble(self.energies, sequence)
            self.structures = scramble(self.structures, sequence)
            self.constrained_indices = scramble(self.constrained_indices, sequence)
            # sorting structures based on energy

            self.outname = f'TSCoDe_NEB_TSs_{self.stamp}.xyz'
            with open(self.outname, 'w') as f:        
                for structure, energy in zip(align_structures(self.structures), self.rel_energies()):
                    write_xyz(structure, self.atomnos, f, title=f'Structure {i+1} - TS - Rel. E. = {round(energy, 3)} kcal/mol')

            self.log(f'Wrote {len(self.structures)} final structures to {self.outname} file\n')

    def saddle_refining(self):
        '''
        Performs a first order saddle optimization for each structure.
        '''
        self.log(f'--> Saddle optimization ({self.options.theory_level} level)')
        t_start = time.perf_counter()

        for i, structure in enumerate(self.structures):

            loadbar(i, len(self.structures), prefix=f'Performing saddle opt {i+1}/{len(self.structures)} ')

            try:

                self.structures[i], self.energies[i], self.exit_status[i] = ase_saddle(self,
                                                                            structure,
                                                                            self.atomnos,
                                                                            self.constrained_indices[i],
                                                                            mols_graphs=self.graphs if self.embed != 'monomolecular' else None,
                                                                            title=f'Saddle opt - Structure {i+1}',
                                                                            logfile=self.logfile,
                                                                            traj=f'Saddle_opt_{i+1}.traj',
                                                                            maxiterations=200)

            except ValueError:
                # Thrown when an ASE read fails (during saddle opt)
                self.exit_status[i] = False

        loadbar(1, 1, prefix=f'Performing saddle opt {len(self.structures)}/{len(self.structures)} ')
        t_end = time.perf_counter()
        self.log(f'{self.options.calculator} {self.options.theory_level} saddle optimization took {time_to_string(t_end-t_start)} ({time_to_string((t_end-t_start)/len(self.structures))} per structure)')
        self.log(f'Saddle opt completed for {len([i for i in self.exit_status if i])}/{len(self.structures)} structures')

        mask = self.exit_status

        self.apply_mask(('structures', 'energies', 'exit_status'), mask)

        ################################################# PRUNING: SIMILARITY (POST SADDLE OPT)

        if len(self.structures) != 0:

            t_start = time.perf_counter()
            self.structures, mask = prune_conformers_rmsd(self.structures, self.atomnos, max_rmsd=self.options.rmsd)
            self.apply_mask(('energies', 'exit_status'), mask)
            t_end = time.perf_counter()
            
            if False in mask:
                self.log(f'Discarded {len([b for b in mask if not b])} candidates for similarity ({len([b for b in mask if b])} left, {time_to_string(t_end-t_start)})')
            self.log()

        ################################################# SADDLE OPT EXTRA XYZ OUTPUT

            _, sequence = zip(*sorted(zip(self.energies, range(len(self.energies))), key=lambda x: x[0]))
            self.energies = scramble(self.energies, sequence)
            self.structures = scramble(self.structures, sequence)
            self.constrained_indices = scramble(self.constrained_indices, sequence)
            # sorting structures based on energy

            self.outname = f'TSCoDe_SADDLE_TSs_{self.stamp}.xyz'
            with open(self.outname, 'w') as f:        
                for structure, energy in zip(align_structures(self.structures), self.rel_energies()):
                    write_xyz(structure, self.atomnos, f, title=f'Structure {i+1} - TS - Rel. E. = {round(energy, 3)} kcal/mol')

            self.log(f'Wrote {len(self.structures)} saddle-optimized structures to {self.outname} file\n')

        else:
            self.log()

    def independent_scans_refining(self):
        '''
        Performs independent scans optimization for each structure.
        '''
        self.log(f'--> Performing independent scans refinement ({self.options.theory_level} level)')
        t_start = time.perf_counter()

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
        t_end = time.perf_counter()
        self.log(f'{self.options.calculator} {self.options.theory_level} independent scans took {time_to_string(t_end-t_start)} ({time_to_string((t_end-t_start)/len(self.structures))} per structure)')
        self.log(f'Independent scans refinement completed for {len([i for i in self.exit_status if i])}/{len(self.structures)} structures')

        mask = self.exit_status
        self.apply_mask(('structures', 'energies', 'exit_status'), mask)


        ################################################# PRUNING: SIMILARITY (POST ISCANS OPT)

        if len(self.structures) != 0:

            t_start = time.perf_counter()
            self.structures, mask = prune_conformers_rmsd(self.structures, self.atomnos, max_rmsd=self.options.rmsd)
            self.apply_mask(('energies', 'exit_status'), mask)
            t_end = time.perf_counter()
            
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

            nci, print_list = get_nci(structure, self.atomnos, self.constrained_indices[i], self.ids)
            self.nci.append(nci)

            if nci != []:
                self.log(f'Structure {i+1}: {len(nci)} interactions')

                for p in print_list:
                    self.log('    '+p)
                self.log()
        
        if not [_l for _l in self.nci if _l != []]:
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
                    self.log(f'    {nci_type} between indices {i1}/{i2} is present in {len(shared_by)}/{len(self.structures)} structures {tuple([i+1 for i in shared_by])}')
                self.log()

    def write_mol_info(self):
        '''
        Writes information about the TSCoDe molecules read from the input file.
        '''

        if self.embed != 'dihedral':

            head = ''
            for i, mol in enumerate(self.objects):

                if hasattr(mol, 'reactive_atoms_classes_dict'):

                    descs = [atom.symbol+f'({str(atom)} type, {round(norm_of(atom.center[0]-atom.coord), 3)} A, ' +
                            f'{len(atom.center)} center{"s" if len(atom.center) != 1 else ""})' for atom in mol.reactive_atoms_classes_dict[0].values()]

                else:

                    descs = [pt[mol.atomnos[i]].symbol for i in mol.reactive_indices]

                t = '\n        '.join([(str(index) + ' ' if len(str(index)) == 1 else str(index)) + ' -> ' + desc for index, desc in zip(mol.reactive_indices, descs)])
               
                mol_line = f' -> {len(mol.atomcoords[0])} atoms, {len(mol.atomcoords)} conformer{"s" if len(mol.atomcoords) != 1 else ""}'
                if hasattr(mol, 'pivots') and len(mol.pivots) > 0:
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

            self.log('--> Input structures & reactive indices data:\n' + head)

    def write_options(self):
        '''
        Writes information about the TSCoDe parameters used in the calculation, if applicable to the run.
        '''

        ######################################################################################################## PAIRINGS

        if not self.pairings_table:
            if all([len(mol.reactive_indices) == 2 for mol in self.objects]):
                self.log('--> No atom pairings imposed. Computing all possible dispositions.\n')
                # only print the no pairings statements if there are multiple regioisomers to be computed
        else:
            self.log(f'--> Atom pairings imposed are {len(self.pairings_table)}: {list(self.pairings_table.values())} (Cumulative index numbering)\n')
            
            for i, letter in enumerate(self.pairings_table):
                kind = 'Constraint' if letter in ('a', 'b', 'c') else 'Interaction'
                kind += ' (Internal)' if any(isinstance(d.get(letter), tuple) for d in self.pairings_dict.values()) else ''
                dist = self.get_pairing_dist_from_letter(letter)

                if self.options.shrink:
                    dist *= self.options.shrink_multiplier

                if dist is None:
                    kind += f' - will relax'
                elif kind == 'Interaction':
                    kind += f' - embedded at {round(dist, 3)} A - will relax'
                else:
                    kind += f' - constrained to {round(dist, 3)} A'

                if self.options.shrink:
                    kind += f' (to be shrinked to {round(dist/self.options.shrink_multiplier, 3)} A)'

                s = f'    {i+1}. {letter} - {kind}\n'

                for mol_id, d in self.pairings_dict.items():
                    atom_id = d.get(letter)

                    if atom_id is not None:
                        mol = self.objects[mol_id]

                        if isinstance(atom_id, int):
                            atom_id = [atom_id]
                        
                        for a in atom_id:
                            s += f'       Index {a} ({pt[mol.atomnos[a]].name}) on {mol.rootname}\n'

                self.log(s)

        ######################################################################################################## EMBEDDING/CALC OPTIONS

        self.log(f'--> Calculation options used were:')
        for line in str(self.options).split('\n'):

            if self.embed in ('monomolecular', 'string', 'refine') and line.split()[0] in ('rotation_range',
                                                                                          'rotation_steps',
                                                                                          'rigid',
                                                                                          'suprafacial',
                                                                                          'fix_angles_in_deformation',
                                                                                          'double_bond_protection'):
                continue

            if self.embed == 'dihedral' and line.split()[0] not in ('optimization',
                                                                    'calculator',
                                                                    'theory_level',
                                                                    'kcal_thresh',
                                                                    'debug',
                                                                    'rmsd',
                                                                    'neb',
                                                                    'saddle'):
                continue

            if self.embed == 'refine' and line.split()[0] in ('shrink',
                                                              'shrink_multiplier',
                                                              'fix_angles_in_deformation',
                                                              'double_bond_protection'):
                continue

            if not self.options.optimization and line.split()[0] in ('calculator',
                                                                     'double_bond_protection',
                                                                     'ff_opt',
                                                                     'ff_calc',
                                                                     'ff_level',
                                                                     'fix_angles_in_deformation',
                                                                     'only_refined',
                                                                     'rigid',
                                                                     'theory_level'):
                continue
            
            if self.options.rigid and line.split()[0] in ('double_bond_protection',
                                                          'fix_angles_in_deformation'):
                continue

            if not self.options.shrink and line.split()[0] in ('shrink_multiplier',):
                continue

            if not self.options.ff_opt and line.split()[0] in ('ff_calc', 'ff_level'):
                continue

            self.log(f'    - {line}')

    # def write_vmd(self, indices=None):
    #     '''
    #     Write VMD file with bonds and reactive atoms highlighted.
    #     '''

    #     if indices is None:
    #         indices = self.constrained_indices[0].ravel()

    #     self.vmd_name = self.outname.split('.')[0] + '.vmd'
    #     path = os.path.join(os.getcwd(), self.vmd_name)
    #     with open(path, 'w') as f:
    #         s = ('display resetview\n' +
    #             'mol new {%s.xyz}\n' % (os.path.join(os.getcwd() + self.vmd_name).rstrip('.vmd')) +
    #             'mol selection index %s\n' % (' '.join([str(i) for i in indices])) +
    #             'mol representation CPK 0.7 0.5 50 50\n' +
    #             'mol color ColorID 7\n' +
    #             'mol material Transparent\n' +
    #             'mol addrep top\n')

    #         for a, b in self.pairings_table.values():
    #             s += f'label add Bonds 0/{a} 0/{b}\n'

    #         f.write(s)

    #     self.log(f'--> Wrote VMD {self.vmd_name} file\n')

    def write_structures(self, tag, indices=None, energies=True, relative=True, extra='', p=True):
        '''
        '''

        # if indices is None:
        #     indices = self.constrained_indices[0]

        if energies:
            rel_e = self.energies

            if relative:
                rel_e -= np.min(self.energies)

        # truncate if there are too many (embed debug first dump)
        if len(self.structures) > 10000 and not self.options.let:
            self.log(f'Truncated {tag} output structures to 10000 (from {len(self.structures)} - keyword LET to override).')
            output_structures = self.structures[0:10000]
        else:
            output_structures = self.structures

        self.outname = f'TSCoDe_{tag}_{self.stamp}.xyz'
        with open(self.outname, 'w') as f:        
            for i, structure in enumerate(align_structures(output_structures, indices)):
                title = f'TS candidate {i+1} - {tag}'

                if energies:
                    title += f' - Rel. E. = {round(rel_e[i], 3)} kcal/mol '
                
                title += extra

                write_xyz(structure, self.atomnos, f, title=title)

        if p:
            self.log(f'Wrote {len(output_structures)} {tag} structures to {self.outname} file.\n')

    def normal_termination(self):
        '''
        Terminate the run, printing the total time and the
        relative energies of the first 50 structures, if possible.

        '''
        clean_directory()
        self.log(f'\n--> TSCoDe normal termination: total time {time_to_string(time.perf_counter() - self.t_start_run, verbose=True)}.')
        
        if hasattr(self, "structures"):
            if len(self.structures) > 0 and hasattr(self, "energies"):
                self.log(f'\n--> Energies of output structures ({self.options.theory_level}/{self.options.calculator}{f"/{self.options.solvent}" if self.options.solvent is not None else ""})\n')

                self.energies = self.energies if len(self.energies) <= 50 else self.energies[0:50]

                self.log(f'> #                Rel. E.           RMSD')
                self.log('-------------------------------------------')
                for i, energy in enumerate(self.energies-self.energies[0]):

                    rmsd_value = '(ref)' if i == 0 else str(round(kabsch_rmsd(self.structures[i], self.structures[0], translate=True), 2))+' Å'

                    self.log('> Candidate {:2}  :  {:4} kcal/mol  :  {}'.format(
                                                                        str(i+1),
                                                                        round(energy, 2),
                                                                        rmsd_value))

        self.write_quote()
        self.logfile.close()
        sys.exit()

    def run(self):
        '''
        Run the TSCoDe program.
        '''
        self.write_mol_info()

        if self.embed is None:
            self.log(f'--> No embed requested, exiting.\n')
            self.normal_termination()

        if self.embed == 'error':
            self.log(f'--> Embed type not recognized, exiting.\n')
            self.normal_termination()

        if self.embed == 'data':
            self.data_termination()

        if not self.options.let and (
               self.embed in ('cyclical', 'chelotropic')) and (
               max([len(mol.atomcoords) for mol in self.objects]) > 100) and (
               not self.options.rigid):

            self.options.rigid = True

            self.log(f'--> Large embed: RIGID keyword added for efficiency (override with LET)')

        self.write_options()
        self.t_start_run = time.perf_counter()

        if self.options.dryrun:
            self.log('\n--> Dry run requested: exiting.')
            self.normal_termination()

        try: # except KeyboardInterrupt
            try: # except ZeroCandidatesError()
                self.generate_candidates()
                
                if self.options.bypass:
                    self.write_structures('unoptimized', energies=False)
                    self.normal_termination()

                self.compenetration_refining()
                self.similarity_refining(rmsd=False, verbose=True)

                if self.options.optimization:

                    if self.options.ff_opt:

                        if len(self.objects) > 1 and self.options.ff_calc == 'XTB':
                            self.log(f"--> Performing {self.options.calculator} FF pre-optimization (loose convergence, molecular and pairing constraints)\n")
                            self.force_field_refining(conv_thr="loose", prevent_scrambling=True)


                        self.log(f"--> Performing {self.options.calculator} FF optimization (loose convergence, pairing constraints, step 1/2)\n")
                        self.force_field_refining(conv_thr="loose")

                        self.log(f"--> Performing {self.options.calculator} FF optimization (tight convergence, fixed constraints only, step 2/2)\n")
                        self.force_field_refining(conv_thr="tight", only_fixed_constraints=True)

                        self.csearch_augmentation_routine()

                    if not (self.options.ff_opt and self.options.theory_level == self.options.ff_level):
                        # If we just optimized at a (FF) level and the final
                        # optimization level is the same, avoid repeating it

                        if self.options.calculator == "ORCA":
                        # Perform stepwise pruning of the ensemble for more expensive theory levels
                            
                            self.log("--> Performing ORCA optimization (3 iterations, step 1/3)\n")
                            self.optimization_refining(maxiter=3)

                            self.log("--> Performing ORCA optimization (5 iterations, step 2/3)\n")
                            self.optimization_refining(maxiter=5)

                            self.log("--> Performing ORCA optimization (convergence, step 3/3)\n")

                        self.optimization_refining(conv_thr='loose')
                        # final uncompromised optimization (with fixed constraints and interactions active)

                        self.optimization_refining(conv_thr='tight', only_fixed_constraints=True)
                        # self.distance_refining()
                        # final uncompromised optimization (with only fixed constraints active)

                else:
                    self.write_structures('unoptimized', energies=False)
                    # accounting for output in "refine" runs with NOOPT

            except ZeroCandidatesError:
                t_end_run = time.perf_counter()
                s = ('    Sorry, the program did not find any reasonable embedded structure. Are you sure the input indices and pairings were correct? If so, try these tips:\n'
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
                sys.exit()

            ##################### AUGMENTATION - METADYNAMICS / CSEARCH

            if self.options.metadynamics:

                self.metadynamics_augmentation()
                self.optimization_refining()
                self.similarity_refining()

            ##################### POST TSCODE - SADDLE, NEB, NCI, VMD

            # if (self.options.optimization or self.options.ff_opt) and not self.options.bypass:
            #     self.write_vmd()

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

            ################################################ END

        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt requested by user. Quitting.')
            sys.exit()

    def data_termination(self):
        '''
        Type of termination for runs when there is no embedding,
        but some computed data are to be shown in a formatted way.
        '''

        if any('pka>' in op for op in self.options.operators):
            self.pka_termination()

        if len([op for op in self.options.operators if 'scan>' in op]) > 1:
            self.scan_termination()

        self.normal_termination()

    def pka_termination(self):
        '''
        Print data acquired during pKa energetics calculation
        for every molecule in input
        '''

        self.log(f'\n--> pKa energetics (from best conformers)')
        solv = 'gas phase' if self.options.solvent is None else self.options.solvent

        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ['Name', '#(Symb)', 'Process', 'Energy (kcal/mol)']

        for mol in self.objects:
            if hasattr(mol, 'pka_data'):
                table.add_row([mol.rootname,
                               f'{mol.reactive_indices[0]}({pt[mol.atomnos[mol.reactive_indices[0]]].symbol})',
                               mol.pka_data[0],
                               mol.pka_data[1]])

        # Add pKa column if we were given a reference
        if hasattr(self, 'pka_ref'):

            pkas = []
            for mol in self.objects:
                if mol.name == self.pka_ref[0]:
                    dG_ref = mol.pka_data[1]
                    break

            for mol in self.objects:
                process, free_energy = mol.pka_data

                dG = free_energy - dG_ref if process == 'HA -> A-' else dG_ref - free_energy
                # The free energy difference has a different sign for acids or bases, since
                # the pKa for a base is the one of its conjugate acid, BH+

                pka = dG / (np.log(10) * 1.9872036e-3 * 298.15) + self.pka_ref[1]
                pkas.append(round(pka, 3))

            table.add_column(f'pKa ({solv}, 298.15 K)', pkas)

        self.log(table.get_string())
        self.log(f'\n  Level used is {self.options.theory_level} via {self.options.calculator}' + 
                 f", using the ALPB solvation model for {self.options.solvent}" if self.options.solvent is not None else "")

        if len(self.objects) == 2:
            mol0, mol1 = self.objects
            if hasattr(mol0, 'pka_data') and hasattr(mol1, 'pka_data'):
                tags = (mol0.pka_data[0],
                        mol1.pka_data[0])
                if 'HA -> A-' in tags and 'B -> BH+' in tags:
                    dG = mol0.pka_data[1] + mol1.pka_data[1]
                    self.log(f'\n  Equilibrium data:')
                    self.log(f'\n    HA + B -> BH+ + A-    K({solv}, 298.15 K) = {round(np.exp(-dG/(1.9872036e-3 * 298.15)), 3)}')
                    self.log(f'\n                         dG({solv}, 298.15 K) = {round(dG, 3)} kcal/mol')

    def scan_termination(self):
        '''
        Print the unified data and write the cumulative plot
        for the approach of all the molecules in input
        '''
        import pickle

        import matplotlib.pyplot as plt

        fig = plt.figure()

        for mol in self.objects:
            if hasattr(mol, 'scan_data'):
                plt.plot(*mol.scan_data, label=mol.rootname)

        plt.legend()
        plt.title('Unified scan energetics')
        plt.xlabel(f'Distance (A)')
        plt.gca().invert_xaxis()
        plt.ylabel('Rel. E. (kcal/mol)')
        plt.savefig(f'{self.stamp}_cumulative_plt.svg')
        with open(f'{self.stamp}_cumulative_plt.pickle', 'wb') as _f:
            pickle.dump(fig, _f)
        
        self.log(f'\n--> Written cumulative scan plot at {self.stamp}_cumulative_plt.svg')