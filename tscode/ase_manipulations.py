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

from ase import Atoms
from ase.calculators.gaussian import Gaussian
from ase.calculators.mopac import MOPAC
from ase.calculators.orca import ORCA
from ase.constraints import FixInternals
from ase.dyneb import DyNEB
from ase.optimize import BFGS, LBFGS
from ase.vibrations import Vibrations

import numpy as np
import time
import os
from copy import deepcopy

from utils import (
                    norm,
                    time_to_string,
                    HiddenPrints,
                    write_xyz,
                    neighbors,
                    get_double_bonds_indexes,
                    findPaths
                    )
from settings import COMMANDS, MEM_GB
from utils import scramble_check, molecule_check
from sella import Sella
from rmsd import kabsch


class Spring:
    '''
    ASE Custom Constraint Class
    Adds an harmonic force between a pair of atoms.
    Spring constant is very high to achieve tight convergence,
    but force is dampened so as not to ruin structures.
    '''
    def __init__(self, i1, i2, d_eq, k=1000):
        self.i1, self.i2 = i1, i2
        self.d_eq = d_eq
        self.k = k

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_forces(self, atoms, forces):

        direction = atoms.positions[self.i2] - atoms.positions[self.i1]
        # vector connecting atom1 to atom2

        spring_force = self.k * (np.linalg.norm(direction) - self.d_eq)
        # absolute spring force (float). Positive if spring is overstretched.

        # spring_force = np.clip(spring_force, -10, 10)
        # force is clipped at 10 eV/A

        forces[self.i1] += (norm(direction) * spring_force)
        forces[self.i2] -= (norm(direction) * spring_force)
        # applying harmonic force to each atom, directed toward the other one

    def __repr__(self):
        return f'Spring - ids:{self.i1}/{self.i2} - d_eq:{self.d_eq}, k:{self.k}'

def get_ase_calc(calculator, procs, method):
    '''
    Attach the correct ASE calculator
    to the ASE Atoms object
    '''

    if calculator == 'XTB':
        try:
            from xtb.ase.calculator import XTB
        except ImportError:
            raise Exception(('Cannot import xtb python bindings. Install them with:\n'
                             '>>> conda install -c conda-forge xtb-python\n'
                             '(See https://github.com/grimme-lab/xtb-python)'))
        return XTB(method=method)

    
    command = COMMANDS[calculator]

    if calculator == 'MOPAC':
        return MOPAC(label='temp',
                    command=f'{command} temp.mop > temp.cmdlog 2>&1',
                    method=method)

    if calculator == 'ORCA':
        if procs > 1:
            orcablocks = f'%pal nprocs {procs} end'
            return ORCA(label='temp',
                        command=f'{command} temp.inp > temp.out 2>&1',
                        orcasimpleinput=method,
                        orcablocks=orcablocks)
        return ORCA(label='temp',
                    command=f'{command} temp.inp > temp.out 2>&1',
                    orcasimpleinput=method)

    if calculator == 'GAUSSIAN':

        # firstline = method if procs == 1 else f'%nprocshared={procs}\n{method}'

        calc = Gaussian(label='temp',
                        command=f'{command} temp.com',
                        method=method,
                        nprocshared=procs,
                        mem=str(MEM_GB)+'GB',
                        )

        if 'g09' in command:

            from ase.io import read
            def g09_read_results(self=calc):
                output = read(self.label + '.out', format='gaussian-out')
                self.calc = output.calc
                self.results = output.calc.results

            calc.read_results = g09_read_results

            # Adapting for g09 outputting .out files instead of g16 .log files.
            # This is a bad fix and the issue should be corrected in
            # the ASE source code: pull request on GitHub pending

            return calc

def ase_adjust_spacings(docker, structure, atomnos, constrained_indexes, title=0, traj=None):
    '''
    docker: TSCoDe docker object
    structure: TS candidate coordinates to be adjusted
    atomnos: 1-d array with element numbering for the TS
    constrained_indexes: (n,2)-shaped array of indexes to be distance constrained
    mols_graphs: list of NetworkX graphs, ordered as the single molecules in the TS
    title: number to be used for referring to this structure in the docker log
    traj: if set to a string, traj+'.traj' is used as a filename for the refinement trajectory.
    '''
    atoms = Atoms(atomnos, positions=structure)

    atoms.calc = get_ase_calc(docker.options.calculator, docker.options.procs, docker.options.theory_level)
    
    springs = []

    for i1, i2 in constrained_indexes:
        pair = tuple(sorted((i1, i2)))
        springs.append(Spring(i1, i2, docker.target_distances[pair]))

    atoms.set_constraint(springs)

    t_start_opt = time.time()
    with LBFGS(atoms, maxstep=0.2, logfile=None, trajectory=traj) as opt:
        opt.run(fmax=0.05, steps=500)
        iterations = opt.nsteps

    new_structure = atoms.get_positions()

    success = scramble_check(new_structure, atomnos, constrained_indexes, docker.graphs)
    exit_str = 'REFINED' if success else 'SCRAMBLED'

    docker.log(f'    - {docker.options.calculator} {docker.options.theory_level} refinement: Structure {title} {exit_str} ({iterations} iterations, {time_to_string(time.time()-t_start_opt)})', p=False)

    return new_structure, atoms.get_total_energy(), success

def ase_saddle(coords, atomnos, calculator, method, procs=1, title='temp', logfile=None, traj=None, freq=False, maxiterations=200):
    '''
    Runs a first order saddle optimization through the ASE package
    '''
    atoms = Atoms(atomnos, positions=coords)

    atoms.calc = get_ase_calc(calculator, procs, method)
    
    t_start = time.time()
    with HiddenPrints():
        with Sella(atoms,
                   logfile=None,
                   order=1,
                   trajectory=traj) as opt:

            opt.run(fmax=0.05, steps=maxiterations)
            iterations = opt.nsteps

    if logfile is not None:
        t_end_berny = time.time()
        elapsed = t_end_berny - t_start
        exit_str = 'converged' if iterations < maxiterations else 'stopped'
        logfile.write(f'{title} - {exit_str} in {iterations} steps ({time_to_string(elapsed)})\n')

    new_structure = atoms.get_positions()
    energy = atoms.get_total_energy() * 23.06054194532933 #eV to kcal/mol

    if freq:
        vib = Vibrations(atoms, name='temp')
        with HiddenPrints():
            vib.run()
        freqs = vib.get_frequencies()

        if logfile is not None:
            elapsed = time.time() - t_end_berny
            logfile.write(f'{title} - frequency calculation completed ({time_to_string(elapsed)})\n')
        
        return new_structure, energy, freqs

    # if logfile is not None:
    #     logfile.write('\n')

    return new_structure, energy, None

def ase_neb(docker, reagents, products, atomnos, n_images=6, title='temp', optimizer=LBFGS, logfile=None):
    '''
    docker: tscode docker object
    reagents: coordinates for the atom arrangement to be used as reagents
    products: coordinates for the atom arrangement to be used as products
    atomnos: 1-d array of atomic numbers
    n_images: number of optimized images connecting reag/prods
    title: name used to write the final MEP as a .xyz file
    optimizer: ASE optimizer to be used in 
    logfile: filename to dump the optimization data to. If None, no file is written.

    return: 2- element tuple with coodinates of highest point along the MEP and its energy in kcal/mol
    '''

    first = Atoms(atomnos, positions=reagents)
    last = Atoms(atomnos, positions=products)

    images =  [first]
    images += [first.copy() for i in range(n_images)]
    images += [last]

    neb = DyNEB(images, fmax=0.05, climb=False,  method='eb', scale_fmax=1, allow_shared_calculator=True)
    neb.interpolate()

    ase_dump(f'{title}_MEP_guess.xyz', images, atomnos)
    
    neb_method = docker.options.theory_level + (' GEO-OK' if docker.options.calc == 'MOPAC' else '')
    # avoid MOPAC from rejecting structures with atoms too close to each other

    # Set calculators for all images
    for i, image in enumerate(images):
        image.calc = get_ase_calc(docker.options.calculator, docker.options.procs, neb_method)

    # Set the optimizer and optimize
    try:
        with optimizer(neb, maxstep=0.1, logfile=logfile) as opt:

            opt.run(fmax=0.05, steps=20)
            # some free relaxation before starting to climb

            neb.climb = True
            opt.run(fmax=0.05, steps=500)

    except Exception as e:
        print(f'Stopped NEB for {title}:')
        print(e)

    energies = [image.get_total_energy() for image in images]
    ts_id = energies.index(max(energies))
    # print(f'TS structure is number {ts_id}, energy is {max(energies)}')

    os.remove(f'{title}_MEP_guess.xyz')
    ase_dump(f'{title}_MEP.xyz', images, atomnos)
    # Save the converged MEP (minimum energy path) to an .xyz file


    return images[ts_id].get_positions(), images[ts_id].get_total_energy()

class OrbitalSpring:
    '''
    ASE Custom Constraint Class
    Adds a series of forces based on a pair of orbitals, that is
    virtual points "bonded" to a given atom.

    :params i1, i2: indexes of reactive atoms
    :params orb1, orb2: 3D coordinates of orbitals
    :params neighbors_of_1, neighbors_of_2: lists of indexes for atoms bonded to i1/i2
    :params d_eq: equilibrium target distance between orbital centers
    '''
    def __init__(self, i1, i2, orb1, orb2, neighbors_of_1, neighbors_of_2, d_eq, k=1000):
        self.i1, self.i2 = i1, i2
        self.orb1, self.orb2 = orb1, orb2
        self.neighbors_of_1, self.neighbors_of_2 = neighbors_of_1, neighbors_of_2
        self.d_eq = d_eq
        self.k = k

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_forces(self, atoms, forces):

        # First, assess if we have to move atoms 1 and 2 at all

        sum_of_distances = (np.linalg.norm(atoms.positions[self.i1] - self.orb1) +
                            np.linalg.norm(atoms.positions[self.i2] - self.orb2) + self.d_eq)

        reactive_atoms_distance = np.linalg.norm(atoms.positions[self.i1] - atoms.positions[self.i2])

        orb_direction = self.orb2 - self.orb1
        # vector connecting orb1 to orb2

        spring_force = self.k * (np.linalg.norm(orb_direction) - self.d_eq)
        # absolute spring force (float). Positive if spring is overstretched.

        # spring_force = np.clip(spring_force, -50, 50)
        # # force is clipped at 5 eV/A

        force_direction1 = np.sign(spring_force) * norm(np.mean((norm(+orb_direction),
                                                                    norm(self.orb1-atoms.positions[self.i1])), axis=0))

        force_direction2 = np.sign(spring_force) * norm(np.mean((norm(-orb_direction),
                                                                    norm(self.orb2-atoms.positions[self.i2])), axis=0))

        # versors specifying the direction at which forces act, that is on the
        # bisector of the angle between vector connecting atom to orbital and
        # vector connecting the two orbitals

        if np.abs(sum_of_distances - reactive_atoms_distance) > 0.2:

            forces[self.i1] += (force_direction1 * spring_force)
            forces[self.i2] += (force_direction2 * spring_force)
            # applying harmonic force to each atom, directed toward the other one

        # Now applying to neighbors the force derived by torque, scaled to match the spring_force,
        # but only if atomic orbitals are more than two Angstroms apart. This improves convergence.

        if np.linalg.norm(orb_direction) > 2:
            torque1 = np.cross(self.orb1 - atoms.positions[self.i1], force_direction1)
            for i in self.neighbors_of_1:
                forces[i] += norm(np.cross(torque1, atoms.positions[i] - atoms.positions[self.i1])) * spring_force

            torque2 = np.cross(self.orb2 - atoms.positions[self.i2], force_direction2)
            for i in self.neighbors_of_2:
                forces[i] += norm(np.cross(torque2, atoms.positions[i] - atoms.positions[self.i2])) * spring_force

def PreventScramblingConstraint(graph, atoms, double_bond_protection=False, fix_angles=False):
    '''
    graph: NetworkX graph of the molecule
    atoms: ASE atoms object

    return: FixInternals constraint to apply to ASE calculations
    '''
    angles_deg = None
    if fix_angles:
        allpaths = []

        for node in graph:
            allpaths.extend(findPaths(graph, node, 2))

        allpaths = {tuple(sorted(path)) for path in allpaths}

        angles_deg = []
        for path in allpaths:
            angles_deg.append([atoms.get_angle(*path), list(path)])

    bonds = []
    for bond in [[a, b] for a, b in graph.edges if a != b]:
        bonds.append([atoms.get_distance(*bond), bond])

    dihedrals_deg = None
    if double_bond_protection:
        double_bonds = get_double_bonds_indexes(atoms.positions, atoms.get_atomic_numbers())
        if double_bonds != []:
            dihedrals_deg = []
            for a, b in double_bonds:
                n_a = neighbors(graph, a)
                n_a.remove(b)

                n_b = neighbors(graph, b)
                n_b.remove(a)

                d = [n_a[0], a, b, n_b[0]]
                dihedrals_deg.append([atoms.get_dihedral(*d), d])

    return FixInternals(dihedrals_deg=dihedrals_deg, angles_deg=angles_deg, bonds=bonds, epsilon=1)

def ase_bend(docker, original_mol, pivot, threshold, title='temp', traj=None, check=True):
    '''
    docker: TSCoDe docker object
    original_mol: Hypermolecule object to be bent
    pivot: pivot connecting two Hypermolecule orbitals to be approached/distanced
    threshold: target distance for the specified pivot, in Angstroms
    title: name to be used for referring to this structure in the docker log
    traj: if set to a string, traj+'.traj' is used as a filename for the bending trajectory.
          not only the atoms will be printed, but also all the orbitals and the active pivot.
    check: if True, after bending checks that the bent structure did not scramble.
           If it did, returns the initial molecule.
    '''

    identifier = np.sum(original_mol.atomcoords[0])

    if hasattr(docker, "ase_bent_mols_dict"):
        cached = docker.ase_bent_mols_dict.get((identifier, tuple(sorted(pivot.index)), round(threshold, 3)))
        if cached is not None:
            return cached

    if traj is not None:

        from ase.io.trajectory import Trajectory

        def orbitalized(atoms, orbitals, pivot=None):
            positions = np.concatenate((atoms.positions, orbitals))

            if pivot is not None:
                positions = np.concatenate((positions, [pivot.start], [pivot.end]))

            symbols = list(atoms.numbers) + [0 for _ in orbitals]

            if pivot is not None:
                symbols += [9 for _ in range(2)]
            # Fluorine (9) represents active orbitals
    
            new_atoms = Atoms(symbols, positions=positions)
            return new_atoms

        try:
            os.remove(traj)
        except FileNotFoundError:
            pass

    i1, i2 = original_mol.reactive_indexes

    neighbors_of_1 = list([(a, b) for a, b in original_mol.graph.adjacency()][i1][1].keys())
    neighbors_of_1.remove(i1)

    neighbors_of_2 = list([(a, b) for a, b in original_mol.graph.adjacency()][i2][1].keys())
    neighbors_of_2.remove(i2)

    mols = [deepcopy(original_mol) for _ in original_mol.atomcoords]
    for m, mol in enumerate(mols):
        mol.atomcoords = np.array([mol.atomcoords[m]])

    final_mol = deepcopy(original_mol)

    for conf, mol in enumerate(mols):

        for p in mol.pivots:
            if p.index == pivot.index:
                active_pivot = p
                break
        
        dist = np.linalg.norm(active_pivot.pivot)

        atoms = Atoms(mol.atomnos, positions=mol.atomcoords[0])

        atoms.calc = get_ase_calc(docker.options.calculator, docker.options.procs, docker.options.theory_level)
        
        if traj is not None:
            traj_obj = Trajectory(traj + f'_conf{conf}.traj', mode='a', atoms=orbitalized(atoms, np.vstack([atom.center for atom in mol.reactive_atoms_classes_dict.values()]), active_pivot))
            traj_obj.write()

        unproductive_iterations = 0
        break_reason = 'MAX ITER'
        t_start = time.time()

        for iteration in range(500):

            atoms.positions = mol.atomcoords[0]

            orb_memo = {index:np.linalg.norm(atom.center[0]-atom.coord) for index, atom in mol.reactive_atoms_classes_dict.items()}

            orb1, orb2 = active_pivot.start, active_pivot.end

            c1 = OrbitalSpring(i1, i2, orb1, orb2, neighbors_of_1, neighbors_of_2, d_eq=threshold)

            c2 = PreventScramblingConstraint(mol.graph,
                                             atoms,
                                             double_bond_protection=docker.options.double_bond_protection,
                                             fix_angles=docker.options.fix_angles_in_deformation)

            atoms.set_constraint([
                                  c1,
                                  c2,
                                  ])

            opt = BFGS(atoms, maxstep=0.2, logfile=None, trajectory=None)

            try:
                opt.run(fmax=0.5, steps=1)
            except ValueError:
                # Shake did not converge
                break_reason = 'CRASHED'
                break

            if traj is not None:
                traj_obj.atoms = orbitalized(atoms, np.vstack([atom.center for atom in mol.reactive_atoms_classes_dict.values()]))
                traj_obj.write()

            # check if we are stuck
            if np.max(np.abs(np.linalg.norm(atoms.get_positions() - mol.atomcoords[0], axis=1))) < 0.01:
                unproductive_iterations += 1

                if unproductive_iterations == 10:
                    break_reason = 'STUCK'
                    break

            else:
                unproductive_iterations = 0

            mol.atomcoords[0] = atoms.get_positions()

            # Update orbitals and get temp pivots
            for index, atom in mol.reactive_atoms_classes_dict.items():
                atom.init(mol, index, update=True, orb_dim=orb_memo[index])
                # orbitals positions are calculated based on the conformer we are working on

            temp_pivots = docker._get_pivots(mol)

            for p in temp_pivots:
                if p.index == pivot.index:
                    active_pivot = p
                    break
            # print(active_pivot)

            dist = np.linalg.norm(active_pivot.pivot)
            # print(f'{iteration}. {mol.name} conf {conf}: pivot is {round(dist, 3)} (target {round(threshold, 3)})')

            if dist - threshold < 0.1:
                break_reason = 'CONVERGED'
                break
            # else:
                # print('delta is ', round(dist - threshold, 3))

        docker.log(f'    {title} - conformer {conf} - {break_reason}{" "*(9-len(break_reason))} ({iteration+1}{" "*(3-len(str(iteration+1)))} iterations, {time_to_string(time.time()-t_start)})', p=False)

        if check:
            if not molecule_check(original_mol.atomcoords[conf], mol.atomcoords[0], mol.atomnos, max_newbonds=1):
                mol.atomcoords[0] = original_mol.atomcoords[conf]
            # keep the bent structures only if no scrambling occurred between atoms

        final_mol.atomcoords[conf] = mol.atomcoords[0]

    # Now align the ensembles on the new reactive atoms positions

    reference, *targets = final_mol.atomcoords
    reference = np.array(reference)
    targets = np.array(targets)

    r = reference - np.mean(reference[final_mol.reactive_indexes], axis=0)
    ts = np.array([t - np.mean(t[final_mol.reactive_indexes], axis=0) for t in targets])

    output = []
    output.append(r)
    for target in ts:
        matrix = kabsch(r, target)
        output.append([matrix @ vector for vector in target])

    final_mol.atomcoords = np.array(output)

    # Update orbitals and pivots
    for index, atom in final_mol.reactive_atoms_classes_dict.items():
        atom.init(final_mol, index, update=True, orb_dim=orb_memo[index])

    docker._set_pivots(final_mol)

    # add result to cache (if we have it) so we avoid recomputing it
    if hasattr(docker, "ase_bent_mols_dict"):
        docker.ase_bent_mols_dict[(identifier, tuple(sorted(pivot.index)), round(threshold, 3))] = final_mol

    return final_mol

def ase_dump(filename, images, atomnos):
    with open(filename, 'w') as f:
                for i, image in enumerate(images):
                    coords = image.get_positions()
                    write_xyz(coords, atomnos, f, title=f'{filename[:-4]}_image_{i}')