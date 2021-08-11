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
from subprocess import DEVNULL, STDOUT, check_call

import numpy as np
import networkx as nx
from cclib.io import ccread
from scipy.spatial.transform import Rotation as R

from settings import COMMANDS, MEM_GB
from hypermolecule_class import graphize
from utils import (
                   center_of_mass,
                   clean_directory,
                   diagonalize,
                   dihedral,
                   kronecker_delta,
                   norm,
                   pt,
                   write_xyz,
                   scramble_check,
                   molecule_check,
                   )

from ase_manipulations import ase_neb

class MopacReadError(Exception):
    '''
    Thrown when reading MOPAC output files fails for some reason.
    '''

def scramble(array, sequence):
    return np.array([array[s] for s in sequence])

def read_mop_out(filename):
    '''
    Reads a MOPAC output looking for optimized coordinates and energy.
    :params filename: name of MOPAC filename (.out extension)
    :return coords, energy: array of optimized coordinates and absolute energy, in kcal/mol
    '''
    coords = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()

            if 'Too many variables. By definition, at least one force constant is exactly zero' in line:
                success = False
                return None, np.inf, success

            if not line:
                break
            
            if 'SCF FIELD WAS ACHIEVED' in line:
                    while True:
                        line = f.readline()
                        if not line:
                            break

                        if 'FINAL HEAT OF FORMATION' in line:
                            energy = float(line.split()[5])
                            # in kcal/mol

                        if 'CARTESIAN COORDINATES' in line:
                            line = f.readline()
                            line = f.readline()
                            while line != '\n':
                                splitted = line.split()
                                # symbols.append(splitted[1])
                                coords.append([float(splitted[2]),
                                               float(splitted[3]),
                                               float(splitted[4])])
                                            
                                line = f.readline()
                                if not line:
                                    break
                            break
                    break

    coords = np.array(coords)

    if coords.shape[0] != 0:
        success = True
        return coords, energy, success
    
    raise MopacReadError(f'Cannot read file {filename}: maybe a badly specified MOPAC keyword?')

def mopac_opt(coords, atomnos, constrained_indexes=None, method='PM7', title='temp', read_output=True):
    '''
    This function writes a MOPAC .mop input, runs it with the subprocess
    module and reads its output. Coordinates used are mixed
    (cartesian and internal) to be able to constrain the reactive atoms
    distances specified in constrained_indexes.

    :params coords: array of shape (n,3) with cartesian coordinates for atoms
    :params atomnos: array of atomic numbers for atoms
    :params constrained_indexes: array of shape (n,2), with the indexes
                                 of atomic pairs to be constrained
    :params method: string, specifiyng the first line of keywords for the MOPAC input file.
    :params title: string, used as a file name and job title for the mopac input file.
    :params read_output: Whether to read the output file and return anything.
    '''

    constrained_indexes_list = constrained_indexes.ravel() if constrained_indexes is not None else []
    constrained_indexes = constrained_indexes if constrained_indexes is not None else []

    order = []
    s = [method + '\n' + title + '\n\n']
    for i, num in enumerate(atomnos):
        if i not in constrained_indexes:
            order.append(i)
            s.append(' {} {} 1 {} 1 {} 1\n'.format(pt[num].symbol, coords[i][0], coords[i][1], coords[i][2]))

    free_indexes = list(set(range(len(atomnos))) - set(constrained_indexes_list))
    # print('free indexes are', free_indexes, '\n')

    if len(constrained_indexes_list) == len(set(constrained_indexes_list)):
    # block pairs of atoms if no atom is involved in more than one distance constrain

        for a, b in constrained_indexes:
            
            order.append(b)
            order.append(a)

            c, d = np.random.choice(free_indexes, 2)
            while c == d:
                c, d = np.random.choice(free_indexes, 2)
            # indexes of reference atoms, from unconstraind atoms set

            dist = np.linalg.norm(coords[a] - coords[b]) # in Angstrom
            # print(f'DIST - {dist} - between {a} {b}')

            angle = np.arccos(norm(coords[a] - coords[b]) @ norm(coords[c] - coords[b]))*180/np.pi # in degrees
            # print(f'ANGLE - {angle} - between {a} {b} {c}')

            d_angle = dihedral([coords[a],
                                coords[b],
                                coords[c],
                                coords[d]])
            d_angle += 360 if d_angle < 0 else 0
            # print(f'D_ANGLE - {d_angle} - between {a} {b} {c} {d}')

            list_len = len(s)
            s.append(' {} {} 1 {} 1 {} 1\n'.format(pt[atomnos[b]].symbol, coords[b][0], coords[b][1], coords[b][2]))
            s.append(' {} {} 0 {} 1 {} 1 {} {} {}\n'.format(pt[atomnos[a]].symbol, dist, angle, d_angle, list_len, free_indexes.index(c)+1, free_indexes.index(d)+1))
            # print(f'Blocked bond between mopac ids {list_len} {list_len+1}\n')

    elif len(set(constrained_indexes_list)) == 3:
    # three atoms, the central bound to the other two
    # OTHERS[0]: cartesian
    # CENTRAL: internal (self, others[0], two random)
    # OTHERS[1]: internal (self, central, two random)
        
        central = max(set(constrained_indexes_list), key=lambda x: list(constrained_indexes_list).count(x))
        # index of the atom that is constrained to two other

        others = list(set(constrained_indexes_list) - {central})

    # OTHERS[0]

        order.append(others[0])
        s.append(' {} {} 1 {} 1 {} 1\n'.format(pt[atomnos[others[0]]].symbol, coords[others[0]][0], coords[others[0]][1], coords[others[0]][2]))
        # first atom is placed in cartesian coordinates, the other two have a distance constraint and are expressed in internal coordinates

    #CENTRAL

        order.append(central)
        c, d = np.random.choice(free_indexes, 2)
        while c == d:
            c, d = np.random.choice(free_indexes, 2)
        # indexes of reference atoms, from unconstraind atoms set

        dist = np.linalg.norm(coords[central] - coords[others[0]]) # in Angstrom

        angle = np.arccos(norm(coords[central] - coords[others[0]]) @ norm(coords[others[0]] - coords[c]))*180/np.pi # in degrees

        d_angle = dihedral([coords[central],
                            coords[others[0]],
                            coords[c],
                            coords[d]])
        d_angle += 360 if d_angle < 0 else 0

        list_len = len(s)
        s.append(' {} {} 0 {} 1 {} 1 {} {} {}\n'.format(pt[atomnos[central]].symbol, dist, angle, d_angle, list_len-1, free_indexes.index(c)+1, free_indexes.index(d)+1))

    #OTHERS[1]

        order.append(others[1])
        c1, d1 = np.random.choice(free_indexes, 2)
        while c1 == d1:
            c1, d1 = np.random.choice(free_indexes, 2)
        # indexes of reference atoms, from unconstraind atoms set

        dist1 = np.linalg.norm(coords[others[1]] - coords[central]) # in Angstrom

        angle1 = np.arccos(norm(coords[others[1]] - coords[central]) @ norm(coords[others[1]] - coords[c1]))*180/np.pi # in degrees

        d_angle1 = dihedral([coords[others[1]],
                             coords[central],
                             coords[c1],
                             coords[d1]])
        d_angle1 += 360 if d_angle < 0 else 0

        list_len = len(s)
        s.append(' {} {} 0 {} 1 {} 1 {} {} {}\n'.format(pt[atomnos[others[1]]].symbol, dist1, angle1, d_angle1, list_len-1, free_indexes.index(c1)+1, free_indexes.index(d1)+1))

    else:
        raise NotImplementedError('The constraints provided for MOPAC optimization are not yet supported')


    s = ''.join(s)
    with open(f'{title}.mop', 'w') as f:
        f.write(s)
    
    try:
        check_call(f'{COMMANDS["MOPAC"]} {title}.mop'.split(), stdout=DEVNULL, stderr=STDOUT)
    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    os.remove(f'{title}.mop')
    # delete input, we do not need it anymore

    if read_output:

        inv_order = [order.index(i) for i, _ in enumerate(order)]
        # undoing the atomic scramble that was needed by the mopac input requirements

        opt_coords, energy, success = read_mop_out(f'{title}.out')
        os.remove(f'{title}.out')

        opt_coords = scramble(opt_coords, inv_order) if opt_coords is not None else coords
        # If opt_coords is None, that is if TS seeking crashed,
        # sets opt_coords to the old coords. If not, unscrambles
        # coordinates read from mopac output.

        return opt_coords, energy, success

def orca_opt(coords, atomnos, constrained_indexes=None, method='PM3', procs=1, title='temp', read_output=True):
    '''
    This function writes an ORCA .inp file, runs it with the subprocess
    module and reads its output.

    :params coords: array of shape (n,3) with cartesian coordinates for atoms.
    :params atomnos: array of atomic numbers for atoms.
    :params constrained_indexes: array of shape (n,2), with the indexes
                                 of atomic pairs to be constrained.
    :params method: string, specifiyng the first line of keywords for the MOPAC input file.
    :params title: string, used as a file name and job title for the mopac input file.
    :params read_output: Whether to read the output file and return anything.
    '''

    s = '! %s Opt\n\n# ORCA input generated by TSCoDe\n\n' % (method)

    if procs > 1:
        s += f'%pal nprocs {procs} end\n'

    if constrained_indexes is not None:
        s += '%geom\nConstraints\n'

        for a, b in constrained_indexes:
            s += '{B %s %s C}\n' % (a, b)

        s += 'end\nend\n\n'

    s += '*xyz 0 1\n'

    for i, atom in enumerate(coords):
        s += '%s     % .6f % .6f % .6f\n' % (pt[atomnos[i]].symbol, atom[0], atom[1], atom[2])

    s += '*\n'

    s = ''.join(s)
    with open(f'{title}.inp', 'w') as f:
        f.write(s)
    
    try:
        check_call(f'{COMMANDS["ORCA"]} {title}.inp'.split(), stdout=DEVNULL, stderr=STDOUT)

    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    if read_output:

        try:
            opt_coords = ccread(f'{title}.xyz').atomcoords[0]
            energy = read_orca_property(f'{title}_property.txt')

            clean_directory()

            return opt_coords, energy, True

        except FileNotFoundError:
            return None, None, False

def read_orca_property(filename):
    '''
    Read energy from ORCA property output file
    '''
    energy = None

    with open(filename, 'r') as f:

        while True:
            line = f.readline()

            if not line:
                break

            if 'SCF Energy:' in line:
                energy = float(line.split()[2])

    return energy

def gaussian_opt(coords, atomnos, constrained_indexes=None, method='PM6', procs=1, title='temp', read_output=True):
    '''
    This function writes a Gaussian .inp file, runs it with the subprocess
    module and reads its output.

    :params coords: array of shape (n,3) with cartesian coordinates for atoms.
    :params atomnos: array of atomic numbers for atoms.
    :params constrained_indexes: array of shape (n,2), with the indexes
                                 of atomic pairs to be constrained.
    :params method: string, specifiyng the first line of keywords for the MOPAC input file.
    :params title: string, used as a file name and job title for the mopac input file.
    :params read_output: Whether to read the output file and return anything.
    '''

    s = ''

    if MEM_GB is not None:
        s += f'%mem{MEM_GB}GB\n'

    if procs > 1:
        s += f'%nprocshared={procs}\n'

    s += '# ' + method
    
    if constrained_indexes is not None:
        s += 'opt=modredundant'
        
    s += '\n\nGaussian input generated by TSCoDe\n\n0 1\n'

    for i, atom in enumerate(coords):
        s += '%s     % .6f % .6f % .6f\n' % (pt[atomnos[i]].symbol, atom[0], atom[1], atom[2])

    s += '\n'

    if constrained_indexes is not None:

        for a, b in constrained_indexes:
            s += 'B %s %s F\n' % (a, b)

    s = ''.join(s)
    with open(f'{title}.com', 'w') as f:
        f.write(s)
    
    try:
        check_call(f'{COMMANDS["GAUSSIAN"]} {title}.com'.split(), stdout=DEVNULL, stderr=STDOUT)

    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    if read_output:

        try:
            data = ccread(f'{title}.out')
            opt_coords = data.atomcoords[0]
            energy = data.scfenergies[-1] * 23.060548867 # eV to kcal/mol

            clean_directory()

            return opt_coords, energy, True

        except FileNotFoundError:
            return None, None, False

def xtb_opt(coords, atomnos, constrained_indexes=None, method='GFN2-xTB', title='temp', read_output=True):
    '''
    This function writes an XTB .inp file, runs it with the subprocess
    module and reads its output.

    :params coords: array of shape (n,3) with cartesian coordinates for atoms.
    :params atomnos: array of atomic numbers for atoms.
    :params constrained_indexes: array of shape (n,2), with the indexes
                                 of atomic pairs to be constrained.
    :params method: string, specifiyng the first line of keywords for the MOPAC input file.
    :params title: string, used as a file name and job title for the mopac input file.
    :params read_output: Whether to read the output file and return anything.
    '''

    with open(f'{title}.xyz', 'w') as f:
        write_xyz(coords, atomnos, f, title=title)

    s = f'$opt\n   logfile={title}_opt.log\n$end'
         
    if constrained_indexes is not None:
        s += '\n$constrain\n'
        for a, b in constrained_indexes:
            s += '   distance: %s, %s, %s\n' % (a+1, b+1, round(np.linalg.norm(coords[a]-coords[b]), 5))
    
    if method.upper() in ('GFN-XTB', 'GFNXTB'):
        s += '\n$gfn\n   method=1\n'

    elif method.upper() in ('GFN2-XTB', 'GFN2XTB'):
        s += '\n$gfn\n   method=2\n'
    
    s += '\n$end'

    s = ''.join(s)
    with open(f'{title}.inp', 'w') as f:
        f.write(s)
    
    flags = '--opt'
    if method in ('GFN-FF', 'GFNFF'):
        flags += ' --gfnff'

    try:
        check_call(f'xtb --input {title}.inp {title}.xyz {flags} > temp.log 2>&1'.split(), stdout=DEVNULL, stderr=STDOUT)

    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    if read_output:

        try:
            outname = 'xtbopt.xyz'
            opt_coords = ccread(outname).atomcoords[0]
            energy = read_xtb_energy(outname)

            clean_directory()
            os.remove(outname)

            for filename in ('gfnff_topo', 'charges', 'wbo', 'xtbrestart', 'xtbtopo.mol', '.xtboptok'):
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    pass

            return opt_coords, energy, True

        except FileNotFoundError:
            return None, None, False

def read_xtb_energy(filename):
    '''
    returns energy in kcal/mol from an XTB
    .xyz result file (xtbotp.xyz)
    '''
    with open(filename, 'r') as f:
        line = f.readline()
        line = f.readline() # second line is where energy is printed
        return float(line.split()[1]) * 627.5096080305927 # Eh to kcal/mol

def optimize(calculator, TS_structure, TS_atomnos, mols_graphs, constrained_indexes=None, method='PM6', procs=1, max_newbonds=0, title='temp', debug=False):
    '''
    Performs a geometry partial optimization (POPT) with MOPAC, ORCA or Gaussian at $method level, 
    constraining the distance between the specified atom pair. Moreover, performs a check of atomic
    pairs distances to ensure to have preserved molecular identities and prevented atom scrambling.

    :params calculator: Calculator to be used. Either 'MOPAC' or 'ORCA'
    :params TS_structure: list of coordinates for each atom in the TS
    :params TS_atomnos: list of atomic numbers for each atom in the TS
    :params constrained_indexes: indexes of constrained atoms in the TS geometry
    :params mols_graphs: list of molecule.graph objects, containing connectivity information
    :params method: Level of theory to be used in geometry optimization. Default if UFF.

    :return opt_coords: optimized structure
    :return energy: absolute energy of structure, in kcal/mol
    :return not_scrambled: bool, indicating if the optimization shifted up some bonds (except the constrained ones)
    '''
    assert len(TS_structure) == sum([len(graph.nodes) for graph in mols_graphs])

    if constrained_indexes is None:
        constrained_indexes = np.array(())

    if calculator == 'MOPAC':
        opt_coords, energy, success = mopac_opt(TS_structure, TS_atomnos, constrained_indexes, method=method, title=title)

    elif calculator == 'ORCA':
        opt_coords, energy, success = orca_opt(TS_structure, TS_atomnos, constrained_indexes, method=method, procs=procs, title=title)

    elif calculator == 'GAUSSIAN':
        opt_coords, energy, success = gaussian_opt(TS_structure, TS_atomnos, constrained_indexes, method=method, procs=procs, title=title)

    elif calculator == 'XTB':
        opt_coords, energy, success = xtb_opt(TS_structure, TS_atomnos, constrained_indexes, method=method, title=title)


    if success:
        success = scramble_check(opt_coords, TS_atomnos, constrained_indexes, mols_graphs, max_newbonds=max_newbonds)

    return opt_coords, energy, success

def hyperNEB(docker, coords, atomnos, ids, constrained_indexes, title='temp'):
    '''
    Turn a geometry close to TS to a proper TS by getting
    reagents and products and running a climbing image NEB calculation through ASE.
    '''

    reagents = get_reagent(coords, atomnos, ids, constrained_indexes, method=docker.options.theory_level)
    products = get_product(coords, atomnos, ids, constrained_indexes, method=docker.options.theory_level)
    # get reagents and products for this reaction

    reagents -= np.mean(reagents, axis=0)
    products -= np.mean(products, axis=0)
    # centering both structures on the centroid of reactive atoms

    aligment_rotation = R.align_vectors(reagents, products)
    products = np.array([aligment_rotation @ v for v in products])
    # rotating the two structures to minimize differences

    ts_coords, ts_energy = ase_neb(docker, reagents, products, atomnos, title=title)
    # Use these structures plus the TS guess to run a NEB calculation through ASE

    return ts_coords, ts_energy

def get_product(coords, atomnos, ids, constrained_indexes, method='PM7'):
    '''
    Part of the automatic NEB implementation.
    Returns a structure that presumably is the association reaction product
    ([cyclo]additions reactions in mind)
    '''

    bond_factor = 1.2
    # multiple of sum of covalent radii for two atoms.
    # If two atoms are closer than this times their sum
    # of c_radii, they are considered to converge to
    # products when their geometry is optimized. 

    step_size = 0.1
    # in Angstroms

    if len(ids) == 2:

        mol1_center = np.mean([coords[a] for a, _ in constrained_indexes], axis=0)
        mol2_center = np.mean([coords[b] for _, b in constrained_indexes], axis=0)
        motion = norm(mol2_center - mol1_center)
        # norm of the motion that, when applied to mol1,
        # superimposes its reactive atoms to the ones of mol2

        threshold_dists = [bond_factor*(pt[atomnos[a]].covalent_radius +
                                        pt[atomnos[b]].covalent_radius) for a, b in constrained_indexes]

        reactive_dists = [np.linalg.norm(coords[a] - coords[b]) for a, b in constrained_indexes]
        # distances between reactive atoms

        while not np.all([reactive_dists[i] < threshold_dists[i] for i, _ in enumerate(constrained_indexes)]):
            # print('Reactive distances are', reactive_dists)

            coords[:ids[0]] += motion*step_size

            coords, _, _ = mopac_opt(coords, atomnos, constrained_indexes, method=method)

            reactive_dists = [np.linalg.norm(coords[a] - coords[b]) for a, b in constrained_indexes]

        newcoords, _, _ = mopac_opt(coords, atomnos, method=method)
        # finally, when structures are close enough, do a free optimization to get the reaction product

        new_reactive_dists = [np.linalg.norm(newcoords[a] - newcoords[b]) for a, b in constrained_indexes]

        if np.all([new_reactive_dists[i] < threshold_dists[i] for i, _ in enumerate(constrained_indexes)]):
        # return the freely optimized structure only if the reagents did not repel each other
        # during the optimization, otherwise return the last coords, where partners were close
            return newcoords

        return coords

    # trimolecular TSs: the approach is to bring the first pair of reactive
    # atoms closer until optimization bounds the molecules together

    index_to_be_moved = constrained_indexes[0,0]
    reference = constrained_indexes[0,1]
    moving_molecule_index = next(i for i,n in enumerate(np.cumsum(ids)) if index_to_be_moved < n)
    bounds = [0] + [n+1 for n in np.cumsum(ids)]
    moving_molecule_slice = slice(bounds[moving_molecule_index], bounds[moving_molecule_index+1])
    threshold_dist = bond_factor*(pt[atomnos[constrained_indexes[0,0]]].covalent_radius +
                                    pt[atomnos[constrained_indexes[0,1]]].covalent_radius)

    motion = (coords[reference] - coords[index_to_be_moved])
    # vector from the atom to be moved to the target reactive atom

    while np.linalg.norm(motion) > threshold_dist:
    # check if the reactive atoms are sufficiently close to converge to products

        for i, atom in enumerate(coords[moving_molecule_slice]):
            dist = np.linalg.norm(atom - coords[index_to_be_moved])
            # for any atom in the molecule, distance from the reactive atom

            atom_step = step_size*np.exp(-0.5*dist)
            coords[moving_molecule_slice][i] += norm(motion)*atom_step
            # the more they are close, the more they are moved

        # print('Reactive dist -', np.linalg.norm(motion))
        coords, _, _ = mopac_opt(coords, atomnos, constrained_indexes, method=method)
        # when all atoms are moved, optimize the geometry with the previous constraints

        motion = (coords[reference] - coords[index_to_be_moved])

    newcoords, _, _ = mopac_opt(coords, atomnos, method=method)
    # finally, when structures are close enough, do a free optimization to get the reaction product

    new_reactive_dist = np.linalg.norm(newcoords[constrained_indexes[0,0]] - newcoords[constrained_indexes[0,0]])

    if new_reactive_dist < threshold_dist:
    # return the freely optimized structure only if the reagents did not repel each other
    # during the optimization, otherwise return the last coords, where partners were close
        return newcoords

    return coords

def get_reagent(coords, atomnos, ids, constrained_indexes, method='PM7'):
    '''
    Part of the automatic NEB implementation.
    Returns a structure that presumably is the association reaction reagent.
    ([cyclo]additions reactions in mind)
    '''

    bond_factor = 1.5
    # multiple of sum of covalent radii for two atoms.
    # Putting reactive atoms at this times their bonding
    # distance and performing a constrained optimization
    # is the way to get a good guess for reagents structure. 

    if len(ids) == 2:

        mol1_center = np.mean([coords[a] for a, _ in constrained_indexes], axis=0)
        mol2_center = np.mean([coords[b] for _, b in constrained_indexes], axis=0)
        motion = norm(mol2_center - mol1_center)
        # norm of the motion that, when applied to mol1,
        # superimposes its reactive centers to the ones of mol2

        threshold_dists = [bond_factor*(pt[atomnos[a]].covalent_radius + pt[atomnos[b]].covalent_radius) for a, b in constrained_indexes]

        reactive_dists = [np.linalg.norm(coords[a] - coords[b]) for a, b in constrained_indexes]
        # distances between reactive atoms

        coords[:ids[0]] -= norm(motion)*(np.mean(threshold_dists) - np.mean(reactive_dists))
        # move reactive atoms away from each other just enough

        coords, _, _ = mopac_opt(coords, atomnos, constrained_indexes=constrained_indexes, method=method)
        # optimize the structure but keeping the reactive atoms distanced

        return coords

    # trimolecular TSs: the approach is to bring the first pair of reactive
    # atoms apart just enough to get a good approximation for reagents

    index_to_be_moved = constrained_indexes[0,0]
    reference = constrained_indexes[0,1]
    moving_molecule_index = next(i for i,n in enumerate(np.cumsum(ids)) if index_to_be_moved < n)
    bounds = [0] + [n+1 for n in np.cumsum(ids)]
    moving_molecule_slice = slice(bounds[moving_molecule_index], bounds[moving_molecule_index+1])
    threshold_dist = bond_factor*(pt[atomnos[constrained_indexes[0,0]]].covalent_radius +
                                    pt[atomnos[constrained_indexes[0,1]]].covalent_radius)

    motion = (coords[reference] - coords[index_to_be_moved])
    # vector from the atom to be moved to the target reactive atom

    displacement = norm(motion)*(threshold_dist-np.linalg.norm(motion))
    # vector to be applied to the reactive atom to push it far just enough

    for i, atom in enumerate(coords[moving_molecule_slice]):
        dist = np.linalg.norm(atom - coords[index_to_be_moved])
        # for any atom in the molecule, distance from the reactive atom

        coords[moving_molecule_slice][i] -= displacement*np.exp(-0.5*dist)
        # the closer they are to the reactive atom, the further they are moved

    coords, _, _ = mopac_opt(coords, atomnos, constrained_indexes=np.array([constrained_indexes[0]]), method=method)
    # when all atoms are moved, optimize the geometry with only the first of the previous constraints

    newcoords, _, _ = mopac_opt(coords, atomnos, method=method)
    # finally, when structures are close enough, do a free optimization to get the reaction product

    new_reactive_dist = np.linalg.norm(newcoords[constrained_indexes[0,0]] - newcoords[constrained_indexes[0,0]])

    if new_reactive_dist > threshold_dist:
    # return the freely optimized structure only if the reagents did not approached back each other
    # during the optimization, otherwise return the last coords, where partners were further away
        return newcoords
    
    return coords

def get_inertia_moments(coords, atomnos):
    '''
    Returns the diagonal of the diagonalized inertia tensor, that is
    a shape (3,) array with the moments of inertia along the main axes.
    (I_x, I_y and largest I_z last)
    '''

    coords -= center_of_mass(coords, atomnos)
    inertia_moment_matrix = np.zeros((3,3))

    for i in range(3):
        for j in range(3):
            k = kronecker_delta(i,j)
            inertia_moment_matrix[i][j] = np.sum([pt[atomnos[n]].mass*((np.linalg.norm(coords[n])**2)*k - coords[n][i]*coords[n][j]) for n, _ in enumerate(atomnos)])

    inertia_moment_matrix = diagonalize(inertia_moment_matrix)

    return np.diag(inertia_moment_matrix)

def prune_enantiomers(structures, atomnos, max_delta=10):
    '''
    Remove duplicate (enantiomeric) structures based on the
    moments of inertia on principal axes. If all three MOI
    are within max_delta from another structure, they are
    classified as enantiomers and therefore only one of them
    is kept.
    '''

    l = len(structures)
    mat = np.zeros((l,l), dtype=int)
    for i in range(l):
        for j in range(i+1,l):
            im_i = get_inertia_moments(structures[i], atomnos)
            im_j = get_inertia_moments(structures[j], atomnos)
            delta = np.abs(im_i - im_j)
            mat[i,j] = 1 if np.all(delta < max_delta) else 0

    where = np.where(mat == 1)
    matches = [(i,j) for i,j in zip(where[0], where[1])]

    g = nx.Graph(matches)

    subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
    groups = [tuple(graph.nodes) for graph in subgraphs]

    best_of_cluster = [group[0] for group in groups]

    rejects_sets = [set(a) - {b} for a, b in zip(groups, best_of_cluster)]
    rejects = []
    for s in rejects_sets:
        for i in s:
            rejects.append(i)

    mask = np.array([True for _ in range(l)], dtype=bool)
    for i in rejects:
        mask[i] = False

    return structures[mask], mask

def xtb_metadyn_augmentation(coords, atomnos, constrained_indexes=None, new_structures:int=5, title=0, debug=False):
    '''
    Runs a metadynamics simulation (MTD) through
    the XTB program to obtain new conformations.
    The GFN-FF force field is used.
    '''
    with open(f'temp.xyz', 'w') as f:
        write_xyz(coords, atomnos, f, title='temp')

    s = (
        '$md\n'
        '   time=%s\n' % (new_structures) +
        '   step=1\n'
        '   temp=300\n'
        '$end\n'
        '$metadyn\n'
        '   save=%s\n' % (new_structures) +
        '$end'
        )
         
    if constrained_indexes is not None:
        s += '\n$constrain\n'
        for a, b in constrained_indexes:
            s += '   distance: %s, %s, %s\n' % (a+1, b+1, round(np.linalg.norm(coords[a]-coords[b]), 5))

    s = ''.join(s)
    with open(f'temp.inp', 'w') as f:
        f.write(s)

    try:
        check_call(f'xtb --md --input temp.inp temp.xyz --gfnff > Structure{title}_MTD.log 2>&1'.split(), stdout=DEVNULL, stderr=STDOUT)

    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    structures = [coords]
    for n in range(1,new_structures):
        name = 'scoord.'+str(n)
        structures.append(parse_xtb_out(name))
        os.remove(name)

    for filename in ('gfnff_topo', 'xtbmdoc', 'mdrestart'):
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

    # if debug:
    os.rename('xtb.trj', f'Structure{title}_MTD_traj.xyz')

    # else:
    #     os.remove('xtb.traj')  

    structures = np.array(structures)

    return structures

def parse_xtb_out(filename):
    '''
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()

    coords = np.zeros((len(lines)-3,3))

    for l, line in enumerate(lines[1:-2]):
        coords[l] = line.split()[:-1]

    return coords * 0.529177249 # Bohrs to Angstroms

from settings import OPENBABEL_OPT_BOOL

if OPENBABEL_OPT_BOOL:
    
    from openbabel import openbabel as ob

    def openbabel_opt(structure, atomnos, constrained_indexes, graphs, method='UFF'):
        '''
        return : MM-optimized structure (UFF/MMFF)
        '''

        filename='temp_ob_in.xyz'

        with open(filename, 'w') as f:
            write_xyz(structure, atomnos, f)

        outname = 'temp_ob_out.xyz'

        # Standard openbabel molecule load
        conv = ob.OBConversion()
        conv.SetInAndOutFormats('xyz','xyz')
        mol = ob.OBMol()
        more = conv.ReadFile(mol, filename)
        i = 0

        # Define constraints
        constraints = ob.OBFFConstraints()

        for a, b in constrained_indexes:

            first_atom = mol.GetAtom(int(a+1))
            length = first_atom.GetDistance(int(b+1))

            constraints.AddDistanceConstraint(int(a+1), int(b+1), length)       # Angstroms
            # constraints.AddAngleConstraint(1, 2, 3, 120.0)      # Degrees
            # constraints.AddTorsionConstraint(1, 2, 3, 4, 180.0) # Degrees

        # Setup the force field with the constraints
        forcefield = ob.OBForceField.FindForceField(method)
        forcefield.Setup(mol, constraints)
        forcefield.SetConstraints(constraints)

        # Do a 500 steps conjugate gradient minimization
        # (or less if converges) and save the coordinates to mol.
        forcefield.ConjugateGradients(500)
        forcefield.GetCoordinates(mol)

        # Write the mol to a file
        conv.WriteFile(mol,outname)
        conv.CloseOutFile()

        opt_coords = ccread(outname).atomcoords[0]

        success = scramble_check(opt_coords, atomnos, constrained_indexes, graphs)

        return opt_coords, success