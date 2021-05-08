import os, time
import numpy as np
import networkx as nx
from spyrmsd.rmsd import rmsd
from cclib.io import ccread
from ase import Atoms
from ase.visualize import view
from ase.constraints import FixBondLength, FixBondLengths, Hookean
from ase.calculators.mopac import MOPAC
from ase.neb import idpp_interpolate
from ase.dyneb import DyNEB
from ase.optimize import BFGS, LBFGS, FIRE
from hypermolecule_class import pt, graphize
from parameters import MOPAC_COMMAND
from linalg_tools import norm, dihedral
from subprocess import DEVNULL, STDOUT, check_call
from rmsd import kabsch, centroid

# from functools import lru_cache

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

import sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

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
                            energy = line.split()[5]
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
    else:
        raise Exception(f'Cannot read file {filename}: maybe a badly specified MOPAC keyword?')

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

    s = ''.join(s)
    with open(f'{title}.mop', 'w') as f:
        f.write(s)
    
    try:
        check_call(f'{MOPAC_COMMAND} {title}.mop'.split(), stdout=DEVNULL, stderr=STDOUT)
    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    os.remove(f'{title}.mop')
    # delete input, we do not need it anymore

    if read_output:

        inv_order = [order.index(i) for i in range(len(order))]
        # undoing the atomic scramble that was needed by the mopac input requirements

        opt_coords, energy, success = read_mop_out(f'{title}.out')
        os.remove(f'{title}.out')

        opt_coords = scramble(opt_coords, inv_order) if opt_coords is not None else coords
        # If opt_coords is None, that is if TS seeking crashed,
        # sets opt_coords to the old coords. If not, unscrambles
        # coordinates read from mopac output.

        return opt_coords, energy, success

def optimize(TS_structure, TS_atomnos, mols_graphs, constrained_indexes=None, method='PM7 GEO-OK', max_newbonds=2, title='temp', debug=False):
    '''
    Performs a geometry partial optimization (POPT) with Mopac at $method level, 
    constraining the distance between the specified atom pair. Moreover, performs a check of atomic
    pairs distances to ensure to have preserved molecular identities and prevented atom scrambling.

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

    bonds = []
    for i, graph in enumerate(mols_graphs):

        pos = 0
        while i != 0:
            pos += len(mols_graphs[i-1].nodes)
            i -= 1

        for bond in [(a+pos, b+pos) for a, b in list(graph.edges) if a != b]:
            bonds.append(bond)
    bonds = set(bonds)
    # creating bond set containing all bonds present in the desired transition state

    opt_coords, energy, success = mopac_opt(TS_structure, TS_atomnos, constrained_indexes, method=method, title=title)

    new_bonds = {(a, b) for a, b in list(graphize(opt_coords, TS_atomnos).edges) if a != b}
    delta_bonds = (bonds | new_bonds) - (bonds & new_bonds)
    # print('delta_bonds is', list(delta_bonds))

    # delta_bonds -= set(((a,b) for a,b in constrained_indexes))
    # delta_bonds -= set(((b,a) for a,b in constrained_indexes))

    # delta = list(delta_bonds)[:]
    # c_ids = list(constrained_indexes.ravel())
    # for a, b in delta:
    #     if a in c_ids or b in c_ids:
    #         delta_bonds -= {(a, b)}

    if len(delta_bonds) > max_newbonds:
        success = False

    return opt_coords, energy, success

def dump(filename, images, atomnos):
    with open(filename, 'w') as f:
                for i, image in enumerate(images):
                    coords = image.get_positions()
                    write_xyz(coords, atomnos, f, title=f'{filename[:-4]}_image_{i}')

def ase_neb(reagents, products, atomnos, n_images=6, fmax=0.05, method='PM7 GEO-OK', title='temp', optimizer=LBFGS):
    '''
    TODO:desc
    '''

    first = Atoms(atomnos, positions=reagents)
    last = Atoms(atomnos, positions=products)

    images =  [first]
    images += [first.copy() for i in range(n_images)]
    images += [last]

    neb = DyNEB(images, fmax=fmax, climb=False,  method='eb', scale_fmax=1, allow_shared_calculator=True)
    neb.interpolate()

    dump(f'{title}_MEP_guess.xyz', images, atomnos)
    
    # Set calculators for all images
    for i, image in enumerate(images):
        image.calc = MOPAC(label='temp', command=f'{MOPAC_COMMAND} temp.mop', method=method)

    # Set the optimizer and optimize
    try:
        with optimizer(neb, maxstep=0.1) as opt:

            # with suppress_stdout_stderr():
            # with HiddenPrints():
            opt.run(fmax=fmax, steps=20)

            neb.climb = True
            opt.run(fmax=fmax, steps=500)

    except Exception as e:
        print(f'Stopped NEB for {title}:')
        print(e)

    energies = [image.get_total_energy() for image in images]
    ts_id = energies.index(max(energies))
    # print(f'TS structure is number {ts_id}, energy is {max(energies)}')

    os.remove(f'{title}_MEP_guess.xyz')
    dump(f'{title}_MEP.xyz', images, atomnos)
    # Save the converged MEP (minimum energy path) to an .xyz file


    return images[ts_id].get_positions(), images[ts_id].get_total_energy()

def hyperNEB(coords, atomnos, ids, constrained_indexes, reag_prod_method ='PM7', NEB_method='PM7 GEO-OK', title='temp'):
    '''
    Turn a geometry close to TS to a proper TS by getting
    reagents and products and running a climbing image NEB calculation through ASE.
    '''

    reagents = get_reagent(coords, atomnos, ids, constrained_indexes, method=reag_prod_method)
    products = get_product(coords, atomnos, ids, constrained_indexes, method=reag_prod_method)
    # get reagents and products for this reaction

    # reagents -= centroid(reagents[constrained_indexes.ravel()])
    # products -= centroid(products[constrained_indexes.ravel()])
    # centering both structures on the centroid of reactive atoms

    # align the two reagents and products structures to the TS (minimal RMSD)?

    ts_coords, ts_energy = ase_neb(reagents, products, coords, atomnos, method=NEB_method, optimizer=LBFGS, title=title)
    # Use these structures plus the TS guess to run a NEB calculation through ASE

    return ts_coords, ts_energy

def get_product(coords, atomnos, ids, constrained_indexes, method='PM7'):
    '''
    TODO:desc
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
        # superimposes its reactive centers to the ones of mol2

        threshold_dists = [bond_factor*(pt[atomnos[a]].covalent_radius + pt[atomnos[b]].covalent_radius) for a, b in constrained_indexes]

        reactive_dists = [np.linalg.norm(coords[a] - coords[b]) for a, b in constrained_indexes]
        # distances between reactive atoms

        while not np.all([reactive_dists[i] < threshold_dists[i] for i in range(len(constrained_indexes))]):
            # print('Reactive distances are', reactive_dists)

            coords[:ids[0]] += motion*step_size

            coords, _, _ = mopac_opt(coords, atomnos, constrained_indexes, method=method)

            reactive_dists = [np.linalg.norm(coords[a] - coords[b]) for a, b in constrained_indexes]

        newcoords, _, _ = mopac_opt(coords, atomnos, method=method)
        # finally, when structures are close enough, do a free optimization to get the reaction product

        new_reactive_dists = [np.linalg.norm(newcoords[a] - newcoords[b]) for a, b in constrained_indexes]

        if np.all([new_reactive_dists[i] < threshold_dists[i] for i in range(len(constrained_indexes))]):
        # return the freely optimized structure only if the reagents did not repel each other
        # during the optimization, otherwise return the last coords, where partners were close
            return newcoords
        else:
            return coords

    else:
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
        else:
            return coords

def get_reagent(coords, atomnos, ids, constrained_indexes, method='PM7'):
    '''
    TODO:desc
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

    else:
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

        coords, _, _ = mopac_opt(coords, atomnos, constrained_indexes, method=method)
        # when all atoms are moved, optimize the geometry with the previous constraints

        newcoords, _, _ = mopac_opt(coords, atomnos, method=method)
        # finally, when structures are close enough, do a free optimization to get the reaction product

        new_reactive_dist = np.linalg.norm(newcoords[constrained_indexes[0,0]] - newcoords[constrained_indexes[0,0]])

        if new_reactive_dist > threshold_dist:
        # return the freely optimized structure only if the reagents did not approached back each other
        # during the optimization, otherwise return the last coords, where partners were further away
            return newcoords
        else:
            return coords



# def write_orca(coords:np.array, atomnos:np.array, output, head='! PM3 Opt'):
#     '''
#     output is of _io.TextIOWrapper type

#     '''
#     assert atomnos.shape[0] == coords.shape[0]
#     assert coords.shape[1] == 3
#     head += '\n# Orca input from TSCoDe\n\n* xyz 0 1'
#     for i, atom in enumerate(coords):
#         head += '%s     % .6f % .6f % .6f\n' % (pt[atomnos[i]].symbol, atom[0], atom[1], atom[2])
#     head += '*'
#     output.write(head)

# def write_orca_neb(coords1, coords2, atomnos, title='temp', method='PM3'):

#     assert coords1.shape == coords2.shape
#     assert atomnos.shape[0] == coords1.shape[0]
#     assert coords1.shape[1] == 3

#     with open(f'{title}_start.xyz', 'w') as f:
#         write_xyz(coords1, atomnos, f, title)

#     with open(f'{title}_end.xyz', 'w') as f:
#         write_xyz(coords2, atomnos, f, title)

#     with open(f'{title}_neb.inp', 'w') as output:
#         head = f'! {method} NEB-TS\n\n'
#         head += f'%neb\nNEB_End_XYZFile "{title}_end.xyz"\nNimages 6\nend\n\n'
#         head += f'# Orca NEB input from TSCoDe\n\n* xyzfile 0 1 {title}_start.xyz'
#         output.write(head)