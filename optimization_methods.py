import os, time
import numpy as np
import networkx as nx
from spyrmsd.rmsd import rmsd
from cclib.io import ccread
from ase import Atoms
from ase.visualize import view
from ase.constraints import FixBondLength, FixBondLengths, Hookean
# from ase.calculators.mopac import MOPAC
# from ase.calculators.gaussian import Gaussian, GaussianOptimizer
# from ase.optimize import BFGS
from hypermolecule_class import pt, graphize
from parameters import MOPAC_COMMAND
from linalg_tools import norm, dihedral
from subprocess import DEVNULL, STDOUT, check_call

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

def write_orca(coords:np.array, atomnos:np.array, output, head='! PM3 Opt'):
    '''
    output is of _io.TextIOWrapper type

    '''
    assert atomnos.shape[0] == coords.shape[0]
    assert coords.shape[1] == 3
    head += '\n# Orca input from TSCoDe\n\n* xyz 0 1'
    for i, atom in enumerate(coords):
        head += '%s     % .6f % .6f % .6f\n' % (pt[atomnos[i]].symbol, atom[0], atom[1], atom[2])
    head += '*'
    output.write(head)

def write_orca_neb(coords1, coords2, atomnos, title='temp', method='PM3'):

    assert coords1.shape == coords2.shape
    assert atomnos.shape[0] == coords1.shape[0]
    assert coords1.shape[1] == 3

    with open(f'{title}_start.xyz', 'w') as f:
        write_xyz(coords1, atomnos, f, title)

    with open(f'{title}_end.xyz', 'w') as f:
        write_xyz(coords2, atomnos, f, title)

    with open(f'{title}_neb.inp', 'w') as output:
        head = f'! {method} NEB-TS\n\n'
        head += f'%neb\nNEB_End_XYZFile "{title}_end.xyz"\nNimages 6\nend\n\n'
        head += f'# Orca NEB input from TSCoDe\n\n* xyzfile 0 1 {title}_start.xyz'
        output.write(head)

def IRC_plus_NEB(coords, atomnos, mopac_method='PM3 IRC=1* X-PRIORITY=0.1 RESTART', orca_method='PM3', title='temp'):
    '''
    TODO: desc
    '''

    mopac_opt(coords, atomnos, method=mopac_method, title=title+'_IRC', read_output=False)
    os.remove(f'{title}_IRC.out')
    # run mopac IRC calculation

    irc_coords = ccread(f'{title}_IRC.xyz').atomcoords
    os.remove(f'{title}_IRC.xyz')
    # read the IRC structures

    reagent, _, r_bool = mopac_opt(irc_coords[0], atomnos, title='reag')
    product, _, p_bool = mopac_opt(irc_coords[-1], atomnos, title='prod')

    reagent = reagent if r_bool else irc_coords[0]
    product = product if p_bool else irc_coords[-1]
    # we use optimized coordinates of IRC endpoints,
    # but only if their minimization converged

    # with open(f'{title}_reag_prod.xyz', 'w') as f:
    #     write_xyz(reagent, atomnos, f, f'{title}_reagent')
    #     write_xyz(product, atomnos, f, f'{title}_product')

    write_orca_neb(reagent, product, atomnos, method=orca_method, title=title)
    check_call(f'orca {title}_neb.inp'.split(), stdout=DEVNULL, stderr=STDOUT)
    # Run orca NEB calculation from IRC optimized endpoints

    # read output?