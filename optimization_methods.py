import os, time
import numpy as np
import networkx as nx
from spyrmsd.rmsd import rmsd
from cclib.io import ccread
from ase import Atoms
from ase.visualize import view
from ase.constraints import FixBondLength, FixBondLengths, Hookean
from ase.calculators.mopac import MOPAC
from ase.calculators.gaussian import Gaussian, GaussianOptimizer
from ase.optimize import BFGS
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


def scramble(array, sequence):
    return np.array([array[s] for s in sequence])

def read_mop_out(filename):
    '''
    Reads a MOPAC output looking for optimized coordinates and energy.
    :params filename: name of MOPAC filename (.out extension)
    :return coords, energy: array of optimized coordinates and absolute energy, in kcal/mol
    '''
    coords = []
    with open('temp.out', 'r') as f:
        while True:
            line = f.readline()
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
        return coords, energy
    else:
        raise Exception(f'Cannot read file {filename}: maybe a badly specified MOPAC keyword?')

def mopac_opt(coords, atomnos, constrained_indexes, method='PM7', title='temp'):
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
    order = []
    s = [method + '\n' + title + '\n\n']
    for i, num in enumerate(atomnos):
        if i not in constrained_indexes:
            order.append(i)
            s.append(' {} {} 1 {} 1 {} 1\n'.format(pt[num].symbol, coords[i][0], coords[i][1], coords[i][2]))

    free_indexes = list(set(range(len(atomnos))) - set(constrained_indexes.ravel()))
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

    inv_order = [order.index(i) for i in range(len(order))]
    # undoing the atomic scramble that was needed by the mopac input requirements
    
    try:
        check_call(f'{MOPAC_COMMAND} {title}.mop'.split(), stdout=DEVNULL, stderr=STDOUT)
    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    opt_coords, energy = read_mop_out(f'{title}.out')

    return scramble(opt_coords, inv_order), energy

def optimize(TS_structure, TS_atomnos, constrained_indexes, mols_graphs, method='PM7 GEO-OK', title='temp', debug=False):
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

    opt_coords, energy = mopac_opt(TS_structure, TS_atomnos, constrained_indexes, method=method, title=title)

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

    if len(delta_bonds) > 3:
        not_scrambled = False
    else:
        not_scrambled = True


    return opt_coords, energy, not_scrambled

# def Hookean_optimization(TS_structure, TS_atomnos, constrained_indexes, mols_graphs, calculator, method='PM7', debug=False):
#     '''
#     Performs a geometry partial optimization (POPT) with Gaussian at $method level, 
#     constraining the distance between the specified atom pair. Moreover, it includes a
#     Hookean set of constraints to preserve molecular identities and prevent atom scrambling.

#     :params TS_structure: list of coordinates for each atom in the TS
#     :params TS_atomnos: list of atomic numbers for each atom in the TS
#     :params constrained_indexes: indexes of constrained atoms in the TS geometry
#     :params mols_graphs: list of molecule.graph objects, containing connectivity information
#     :params mols_atomnos: list of molecule.atomnos lists, containing atomic number for all atoms
#     :params method: Level of theory to be used in geometry optimization. Default if UFF.

#     :return opt_struct: optimized structure
#     :return energy: absolute energy
#     '''
#     assert len(TS_structure) == sum([len(graph.nodes) for graph in mols_graphs])

#     atoms = Atoms(''.join([pt[i].symbol for i in TS_atomnos]), positions=TS_structure)

#     # atoms.edit()
#     constraints = []
#     if len(constrained_indexes) == 1:
#         constraints.append(FixBondLength(*constrained_indexes[0]))
#     else:
#         constraints.append(FixBondLengths(constrained_indexes))
        
#     # print(f'Constrained indexes are {constrained_indexes}')

#     rt_dict = {'CH':1.59} # Distance above which Hookean correction kicks in, in Angstroms
#     k_dict  = {'CH':7}    # Spring force constant, in eV/Angstrom^2

#     bonds = []
#     for i, graph in enumerate(mols_graphs):

#         pos = 0
#         while i != 0:
#             pos += len(mols_graphs[i-1].nodes)
#             i -= 1

#         for bond in [(a+pos, b+pos) for a, b in list(graph.edges) if a != b]:
#             bonds.append(bond)
#     # creating bond list containing all bonds present in the desired transition state

#     i = 0
#     for bond_a, bond_b in bonds:
#         key = ''.join(sorted([pt[TS_atomnos[bond_a]].symbol, pt[TS_atomnos[bond_b]].symbol]))
#         try:
#             rt = rt_dict[key]
#             k = k_dict[key]
#             constraints.append(Hookean(a1=bond_a, a2=bond_b, rt=rt, k=k))
#             i += 1
#         except KeyError:
#             pass
#     # print(f'Hookean-protected {i} CH bonds')

#     atoms.set_constraint(constraints)
#     # print('Hookean Constraints are', [c.pairs for c in atoms.constraints if 'pairs' in vars(c)])

#     jobname = 'temp'
#     if calculator == 'Gaussian':
#         atoms.calc = Gaussian(label=jobname, command=f'{GAUSSIAN_COMMAND} {jobname}.com {jobname}.log', method=method)
#         opt = BFGS(atoms, trajectory=f'{jobname}.traj', logfile=f'{jobname}.traj_log')

#     elif calculator == 'Mopac':
#         atoms.calc = MOPAC(label=jobname, command=f'{MOPAC_COMMAND} {jobname}.mop', method=method)
#         opt = BFGS(atoms, trajectory=f'{jobname}.traj', logfile=f'{jobname}.traj_log')

#     try:
#         with suppress_stdout_stderr():
#             opt.run(fmax=0.05)
#     except IndexError as e:
#         # Ase will throw an IndexError if it cannot work out constraints in the
#         # specified partial optimization. We will ignore it here, and return an inifinite energy
#         # return atoms.positions, np.inf
#         raise e

#     try:
#         with suppress_stdout_stderr():
#             energy = atoms.get_total_energy()
#     except Exception as e:
#         raise e
#         # energy = np.inf

#     return atoms.positions, energy

# def prune_conformers(structures, atomnos, k=1, max_rmsd=1, energies=None, debug=False):
# Obsolete, as cython version is faster. NOTE: might still need this one as a fallback, though.
#     '''
#     Initial removal of conformations that are too similar (have a small RMSD value)
#     by splitting the structure set in k subsets and pruning conformations inside those.
    
#     :params structures: numpy array of conformations
#     :params max_rmsd: maximum rmsd value to consider two structures identical, in Angstroms
#     '''
#     if k != 1:
#         r = np.arange(structures.shape[0])
#         sequence = np.random.permutation(r)
#         inv_sequence = np.array([np.where(sequence == i)[0][0] for i in r])

#         structures = scramble(structures, sequence)
#         # scrambling array before splitting, so to improve efficiency when doing
#         # multiple runs of group pruning

#     mask_out = []
#     d = len(structures) // k
#     for step in range(k):
#         if step == k-1:
#             structures_subset = structures[d*step:]
#         else:
#             structures_subset = structures[d*step:d*(step+1)]


#         rmsd_mat = np.zeros((len(structures_subset), len(structures_subset)))
#         rmsd_mat[:] = np.nan
#         for i, tgt in enumerate(structures_subset):
#             for j, ref in enumerate(structures_subset[i+1:]):
#                 rmsd_mat[i, i+j+1] = rmsd(tgt, ref, atomnos, atomnos, center=True, minimize=True)


#         matches = np.where(rmsd_mat < max_rmsd)
#         matches = [(i,j) for i,j in zip(matches[0], matches[1])]

#         g = nx.Graph(matches)

#         if debug:
#             g.add_nodes_from(range(len(structures_subset)))
#             pos = nx.spring_layout(g)
#             nx.draw(g, pos=pos, labels={i:i for i in range(len(g))})
#             import matplotlib.pyplot as plt
#             plt.show()

#         subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
#         groups = [tuple(graph.nodes) for graph in subgraphs]

#         if energies is not None:
#             # if we have energies, the most stable of each cluster is returned
#             def en(tup):
#                 ens = [energies[t] for t in tup]
#                 return tup[ens.index(min(ens))]

#             best_of_cluster = [en(group) for group in groups]

#         else:
#             best_of_cluster = [group[0] for group in groups]
#             # if we do not, the first on of each cluster is returned
#             rejects_sets = [set(a) - {b} for a, b in zip(groups, best_of_cluster)]

#         rejects = []
#         def flatten(seq):
#             for s in seq:
#                 if type(s) in (tuple, list, set):
#                     flatten(s)
#                 else:
#                     rejects.append(s)
#         flatten(rejects_sets)

#         mask = np.array([True for _ in range(len(structures_subset))], dtype=bool)
#         for i in rejects:
#             mask[i] = False

#         mask_out.append(mask)
    
#     mask = np.concatenate(mask_out)

#     if k != 1:
#         mask = scramble(mask, inv_sequence)
#         structures = scramble(structures, inv_sequence)
#         # undoing the previous shuffling, therefore preserving the input order

#     return structures[mask], mask

# if __name__ == '__main__':

#     from hypermolecule_class import Hypermolecule
#     os.chdir(r'C:\Users\Nik\Desktop\Coding\TSCoDe\mopac_tests')

#     mol1 = Hypermolecule('../Resources/indole/indole_ensemble.xyz', 6)
#     mol2 = Hypermolecule('../Resources/SN2/ketone_ensemble.xyz', 2)

#     TS = ccread('TS_out3.xyz')
#     v = (TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph])

    # if compenetration_check(*v, debug=True):
        # Hookean_optimization(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Mopac', method='PM7', debug=True) # faster than Gaussian (8s)
        # Hookean_optimization(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Gaussian', method='PM6', debug=True) # f'in slow (58s)
        # optimize(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Mopac', method='PM6', debug=True) # works
        # optimize(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Gaussian', method='PM6', debug=True)
        # gaussian does not recognize the constraint...

        # checking time cost of Hook+regular compared to regular
        # c, e = Hookean_optimization(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Mopac', method='PM7', debug=True)
        # print(c.shape, TS.atomcoords[0].shape)
        # optimize(c, TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Mopac', method='PM7', debug=True)
        # optimize(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Mopac', method='PM7', debug=True)

    # for i in range(1,9):
    #     TS = ccread(f'TS_out{i}.xyz')
    #     sanity_check(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph])
