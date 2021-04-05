import os, time
import numpy as np
import cynetworkx as nx
from spyrmsd.rmsd import rmsd
from cclib.io import ccread
from ase import Atoms
from ase.visualize import view
from ase.constraints import FixBondLength, FixBondLengths, Hookean
from ase.calculators.mopac import MOPAC
from ase.calculators.gaussian import Gaussian, GaussianOptimizer
from ase.optimize import BFGS
from hypermolecule_class import pt, graphize
from parameters import GAUSSIAN_COMMAND, MOPAC_COMMAND
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


def sanity_check(TS_structure, TS_atomnos, constrained_indexes, mols_graphs, max_new_bonds=3, debug=False):
    '''
    :params TS_structure: list of coordinates for each atom in the TS
    :params TS_atomnos: list of atomic numbers for each atom in the TS
    :params constrained_indexes: indexes of constrained atoms in the TS geometry
    :params mols_graphs: list of molecule.graph objects, containing connectivity information
    :params max_new_bonds: maximum number of apperent new bonds in TS geometry to accept the
                           structure as a valid one. Too high values may cause ugly results,
                           too low might discard structures that would have led to good results.

    :return result: bool, indicating structure sanity
    '''
    bonds = []
    for i in range(len(mols_graphs)):
        pos = 0
        while i != 0:
            pos += len(mols_graphs[i-1].nodes)
            i -= 1
        for bond in [(a+pos, b+pos) for a, b in list(mols_graphs[i].edges) if a != b]:
            bonds.append(bond)
    bonds = set(bonds)

    new_bonds = {(a, b) for a, b in list(graphize(TS_structure, TS_atomnos).edges) if a != b}
    delta_bonds = (bonds | new_bonds) - (bonds & new_bonds)
    for c_bond in constrained_indexes:
        try:
            delta_bonds.remove(tuple(c_bond))
        except KeyError:
            pass
        try:
            delta_bonds.remove(tuple(reversed(tuple(c_bond))))
        except KeyError:
            pass

    if len(delta_bonds) > max_new_bonds:
        return False
    else:
        return True


def Hookean_optimization(TS_structure, TS_atomnos, constrained_indexes, mols_graphs, calculator, method='PM7', debug=False):
    '''
    Performs a geometry partial optimization (POPT) with Gaussian at $method level, 
    constraining the distance between the specified atom pair. Moreover, it includes a
    Hookean set of constraints to preserve molecular identities and prevent atom scrambling.

    :params TS_structure: list of coordinates for each atom in the TS
    :params TS_atomnos: list of atomic numbers for each atom in the TS
    :params constrained_indexes: indexes of constrained atoms in the TS geometry
    :params mols_graphs: list of molecule.graph objects, containing connectivity information
    :params mols_atomnos: list of molecule.atomnos lists, containing atomic number for all atoms
    :params method: Level of theory to be used in geometry optimization. Default if UFF.

    :return opt_struct: optimized structure
    :return energy: absolute energy
    '''
    assert len(TS_structure) == sum([len(graph.nodes) for graph in mols_graphs])

    atoms = Atoms(''.join([pt[i].symbol for i in TS_atomnos]), positions=TS_structure)

    # atoms.edit()
    constraints = []
    if len(constrained_indexes) == 1:
        constraints.append(FixBondLength(*constrained_indexes[0]))
    else:
        constraints.append(FixBondLengths(constrained_indexes))
        
    # print(f'Constrained indexes are {constrained_indexes}')

    rt_dict = {'CH':1.59} # Distance above which Hookean correction kicks in, in Angstroms
    k_dict  = {'CH':7}    # Spring force constant, in eV/Angstrom^2

    bonds = []
    for i, graph in enumerate(mols_graphs):

        pos = 0
        while i != 0:
            pos += len(mols_graphs[i-1].nodes)
            i -= 1

        for bond in [(a+pos, b+pos) for a, b in list(graph.edges) if a != b]:
            bonds.append(bond)
    # creating bond list containing all bonds present in the desired transition state

    i = 0
    for bond_a, bond_b in bonds:
        key = ''.join(sorted([pt[TS_atomnos[bond_a]].symbol, pt[TS_atomnos[bond_b]].symbol]))
        try:
            rt = rt_dict[key]
            k = k_dict[key]
            constraints.append(Hookean(a1=bond_a, a2=bond_b, rt=rt, k=k))
            i += 1
        except KeyError:
            pass
    # print(f'Hookean-protected {i} CH bonds')

    atoms.set_constraint(constraints)
    # print('Hookean Constraints are', [c.pairs for c in atoms.constraints if 'pairs' in vars(c)])

    jobname = 'temp'
    if calculator == 'Gaussian':
        atoms.calc = Gaussian(label=jobname, command=f'{GAUSSIAN_COMMAND} {jobname}.com {jobname}.log', method=method)
        opt = BFGS(atoms, trajectory=f'{jobname}.traj', logfile=f'{jobname}.traj_log')

    elif calculator == 'Mopac':
        atoms.calc = MOPAC(label=jobname, command=f'{MOPAC_COMMAND} {jobname}.mop', method=method)
        opt = BFGS(atoms, trajectory=f'{jobname}.traj', logfile=f'{jobname}.traj_log')

    t_start = time.time()

    try:
        with suppress_stdout_stderr():
            opt.run(fmax=0.05)
    except Exception as e:
        print(e)

    if debug:
        t_end = time.time()
        print('Hookean Constraints are', [c.pairs for c in atoms.constraints if 'pairs' in vars(c)])
        view(atoms)

    try:
        with suppress_stdout_stderr():
            energy = atoms.get_total_energy()
    except:
        energy = np.inf

    return atoms.positions, energy


def optimize(TS_structure, TS_atomnos, constrained_indexes, mols_graphs, calculator='Mopac', method='PM7', debug=False):
    '''
    Performs a geometry partial optimization (POPT) with Gaussian or Mopac at $method level, 
    constraining the distance between the specified atom pair. Moreover, performs a check of atomic
    pairs distances to ensure to have preserved molecular identities and prevented atom scrambling.

    :params TS_structure: list of coordinates for each atom in the TS
    :params TS_atomnos: list of atomic numbers for each atom in the TS
    :params constrained_indexes: indexes of constrained atoms in the TS geometry
    :params mols_graphs: list of molecule.graph objects, containing connectivity information
    :params mols_atomnos: list of molecule.atomnos lists, containing atomic number for all atoms
    :params method: Level of theory to be used in geometry optimization. Default if UFF.

    :return opt_struct: optimized structure
    :return energy: absolute energy of structure, in kcal/mol
    :return scrambled: bool, indicating if the optimization shifted up some bonds (except the constrained ones)
    '''
    assert len(TS_structure) == sum([len(graph.nodes) for graph in mols_graphs])

    atoms = Atoms(''.join([pt[i].symbol for i in TS_atomnos]), positions=TS_structure)

    if len(constrained_indexes) == 1:
        atoms.set_constraint(FixBondLength(*constrained_indexes[0]))
    else:
        atoms.set_constraint(FixBondLengths(constrained_indexes))
    # print('opt Constraints are', [c.pairs for c in atoms.constraints if hasattr(c,'pairs')])

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

    jobname = 'temp'
    if calculator == 'Gaussian':
        # atoms.calc = Gaussian(label=jobname, command=f'{GAUSSIAN_COMMAND} {jobname}.com {jobname}.log', method=method)
        calc_opt = Gaussian(label=jobname,
                            command=f'{GAUSSIAN_COMMAND} {jobname}.com {jobname}.log',
                            method=method,
                            addsec='B %s %s F' % (constrained_indexes[0]+1, constrained_indexes[1]+1))
                            # Gaussian atom indexing starts at 1 and not at 0 like VMD/Python
        opt = GaussianOptimizer(atoms, calc_opt)

        t_start = time.time()

        try:
            with suppress_stdout_stderr():
                opt.run(steps=100, opt='modredundant')
        except Exception as e:
            print(e)

    elif calculator == 'Mopac':
        atoms.calc = MOPAC(label=jobname, command=f'{MOPAC_COMMAND} {jobname}.mop', method=method)
        opt = BFGS(atoms, trajectory=f'{jobname}.traj', logfile=f'{jobname}.traj_log')

        t_start = time.time()

        try:
            with suppress_stdout_stderr():
                opt.run(fmax=0.05)
        except Exception as e:
            print(e)

    else:
        raise Exception('Calculator not recognized')

    t_end = time.time()

    new_bonds = {(a, b) for a, b in list(graphize(atoms.positions, TS_atomnos).edges) if a != b}
    delta_bonds = (bonds | new_bonds) - (bonds & new_bonds)
    # print('delta_bonds is', list(delta_bonds))

    delta_bonds -= set(((a,b) for a,b in constrained_indexes))
    delta_bonds -= set(((b,a) for a,b in constrained_indexes))

    if len(delta_bonds) > 0:
        not_scrambled = False
    else:
        not_scrambled = True

    # with open('log.txt', 'a') as f:
    #     if not_scrambled:
    #         f.write(f'Structure looks good')
    #     else:
    #         f.write(f'rejected - delta bonds are {len(delta_bonds)}: {delta_bonds}')
    #     f.write(f' - constrained ids were {constrained_indexes}\n')


    if debug:
        print(f'{calculator} {method} opt time', round(t_end-t_start, 3), 's')
        if not not_scrambled:
            print('Some scrambling occurred: bonds out of place are', delta_bonds)
        view(atoms)

    try:
        with suppress_stdout_stderr():
            energy = atoms.get_total_energy()
    except:
        energy = np.inf

    return atoms.positions, energy, not_scrambled

# from numpy_lru_cache_decorator import np_cache
# @np_cache(maxsize=10000)
# @lru_cache(maxsize=10000)
def rmsd_cache(tup):
    tgt, ref, atomnos1, atomnos2 = tup
    return rmsd(tgt, ref, atomnos1, atomnos2, center=True, minimize=True)

def scramble(array, sequence):
    return np.array([array[s] for s in sequence])

def prune_conformers(structures, atomnos, energies, max_rmsd=0.5, debug=False):
    '''
    Remove conformations that are too similar (have a small RMSD value).
    When removing structures, only the lowest energy one is kept.

    :params structures: numpy array of conformations
    :params energies: list of energies for each conformation
    :params max_rmsd: maximum rmsd value to consider two structures identical, in Angstroms
    '''
    rmsd_mat = np.zeros((len(structures), len(structures)))
    rmsd_mat[:] = np.nan
    for i, tgt in enumerate(structures):
        for j, ref in enumerate(structures[i+1:]):
            rmsd_mat[i, i+j+1] = rmsd_cache((tgt, ref, atomnos, atomnos))

    matches = np.where(rmsd_mat < max_rmsd)
    matches = [(i,j) for i,j in zip(matches[0], matches[1])]

    g = nx.Graph(matches)

    if debug:
        g.add_nodes_from(range(len(structures)))
        pos = nx.spring_layout(g)
        nx.draw(g, pos=pos, labels={i:i for i in range(len(g))})
        import matplotlib.pyplot as plt
        plt.show()

    subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
    groups = [tuple(graph.nodes) for graph in subgraphs]

    def en(tup):
        ens = [energies[t] for t in tup]
        return tup[ens.index(min(ens))]

    best_of_cluster = [en(group) for group in groups]
    rejects_sets = [set(a) - {b} for a, b in zip(groups, best_of_cluster)]

    rejects = []
    def flatten(seq):
        for s in seq:
            if type(s) in (tuple, list, set):
                flatten(s)
            else:
                rejects.append(s)

    flatten(rejects_sets)

    mask = np.array([True for _ in range(len(structures))], dtype=bool)
    for i in rejects:
        mask[i] = False

    return structures[mask], mask


def pre_prune_conformers(structures, atomnos, k=10, max_rmsd=1, debug=False):
    '''
    Initial removal of conformations that are too similar (have a small RMSD value)
    by splitting the structure set in k subsets and pruning conformations inside those.
    
    :params structures: numpy array of conformations
    :params max_rmsd: maximum rmsd value to consider two structures identical, in Angstroms
    '''
    r = np.arange(structures.shape[0])
    sequence = np.random.permutation(r)
    inv_sequence = np.array([np.where(sequence == i)[0][0] for i in r])


    structures = scramble(structures, sequence)
    # # scrambling array before splitting, so to improve efficiency

    mask_out = []
    d = len(structures) // k
    for step in range(k):
        if step == k-1:
            structures_subset = structures[d*step:]
        else:
            structures_subset = structures[d*step:d*(step+1)]


        rmsd_mat = np.zeros((len(structures_subset), len(structures_subset)))
        rmsd_mat[:] = np.nan
        for i, tgt in enumerate(structures_subset):
            for j, ref in enumerate(structures_subset[i+1:]):
                rmsd_mat[i, i+j+1] = rmsd_cache((tgt, ref, atomnos, atomnos))

        matches = np.where(rmsd_mat < max_rmsd)
        matches = [(i,j) for i,j in zip(matches[0], matches[1])]

        g = nx.Graph(matches)

        if debug:
            g.add_nodes_from(range(len(structures_subset)))
            pos = nx.spring_layout(g)
            nx.draw(g, pos=pos, labels={i:i for i in range(len(g))})
            import matplotlib.pyplot as plt
            plt.show()

        subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
        groups = [tuple(graph.nodes) for graph in subgraphs]

        best_of_cluster = [group[0] for group in groups]
        # the first on of each cluster is returned - does not matter what it is at this stage
        rejects_sets = [set(a) - {b} for a, b in zip(groups, best_of_cluster)]

        rejects = []
        def flatten(seq):
            for s in seq:
                if type(s) in (tuple, list, set):
                    flatten(s)
                else:
                    rejects.append(s)

        flatten(rejects_sets)

        mask = np.array([True for _ in range(len(structures_subset))], dtype=bool)
        for i in rejects:
            mask[i] = False

        mask_out.append(mask)
    
    mask = np.concatenate(mask_out)

    mask = scramble(mask, inv_sequence)
    structures = scramble(structures, inv_sequence)
    # # undoing the previous shuffling, therefore preserving the input order

    return structures[mask], mask

if __name__ == '__main__':

    from hypermolecule_class import Hypermolecule
    os.chdir(r'C:\Users\Nik\Desktop\Coding\TSCoDe\mopac_tests')

    mol1 = Hypermolecule('../Resources/indole/indole_ensemble.xyz', 6)
    mol2 = Hypermolecule('../Resources/SN2/ketone_ensemble.xyz', 2)

    TS = ccread('TS_out3.xyz')
    v = (TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph])

    if sanity_check(*v, debug=True):
        # Hookean_optimization(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Mopac', method='PM7', debug=True) # faster than Gaussian (8s)
        # Hookean_optimization(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Gaussian', method='PM6', debug=True) # f'in slow (58s)
        # optimize(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Mopac', method='PM6', debug=True) # works
        # optimize(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Gaussian', method='PM6', debug=True)
        # gaussian does not recognize the constraint...

        # checking time cost of Hook+regular compared to regular
        c, e = Hookean_optimization(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Mopac', method='PM7', debug=True)
        print(c.shape, TS.atomcoords[0].shape)
        # optimize(c, TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Mopac', method='PM7', debug=True)
        # optimize(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph], calculator='Mopac', method='PM7', debug=True)

    # for i in range(1,9):
    #     TS = ccread(f'TS_out{i}.xyz')
    #     sanity_check(TS.atomcoords[0], TS.atomnos, [6,2+len(mol1.atomnos)], [mol1.graph, mol2.graph])
