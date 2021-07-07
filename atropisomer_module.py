from networkx.algorithms.components.connected import connected_components
import numpy as np
from ase.optimize import LBFGS
from ase.vibrations import Vibrations
from ase.constraints import FixInternals
from optimization_methods import get_ase_calc
from ase import Atoms
from utils import dihedral, suppress_stdout_stderr, clean_directory
from sella import Sella
from hypermolecule_class import graphize
from networkx import connected_components

def ase_torsion_TSs(coords, atomnos, indexes, calculator, method, threshold_kcal=5, title='temp', bernytraj=None, debug=False):
    '''
    '''
    
    assert len(indexes) == 4

    structures, energies = ase_scan(coords,
                                    atomnos,
                                    calculator=calculator,
                                    method=method,
                                    indexes=indexes,
                                    degrees=10,
                                    steps=36)

    if debug:
        import matplotlib.pyplot as plt
        import pickle

        fig = plt.figure()

        x1 = [dihedral(structures[i][indexes]) for i in range(36)]
        min_e = min(energies)

        y1 = [e-min_e for e in energies]

        x1, y1 = zip(*sorted(zip(x1, y1), key=lambda x: x[0]))

        plt.plot(x1,
                 y1,
                 color='tab:blue',
                 label='Main SCAN')

    peaks_indexes = peaks(energies)

    ts_structures, energies, frequencies = [], [], []

    for p, peak in enumerate(peaks_indexes):

        sub_structures, sub_energies = ase_scan(structures[peak-1],
                                                    atomnos,
                                                    calculator=calculator,
                                                    method=method,
                                                    indexes=indexes,
                                                    degrees=1,
                                                    steps=20)

        if debug:
            x2 = [dihedral(sub_structures[i][indexes]) for i in range(20)]
            y2 = [e-min_e for e in sub_energies]

            x2, y2 = zip(*sorted(zip(x2, y2), key=lambda x: x[0]))

            plt.plot(x2, 
                     y2,
                     color='tab:red',
                     label='Accurate SCAN')


        sub_peaks_indexes = peaks(sub_energies)

        for s, sub_peak in enumerate(sub_peaks_indexes):

            if sub_energies[sub_peak] - min_e > threshold_kcal:
            # do not search transition states starting from scan points with Rel.E < threshold_kcal

                candidate, energy, freqs = ase_berny(sub_structures[sub_peak],
                                                        atomnos,
                                                        calculator=calculator,
                                                        method=method,
                                                        traj=bernytraj+f'_{p}_{s}.traj')

                # if is_ts:
                ts_structures.append(candidate)
                energies.append(energy)
                frequencies.append(freqs)

    ts_structures = np.array(ts_structures)
    energies = np.array(energies)

    if debug:
        plt.legend()
        plt.xlabel(f'Dihedral Angle {tuple(indexes)}')
        plt.ylabel('Energy (kcal/mol)')
        pickle.dump(fig, open(f'{title}_plt.pickle', 'wb'))

    return ts_structures, energies, frequencies

def peaks(data):
    '''
    return: list of peak indexes
    '''
    l = len(data)
    peaks = [i for i in range(l) if data[i-1] < data[i] > data[(i+1)%l]]

    return peaks
    
def ase_berny(coords, atomnos, calculator, method, title='temp', traj=None):
    '''
    Runs a Berny optimization through the ASE package
    '''
    atoms = Atoms(atomnos, positions=coords)

    atoms.calc = get_ase_calc(calculator, method)
    
    with suppress_stdout_stderr():
        with Sella(atoms,
                logfile=None,
                order=1,
                trajectory=traj) as opt:

            opt.run(fmax=0.05, steps=200)

    new_structure = atoms.get_positions()

    vib = Vibrations(atoms, name=title)
    vib.run()
    freqs = vib.get_frequencies()

    # neg_freqs = [f for f in freqs if f < 0]

    # if len(neg_freqs) == 1 and abs(neg_freqs[0]) > 50:
    #     is_ts = True
    # else:
    #     is_ts = False

    return new_structure, atoms.get_total_energy(), freqs

def ase_scan(coords, atomnos, calculator, method, indexes, degrees, steps, relaxed=True):
    '''
    '''
    assert len(indexes) == 4

    atoms = Atoms(atomnos, positions=coords)
    structures, energies = [], []

    atoms.calc = get_ase_calc(calculator, method)

    graph = graphize(coords, atomnos)
    _, i2, i3, _ = indexes
    graph.remove_edge(i2, i3)
    subgraphs = connected_components(graph)

    for subgraph in subgraphs:
        if i3 in subgraph:
            indexes_to_be_moved = subgraph - {i3}
            break

    mask = np.array([i in indexes_to_be_moved for i in range(len(atomnos))], dtype=bool)

    for _ in range(steps):

        if relaxed:
            atoms.set_constraint(FixInternals(dihedrals_deg=[[atoms.get_dihedral(*indexes), indexes]]))
            
            with LBFGS(atoms, maxstep=0.2, logfile=None, trajectory=None) as opt:
                opt.run(fmax=0.05, steps=500)

            energies.append(atoms.get_total_energy())

        structures.append(atoms.get_positions())

        atoms.rotate_dihedral(*indexes, angle=degrees, mask=mask)

    structures = np.array(structures)
    energies = np.array(energies)

    clean_directory()

    return structures, energies
