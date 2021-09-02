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
# MAGIC - Machine-Automated Grading of Intricate Conformer ensembles (?)

import numpy as np
from utils import norm, time_to_string, write_xyz
from utils import ase_view
from ase_manipulations import ase_safe_relax

def score_ensemble(mol):
    '''
    return: array of floats, with conformer ratings (1 for worst, 0 for best)
    '''

    scores = np.zeros(len(mol.atomcoords), dtype=float)

    for c, coords in enumerate(mol.atomcoords):
        # indexes = set()
        for r_atom in mol.reactive_atoms_classes_dict.values():
            for center in r_atom.center:

                direction = norm(center - r_atom.coord)
                # direction versor for current orbital

                norms = np.linalg.norm(coords - center, axis=1)
                # distance of atoms from orbital center

                norms = np.where(((coords-center) @ direction) <
                                  np.linalg.norm(center - r_atom.coord), 0, norms)
                # ignore all vectors pointing in the direction opposite
                # to the one of the orbital under consideration (dot product < 0)
                # and the ones that are outside of an ideal "attack cone"
                # (dot product < norm of orb vec)

                # indexes |= {i for i, n in enumerate(norms) if 0 < n < 4}

                scores[c] += _cost_function(norms)

    scores /= np.max(scores)

    return scores

def _cost_function(distances:np.ndarray, close=1, far=4):
    '''
    Trimmed linear cost fuction

    p1 = (close, 1)
    p2 = (far, 0)

    m = dy/dx = 1/(close-far)
    y = mx+q -> q = y - mx -> q = 0 - 1/(close-far)*far

    y = mx+q = x/(c-f) - f/(c-f)
    '''

    distances = np.minimum(distances, close)
    distances = distances[distances < far]
    costs = distances/(close-far) - far/(close-far)

    return np.sum(costs)

def show_clashes(mol):

    indexes = set()

    # for c, coords in enumerate(mol.atomcoords):
    coords = mol.atomcoords[0]
    for r_atom in mol.reactive_atoms_classes_dict.values():
        for center in r_atom.center:

            direction = norm(center - r_atom.coord)
            # direction versor for current orbital

            norms = np.linalg.norm(coords - center, axis=1)
            # distance of atoms from orbital center

            norms = np.where(((coords-center) @ direction) <
                                np.linalg.norm(center - r_atom.coord), 0, norms)
            # ignore all vectors pointing in the direction opposite
            # to the one of the orbital under consideration (dot product < 0)
            # and the ones that are outside of an ideal "attack cone"
            # (dot product < norm of orb vec)

            indexes |= {i for i, n in enumerate(norms) if 0 < n < 4}

    mol.atomnos = np.array([n if i in indexes else 0 for i, n in enumerate(mol.atomnos)])
    ase_view(mol)

def bezier(p1, p2, p3, p4, n=50):
    space = np.linspace(0, 1, n)
    return np.array([p1*(1-t)**3 + p2*3*t*(1-t)**2 + p3*3*(1-t)*t**2 + p4*t**3 for t in space])        

def _get_path_points(start, p1, p2, end, gap=1):
    '''
    Returns a 1-d array with coordinates for points along
    the bezier curve that connects the two orbitals.
    '''

    bez = bezier(start, p1, p2, end)
    # compute (50) points in the bezier curve

    curve_length = np.sum(np.linalg.norm([bez[i+1]-bez[i] for i, _ in enumerate(bez[:-1])], axis=1))
    n_points = (curve_length // gap) + 2
    # get the number of points based on curve length

    points = bez[np.arange(bez.shape[0], dtype=int) % int(bez.shape[0]/n_points) == 0][1:-1]
    # set up the x axis linespace

    return points

def see_chain(docker, coords, atomnos, pivot, multiplier=3, title='temp', traj=None):

    p1 = ((pivot.start - pivot.start_atom.coord) * multiplier) + pivot.start_atom.coord
    p2 = ((pivot.end - pivot.end_atom.coord) * multiplier) + pivot.end_atom.coord

    band_coords = _get_path_points(pivot.start_atom.coord,
                                    p1,
                                    p2,
                                    pivot.end_atom.coord)

    with open(f'thing_{multiplier}.xyz', 'w') as f:
        temp_coords = np.concatenate((coords, band_coords, [p1], [p2]))
        temp_atomnos = np.concatenate((atomnos, [16 for _ in band_coords], [4, 4]))
        write_xyz(temp_coords, temp_atomnos, f)

    new_coords = np.concatenate((coords, band_coords))
    new_atomnos = np.concatenate((atomnos, [16 for _ in band_coords]))

    l = len(atomnos)
    constrained_indexes = np.concatenate((
                                    [[l+i, l+i+1] for i, _ in enumerate(band_coords[:-1])],
                                    [[pivot.start_atom.index, l]],
                                    [[pivot.end_atom.index, l+len(band_coords)]]
                                ))

    mask = [True if i < l else False for i, _ in enumerate(new_atomnos)]

    opt_coords, _, success = ase_safe_relax(docker,
                                            new_coords,
                                            new_atomnos,
                                            constrained_indexes,
                                            targets=[1.5 for _ in constrained_indexes],
                                            mask=mask,
                                            title=title,
                                            traj=traj,
                                            )

    with open(f'thing_{multiplier}_safely_refined.xyz', 'w') as f:
        write_xyz(opt_coords, new_atomnos, f)

    # return opt_coords[mask], 

if __name__ == '__main__':

    from hypermolecule_class import Hypermolecule
    from time import time
    from utils import time_to_string, ase_view

    t0 = time()
    # mol = Hypermolecule(r'C:\Users\Nik\Desktop\crest_conformers.xyz', reactive_atoms=(23, 61))
    mol = Hypermolecule(r'C:\Users\Nik\Desktop\crest_conformers_small.xyz', reactive_atoms=(23, 61))
    show_clashes(mol)
    quit()
    t1 = time()

    ase_view(mol)

    t2 = time()
    scores = score_ensemble(mol)
    t3 = time()
    print(scores)
    print(f'Mean {np.mean(scores)}, std {np.std(scores)}')

    print(f'Took {time_to_string(t1-t0)} to build mol, {time_to_string(t3-t2)} to score it.')

    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(scores, bins=50)
    plt.show()

