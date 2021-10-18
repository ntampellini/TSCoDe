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
from math import sqrt

import numpy as np
from numba import njit


@njit
def norm(vec):
    '''
    Returns the normalized vector.
    Reasonably faster than Numpy version.
    Only for 3D vectors.
    '''
    return vec / sqrt((vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]))

@njit
def norm_of(vec):
    '''
    Returns the norm of the vector.
    Reasonably faster than Numpy version.
    Only for 3D vectors.
    '''
    return sqrt((vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]))

@njit
def kronecker_delta(i, j) -> int:
    if i == j:
        return 1
    return 0

@njit
def get_inertia_moments(coords, atomnos, masses):
    '''
    Returns the diagonal of the diagonalized inertia tensor, that is
    a shape (3,) array with the moments of inertia along the main axes.
    (I_x, I_y and largest I_z last)
    '''
    
    coords -= center_of_mass(coords, atomnos, masses)
    inertia_moment_matrix = np.array([[0.,0.,0.],
                                      [0.,0.,0.],
                                      [0.,0.,0.]])

    for i in range(3):
        for j in range(3):
            k = kronecker_delta(i,j)
            inertia_moment_matrix[i][j] = sum([masses[n]*((norm_of(coords[n])**2)*k - coords[n][i]*coords[n][j])
                                                  for n in range(len(atomnos))])

    inertia_moment_matrix = diagonalize(inertia_moment_matrix)

    return np.diag(inertia_moment_matrix)

@njit
def diagonalize(A):
    eigenvalues_of_A, eigenvectors_of_A = np.linalg.eig(A)
    B = eigenvectors_of_A[:,np.abs(eigenvalues_of_A).argsort()]   
    diagonal_matrix= np.dot(np.linalg.inv(B), np.dot(A, B))
    return diagonal_matrix

@njit
def center_of_mass(coords, atomnos, masses):
    '''
    Returns the center of mass for the atomic system.
    '''
    total_mass = sum([masses[i] for i in range(len(atomnos))])
    w = np.array([0.,0.,0.])
    for i in range(len(atomnos)):
        w += coords[i]*masses[i]
    return w / total_mass

@njit
def align_vec_pair(ref, tgt):
    '''
    ref, tgt: iterables of two 3D vectors each
    
    return: rotation matrix that when applied to tgt,
            optimally aligns it to ref
    '''
    
    B = np.zeros((3,3))
    for i in range(3):
        for k in range(3):
            tot = 0
            for j in range(2):
                tot += ref[j][i]*tgt[j][k]
            B[i,k] = tot

    u, s, vh = np.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    if np.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]

    return np.dot(u, vh)