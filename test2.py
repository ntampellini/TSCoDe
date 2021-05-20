from optimization_methods import get_nci
from cclib.io import ccread
import numpy as np
import os
os.chdir('Resources/tri')
data = ccread('TSCoDe_TSs_guesses.xyz')
ids = np.array([32, 5, 31])
constrained_indexes = np.array([[0, 43], [5, 36], [33, 60]])

get_nci(data.atomcoords[0], data.atomnos, constrained_indexes, ids)