from hypermolecule_class import Hypermolecule
import numpy as np

cutoff = 5
# cutoff value for clashes, in A

def score_func(d):
    '''
    Scoring function

    '''
    if d > cutoff:
        return 0
    else:
        return cutoff-d




# mol = Hypermolecule('Resources/SN2/MeOH_ensemble.xyz', 1) # only one reactive atom at time here

for oh_index in [11, 21, 23]:

    mol = Hypermolecule('Resources/bulk/tax_ensemble.xyz', oh_index)
    orbital_scores = []

    for orbital in mol.centers:
        
        index = mol.hypermolecule == mol.reactive_atoms_classes[0].coord
        index = np.array([i[0] for i in index])
        # index of reactive atom in hypermolecule and weights arrays is True, other are False

        hyp_coords = np.array(mol.hypermolecule)[~index]
        hyp_weights = np.array(mol.weights)[~index]

        dists = np.linalg.norm(hyp_coords - orbital, axis=1) # better to use covalent radii?
        dists = dists[np.argwhere(dists < cutoff)].flatten() # removing atoms further than cutoff value
        # print(f'orb {orbital}, d = {dists}')

        score = np.sum([score_func(d)*hyp_weights[i] for i, d in enumerate(dists)])
        # convert distances into a score (ReLU)

        orbital_scores.append(score)

    print(f'Spherical bulk of atom {oh_index} is {max(orbital_scores)}')