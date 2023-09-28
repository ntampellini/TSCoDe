import os
from itertools import permutations
import time

import numpy as np

from tscode.errors import InputError, ZeroCandidatesError
from tscode.utils import cartesian_product, time_to_string
from shutil import copy
from tscode.reactive_atoms_classes import Single


def multiembed_dispatcher(embedder):
    '''
    Calls the appropriate multiembed subfunction
    based on embedder attributes.
    '''
    if len(embedder.objects) == 2:
        return multiembed_bifunctional(embedder)
    
    raise InputError('The multiembed requested is currently unavailable.')


def multiembed_bifunctional(embedder):
    '''
    Run multiple concurrent bifunctional cyclical embeds
    exploring all relative arrangement of each pair of
    reactive_indices between the two molecules.
    '''

    from tscode.embedder import Embedder
    from tscode.run import RunEmbedding

    mol1, mol2 = embedder.objects

    # get every possible combination of indices in the two molecules
    pairs = cartesian_product(mol1.reactive_indices, mol2.reactive_indices)

    # get every arrangement of interacting pairs that does not insist on the same atom twice
    arrangements = [((ix_1, ix_2), (iy_1, iy_2)) for ((ix_1, ix_2), (iy_1, iy_2)) in permutations(pairs, 2) if ix_1 != iy_1 and ix_2 != iy_2]

    start_dir = os.getcwd()
    structures_out = []

    embedder.t_start_run = time.perf_counter()
    embedder.log()
    constr_ids = []

    # for each arrangement, perform a dedicated embed 
    for i, ((ix_1, ix_2), (iy_1, iy_2)) in enumerate(arrangements):
        
        foldername = f'TSCoDe_embed{i+1}'

        # create a dedicated folder
        if not os.path.isdir(os.path.join(os.getcwd(), foldername)):
            os.mkdir(foldername)

        # copy structure files into it
        copy(os.path.join(os.getcwd(), mol1.name),
                 os.path.join(os.getcwd(), foldername))
        copy(os.path.join(os.getcwd(), mol2.name),
                 os.path.join(os.getcwd(), foldername))

        os.chdir(foldername)
        child_name = f'embed{i+1}_input.txt'

        with open(child_name, 'w') as f:
            f.write('noopt rigid\n')
            f.write(f'{mol1.name} {ix_1}x {iy_1}y\n')
            f.write(f'{mol2.name} {ix_2}x {iy_2}y\n')

        child_embedder = RunEmbedding(Embedder(os.path.join(os.getcwd(), child_name), f'embed{i+1}'))

        for mol in child_embedder.objects:
        #     mol.compute_orbitals(manual=Single)
        #     child_embedder._set_pivots(mol)
            mol.write_hypermolecule()

        child_embedder._set_reactive_atoms_cumnums()
        child_embedder.write_mol_info()
        child_embedder.log(f'\n--> TSCoDe multiembed child process - arrangement {i+1} out of {len(arrangements)}')
        child_embedder.t_start_run = time.perf_counter()

        try:
            child_embedder.generate_candidates()
            child_embedder.compenetration_refining()
            child_embedder.similarity_refining(verbose=True)
            child_embedder.write_structures('unoptimized', energies=False)
            constr_ids.append(child_embedder.constrained_indices)

        except ZeroCandidatesError:
            child_embedder.structures = []

        child_embedder.log(f'\n--> Child process terminated ({time_to_string(time.perf_counter() - child_embedder.t_start_run, verbose=True)})')

        embedder.log(f'--> Child process {i+1}/{len(arrangements)}: generated {len(child_embedder.structures)} candidates in {time_to_string(time.perf_counter() - child_embedder.t_start_run, verbose=True)}.')

        if len(child_embedder.structures) > 0:
            structures_out.append(child_embedder.structures)

        os.chdir(start_dir)

    structures_out = np.concatenate(structures_out)

    embedder.log(f'\n--> Multiembed completed: generated {len(structures_out)} candidates in {time_to_string(time.perf_counter() - embedder.t_start_run, verbose=True)}.')
    
    # only get interaction constraints, as the internal will be added later during refinement
    embedder.constrained_indices = np.concatenate(constr_ids)
    # embedder.constrained_indices = np.array([np.concatenate((embedder.internal_constraints, child_constraints)) for child_constraints in np.concatenate(constr_ids)])

    return structures_out