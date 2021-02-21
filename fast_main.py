'''
TSCoDe - Transition state Seeker from Conformational Density

(Work in Progress)

'''
from hypermolecule_class import Hypermolecule
import numpy as np
from copy import deepcopy
from parameters import *
from scipy import ndimage
from reactive_atoms_classes import pt
from pprint import pprint
from scipy.spatial.transform import Rotation as R
import os
import time

from subprocess import DEVNULL, STDOUT, check_call


def loadbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
	percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
	if iteration == total:
		print()

def norm(vec):
    return vec / np.linalg.norm(vec)

# def quaternion_multiply(quaternion1, quaternion0):
#     w0, x0, y0, z0 = quaternion0
#     w1, x1, y1, z1 = quaternion1
#     return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
#                      x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
#                      -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
#                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

# def align_vectors(reference, target):
#     '''
#     :params reference, target: single vectors, ndarray type, normalized

#     :return: rotation to be applied to target to align with reference
#              as a rotation vector, in radians.
#     '''
#     assert reference.shape[0] == 3
#     assert target.shape[0] == 3

#     p = 0, target[0], target[1], target[2]

#     axis = norm(np.cross(target, reference))
#     angle = np.arccos(target @ reference) / 2

#     sin = np.sin(angle)
#     # q = np.cos(angle), sin*axis[0], sin*axis[1], sin*axis[2]

#     # sin_inv = np.sin(-angle)
#     # q_inv = np.cos(-angle), sin_inv*axis[0], sin_inv*axis[1], sin_inv*axis[2]

#     # quaternion_multiply(quaternion_multiply(q, p), q_inv)[1:]

#     q = sin*axis[0], sin*axis[1], sin*axis[2], np.cos(angle)

#     print(f'ALIGN: rotating {2* angle * 180 / np.pi} degrees')

#     return -R.from_quat(q).as_rotvec()

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

class Docker:
    def __init__(self, *objects):
        self.objects = list(*objects)
    
    def setup(self, population=1, maxcycles=100):
        self.population = population
        self.maxcycles = maxcycles

    def run(self, debug=False):
        '''
        '''
        print()       

        t_start = time.time()

        if len(self.objects) != 2:
            raise Exception('Still to be implemented. Sorry.')


        threads = []
        for _ in range(self.population):
            threads.append([deepcopy(self.objects[0]), deepcopy(self.objects[1])])
        
        for thread in threads: # each run is set up similarly, with a random variation

            for molecule in thread[1:]: #first molecule is always frozen in place

                delta_pos = np.array([(self.objects[0].dimensions[0] + self.objects[1].dimensions[0]), 0, 0]) # in angstroms
                delta_rot = np.array([0, 0, np.pi])                                             # flip along z to face the first molecule
                delta_pos += ((np.random.rand(3)*2. - 1.) * MAX_DIST) * np.pi / 180
                delta_rot += ((np.random.rand(3)*2. - 1.) * MAX_ANGLE) * np.pi / 180

                np.add(molecule.position, delta_pos, out=molecule.position, casting='unsafe')

                delta_rot_mat = R.from_rotvec(delta_rot).as_matrix()
                molecule.rotation = molecule.rotation @ delta_rot_mat

        results = {i:{} for i in range(len(threads))}
        score_record = {i:0 for i in range(len(threads))}
        best_score = 0

        try:
            os.remove('debug_out.xyz')
        except:
            pass

        # pprint(threads)

        for thread_number, thread in enumerate(threads):
            iteration = 0
            unproductive_iterations = 0
            best_score = -1e100
            far = True

            while True:

                score = 0
                biggest_clash_vector = np.array([0,0,0])

                for molecule in thread[1:]:

                    min_dist = 1e5

                    for i, atom in enumerate(molecule.hypermolecule):
                        for j, ref_atom in enumerate(thread[0].hypermolecule):

                            a = molecule.rotation @ atom + molecule.position
                            b = ref_atom + thread[0].position

                            dist = np.linalg.norm(a - b)


                            if dist < D_CLASH:
                                probability = molecule.weights[i] * thread[0].weights[j]
                                score -= SLOPE*(D_CLASH - dist) # D_CLASH is also the minimum distance to feel repulsive interaction
                                if dist < min_dist:
                                    min_dist = dist
                                    biggest_clash_vector = a - b

                    dist = []
                    orb_vecs = []
                    for reactive_atom in molecule.reactive_atoms_classes:
                        for orbital in reactive_atom.center:
                            for ref_reactive_atom in thread[0].reactive_atoms_classes:
                                for ref_orbital in ref_reactive_atom.center:

                                    a = ref_orbital
                                    b = molecule.rotation @ orbital + molecule.position

                                    dist.append(np.linalg.norm(a - b))
                                    orb_vecs.append([a, b])

                    if min(dist) < (K_SOFTNESS/SLOPE):

                        far = False
                       
                        orb1 = orb_vecs[dist.index(min(dist))][0] # closest couple of orbital vectors (absolute positioning)
                        orb2 = orb_vecs[dist.index(min(dist))][1]

                        orb_vers1 = norm(orb1 - thread[0].reactive_atoms_classes[0].coord)   # orbital versors, relative positioning
                        orb_vers2 = norm(orb2 - molecule.reactive_atoms_classes[0].coord - molecule.position)

                        alignment = np.abs(orb_vers1 @ orb_vers2)
                        s = 1 - SLOPE/K_SOFTNESS * min(dist)
                        score += s*alignment
                        # print(f'ORB: s {s}, a {alignment} - {round(s*alignment,2)} points')
                    else:
                        far = True


                # it = '%-4s' % (iteration)
                # print(f'Iteration {it} of thread {thread_number + 1}/{self.population}: score {round(score, 3)}, best {round(best_score, 3)}, unprod. it. {unproductive_iterations}')

                loadbar(iteration + self.maxcycles*thread_number, self.maxcycles*self.population, f'Running generation {thread_number + 1}/{self.population}: ')

                if debug:
                    coords = []
                    for molecule in thread:
                        for i, atom in enumerate(molecule.hypermolecule):
                            adjusted = molecule.rotation @ atom + molecule.position
                            coords.append('%-4s %-12s %-12s %-12s' % (pt[molecule.hypermolecule_atomnos[i]].symbol, adjusted[0], adjusted[1], adjusted[2]))

                        for reactive_atom in molecule.reactive_atoms_classes:

                            for center in reactive_atom.center:
                                adjusted = molecule.rotation @ center + molecule.position
                                coords.append('%-4s %-12s %-12s %-12s' % ('D', adjusted[0], adjusted[1], adjusted[2]))

                            if far:
                                coords.append('%-4s %-12s %-12s %-12s' % ('Li', 5, 5, 5))
                                coords.append('%-4s %-12s %-12s %-12s' % ('Li', 5, 5, 5))

                                # coords.append('%-4s %-12s %-12s %-12s' % ('Be', 5, 5, 5))
                                # coords.append('%-4s %-12s %-12s %-12s' % ('Be', 5, 5, 5))
                                # coords.append('%-4s %-12s %-12s %-12s' % ('Be', 5, 5, 5))

                            else:
                                coords.append('%-4s %-12s %-12s %-12s' % ('Li', orb1[0], orb1[1], orb1[2]))
                                coords.append('%-4s %-12s %-12s %-12s' % ('Li', orb2[0], orb2[1], orb2[2]))

                                # coords.append('%-4s %-12s %-12s %-12s' % ('Be', -orb_vers1[0], -orb_vers1[1], -orb_vers1[2]))
                                # coords.append('%-4s %-12s %-12s %-12s' % ('Be', orb_vers2[0], orb_vers2[1], orb_vers2[2]))
                                # coords.append('%-4s %-12s %-12s %-12s' % ('Be', 0, 0, 0))

                    coords.append('%-4s %-12s %-12s %-12s' % ('O', 0, 0, 0))

                    with open('debug_out.xyz', 'a') as f:
                        f.write(f'{len(coords)}\nTEST\n')
                        f.write('\n'.join(coords))
                        f.write('\n')                    

                if iteration == 0 or score > best_score:

                    unproductive_iterations = 0

                    if score > 0:
                        for i, mol in enumerate(self.objects):
                            results[thread_number] = [deepcopy(mol) for mol in thread]
                            score_record[thread_number] = score

                    last_step = deepcopy(thread)
                    last_score = deepcopy(best_score)
                    best_score = deepcopy(score)
                    restart = False

                elif not restart:
                    thread = deepcopy(last_step)
                    unproductive_iterations += 1

                if iteration == self.maxcycles or score > 0.95:
                    break

                # if score > 1.01:
                #     raise Exception('Something wrong happened. Sorry.')

                for molecule in thread[1:]:

                    if best_score <= 0.5:
                        multiplier = min([1, 1 - best_score])
                        delta_pos = ((np.random.rand(3)*2. - 1.) * MAX_DIST) * multiplier
                        delta_rot_mat = R.from_rotvec(((np.random.rand(3)*2. - 1.) * MAX_ANGLE) * np.pi / 180 * multiplier).as_matrix()

                    else:
                        delta_pos = np.array([0.,0.,0.])
                        delta_rot_mat = np.identity(3)

                    if unproductive_iterations > 20:
                        delta_pos += (np.random.rand(3)*2. - 1.) * 20
                        restart = True
                        unproductive_iterations = 0

                    if far: # if far, bring them closer
                        delta_pos += 0.5*(thread[0].reactive_atoms_classes[0].center[0] - molecule.rotation @ molecule.reactive_atoms_classes[0].center[0] - molecule.position) # will break for n mols

                    # else: # if close, move towards and align angle to closest orbital, move away from biggest clash
                       
                    #     # WILL I EVER FIX THIS ALIGNMENT ISSUE? GOD ONLY KNOWS
                    #     if np.linalg.norm(orb1-orb2) < 0.5:
                    #         # rot_vec = R.align_vectors(np.array([-orb_vers1]), np.array([orb_vers2]))[0].as_rotvec()
                    #         # delta_rot_mat = R.from_rotvec(rot_vec).as_matrix() @ delta_rot_mat

                    #         alignment_mat = rotation_matrix_from_vectors(orb_vers2, -orb_vers1)
                    #         delta_rot_mat = alignment_mat @ delta_rot_mat

                    #         delta_pos += (orb2 - molecule.position) * np.sin(R.from_matrix(alignment_mat).as_rotvec())


                    #     delta_pos += 0.5*(orb1 - orb2)

                    #     # if best_score < 0.8:
                    #     #     delta_pos += 0.1 * biggest_clash_vector


                    np.add(molecule.position, delta_pos, out=molecule.position, casting='unsafe')

                    # delta_rot_mat = R.from_rotvec(delta_rot).as_matrix()
                    # molecule.rotation = molecule.rotation @ delta_rot_mat
                    molecule.rotation = delta_rot_mat @ molecule.rotation
                    # print('MOL ROT is', R.from_matrix(molecule.rotation).as_rotvec() * 180 / np.pi)

                iteration += 1

        t_stop = time.time()

        print(f'Took {round(t_stop - t_start, 2)} s, about {round((t_stop - t_start)/self.maxcycles/self.population*1000, 3)} ms per iteration ({self.maxcycles*self.population})')

        try:
            os.remove('raw_results_out.xyz')
        except:
            pass

        for run, thread in results.items():
            if score_record[run] > 0:
                coords = []
                for molecule in thread:
                    for i, atom in enumerate(molecule.hypermolecule):
                        adjusted = molecule.rotation @ atom + molecule.position
                        coords.append('%-4s %-12s %-12s %-12s' % (pt[molecule.hypermolecule_atomnos[i]].symbol, adjusted[0], adjusted[1], adjusted[2]))

                with open('raw_results_out.xyz', 'a') as f:
                    f.write(f'{len(coords)}\nGENERATION {run} - score {score_record[run]}\n')
                    f.write('\n'.join(coords))
                    f.write('\n')

        print()
        print('   Run   Score')
        print('-----------------')
        for run, score in score_record.items():
            score = 'DNF' if score == 0 else round(score, 3)
            print('   %-3s   %-4s' % (run, score))
        print('-----------------\n')

        # for run, thread in results.items():
        #     if score_record[run] > 0:
        #         for molecule in thread:
        #             for conformer in molecule.atomcoords():
        #                 coords = np.array([molecule.rotation @ atom + molecule.position for atom in conformer])
 

            



#                 # compute clashes
#                 # compute Orb superposition
#                 # score the arrangement
#                 # if the best of this thread, store it
#                 # if worse than before, decide if it is worth to keep it (Monte Carlo)
#                 # if no new structure was found for N cycles, break
#                 # modify it randomly, based on how far we are from ideality (gradient descent idea)


# a = ['Resources/SN2/MeOH_ensemble.xyz', 5]
# a = ['Resources/dienamine/dienamine_ensemble.xyz', 7]
# a = ['Resources/SN2/amine_ensemble.xyz', 22]
a = ['Resources/indole/indole_ensemble.xyz', 9]


# b = ['Resources/SN2/CH3Br_ensemble.xyz', 1]
b = ['Resources/SN2/ketone_ensemble.xyz', 3]


inp = [a,b]

objects = [Hypermolecule(m[0], m[1]) for m in inp]

docker = Docker(objects) # initialize docker with molecule density objects
docker.setup(population=2, maxcycles=50) # set variables

docker.run(debug=True)

path = os.path.join(os.getcwd(), 'debug_out.vmd')
# path = os.path.join(os.getcwd(), 'raw_results_out.vmd')
check_call(f'vmd -e {path}'.split(), stdout=DEVNULL, stderr=STDOUT)