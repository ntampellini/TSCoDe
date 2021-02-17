'''
TSCoDe - Transition state Seeker from Conformational Density

(Work in Progress)

'''
from density_object_class import Density_object
import numpy as np
from copy import deepcopy
from parameters import *
from scipy import ndimage
from reactive_atoms_classes import pt
from pprint import pprint
from scipy.spatial.transform import Rotation as R
import os

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

def _compute_hypermolecule(self):
    '''
    '''
    self.hypermolecule_atomnos = []
    clusters = {i:{} for i,_ in enumerate(self.atomnos)}  # {atom_index:{cluster_number:[position,times_found]}}
    for i, atom_number in enumerate(self.atomnos):
        atoms_arrangement = [conformer[i] for conformer in self.atomcoords]
        cluster_number = 0
        clusters[i][cluster_number] = [atoms_arrangement[0], 1]
        self.hypermolecule_atomnos.append(atom_number)
        radii = pt[atom_number].covalent_radius
        for atom in atoms_arrangement[1:]:
            for cluster_number, reference in clusters[i].items():
                if np.linalg.norm(atom - reference[0]) < radii:
                    clusters[i][cluster_number][1] += 1
            else:
                clusters[i][max(clusters[i].keys())+1] = [atom, 1]
                self.hypermolecule_atomnos.append(atom_number)

    # self.hypermolecule = [[] for _ in range(len(self.atomnos))]
    self.weights = [[] for _ in range(len(self.atomnos))]

    self.hypermolecule = []
    # self.weights = []

    for i in range(len(self.atomnos)):
        for _, data in clusters[i].items():
            # self.hypermolecule[i].append(data[0])
            self.weights[i].append(data[1])
            self.hypermolecule.append(data[0])
            # self.weights.append(data[1])

    
    self.hypermolecule = np.asarray(self.hypermolecule)
    self.weights = np.array(self.weights).flatten()
    self.weights = np.asarray([weights / np.sum(weights) for i, weights in enumerate(self.weights)])

    # pprint(self.hypermolecule)
    # pprint(self.weights)

    self.dimensions = (max([coord[0] for coord in self.hypermolecule]) - min([coord[0] for coord in self.hypermolecule]),
                       max([coord[1] for coord in self.hypermolecule]) - min([coord[1] for coord in self.hypermolecule]),
                       max([coord[2] for coord in self.hypermolecule]) - min([coord[2] for coord in self.hypermolecule]))



class Docker:
    def __init__(self, *objects):
        self.objects = list(*objects)
    
    def setup(self, population=1, maxcycles=100):
        self.population = population
        self.maxcycles = maxcycles

    def run(self, debug=False):
        '''
        '''
        if len(self.objects) != 2:
            raise Exception('Still to be implemented. Sorry.')

        self.objects[1].position = np.array([(self.objects[0].dimensions[0] + self.objects[1].dimensions[0]), 0, 0]) # in angstroms
        self.objects[1].rotation = R.from_rotvec(np.array([0, 0, 180]) * np.pi / 180).as_matrix() # flip along z to face the first molecule

        threads = [self.objects]     # first thread is vanilla, other are randomly modified
        for _ in range(self.population - 1):
            threads.append([deepcopy(self.objects[0]), deepcopy(self.objects[1])])
        
        for thread in threads[1:]:

            for molecule in thread[1:]:

                delta_pos = ((np.random.rand(3)*2. - 1.) * MAX_DIST) * np.pi / 180 * multiplier
                delta_rot = ((np.random.rand(3)*2. - 1.) * MAX_ANGLE) * np.pi / 180 * multiplier

                np.add(molecule.position, delta_pos, out=molecule.position, casting='unsafe')

                delta_rot_mat = R.from_rotvec(delta_rot).as_matrix()
                molecule.rotation = molecule.rotation @ delta_rot_mat

        results = {}
        best_score = 0

        try:
            os.remove('debug_out.xyz')
        except:
            pass

        for thread_number, thread in enumerate(threads):
            iteration = 0
            best_score = -1e100
            far = True
            biggest_clash_vector = np.array([0,0,0])
            while True:
                score = 0

                for molecule in thread[1:]:

                    min_dist = 1e5

                    for i, atom in enumerate(molecule.hypermolecule):
                        for j, ref_atom in enumerate(thread[0].hypermolecule):

                            a = molecule.rotation @ atom + molecule.position
                            b = ref_atom + thread[0].position

                            dist = np.linalg.norm(a - b)


                            if dist < D_CLASH:
                                probability = molecule.weights[i] * thread[0].weights[j]
                                score -= 2.5*(D_CLASH - dist) # D_CLASH is also the minimum distance to feel repulsive interaction
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

                    if min(dist) < (K_SOFTNESS/1.25):

                        far = False
                       
                        orb1 = orb_vecs[dist.index(min(dist))][0] # closest couple of orbital vectors (absolute positioning)
                        orb2 = orb_vecs[dist.index(min(dist))][1]

                        orb_vers1 = norm(orb1 - thread[0].reactive_atoms_classes[0].coord)   # orbital versors, relative positioning
                        orb_vers2 = norm(orb2 - molecule.reactive_atoms_classes[0].coord - molecule.position)

                        alignment = np.abs(orb_vers1 @ orb_vers2)
                        s = 1 - 1.25/K_SOFTNESS * min(dist)
                        score += s*alignment
                        # print(f'ORB: s {s}, a {alignment} - {round(s*alignment,2)} points')
     


                it = '%-4s' % (iteration)
                print(f'Iteration {it} of thread {thread_number + 1}/{self.population}: score {round(score, 3)}, best {round(best_score, 3)}')

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

                            else:
                                coords.append('%-4s %-12s %-12s %-12s' % ('Li', orb1[0], orb1[1], orb1[2]))
                                coords.append('%-4s %-12s %-12s %-12s' % ('Li', orb2[0], orb2[1], orb2[2]))

                    with open('debug_out.xyz', 'a') as f:
                        f.write(f'{len(coords)}\nTEST\n')
                        f.write('\n'.join(coords))
                        f.write('\n')                    

                if iteration == 1 or score > best_score:
                    unproductive_iterations = 0
                    results[thread_number] = [thread[0], thread[1]]
                    last_step = deepcopy(thread)
                    last_score = deepcopy(best_score)
                    best_score = deepcopy(score)
                else:
                    thread = deepcopy(last_step)
                    unproductive_iterations += 1

                if iteration == self.maxcycles or unproductive_iterations == self.maxcycles / 2 or score > 0.99:
                    break

                # if score > 1.01:
                #     raise Exception('Something wrong happened. Sorry.')

                for molecule in thread[1:]:

                    if best_score < 0.5:
                        multiplier = 1 - best_score if best_score > 0.1 else 1
                        delta_pos = ((np.random.rand(3)*2. - 1.) * MAX_DIST) * np.pi / 180 * multiplier
                        delta_rot = ((np.random.rand(3)*2. - 1.) * MAX_ANGLE) * np.pi / 180 * multiplier

                    else:
                        delta_pos = np.array([0.,0.,0.])
                        delta_rot = np.array([0.,0.,0.])
                        # delta_rot = ((np.random.rand(3)*2. - 1.) * MAX_ANGLE) * 0.05


                    if far: # if far, bring them closer
                        delta_pos += 0.5*(thread[0].reactive_atoms_classes[0].center[0] - molecule.reactive_atoms_classes[0].center[0]) # will break for n mols

                    else: # if close, move towards and align angle to closest orbital, move away from biggest clash
                       
                        # WILL I EVER FIX THIS ALIGNMENT ISSUE? GOD ONLY KNOWS
                        # rot_vec = R.align_vectors(np.array([-orb_vers1]), np.array([orb_vers2]))[0].as_rotvec()
                        # delta_rot += 0.5*rot_vec

                        delta_pos += 0.5*(orb1 - orb2)

                        if best_score < 0.8:
                            delta_pos += 0.1 * biggest_clash_vector


                    np.add(molecule.position, delta_pos, out=molecule.position, casting='unsafe')

                    delta_rot_mat = R.from_rotvec(delta_rot).as_matrix()
                    molecule.rotation = molecule.rotation @ delta_rot_mat

                iteration += 1




#                 # compute clashes
#                 # compute Orb superposition
#                 # score the arrangement
#                 # if the best of this thread, store it
#                 # if worse than before, decide if it is worth to keep it (Monte Carlo)
#                 # if no new structure was found for N cycles, break
#                 # modify it randomly, based on how far we are from ideality (gradient descent idea)


a = ['Resources/SN2/MeOH.mol', 5]
# a = ['Resources/dienamine/dienamine.xyz', 7]

b = ['Resources/SN2/CH3Br.mol', 1]

inp = [a,b]

objects = [Density_object(m[0], m[1]) for m in inp]

for m in objects:
    _compute_hypermolecule(m)

docker = Docker(objects) # initialize docker with molecule density objects
docker.setup(maxcycles=50) # set variables

docker.run(debug=True)

path = os.path.join(os.getcwd(), 'debug_out.vmd')
check_call(f'vmd -e {path}'.split(), stdout=DEVNULL, stderr=STDOUT)