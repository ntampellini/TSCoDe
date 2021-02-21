'''
TSCoDe - Transition state Seeker from Conformational Density

(Work in Progress)

'''
from density_object_class import Density_object
import numpy as np
from copy import deepcopy
from constants import *
from scipy import ndimage

def loadbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
	percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
	if iteration == total:
		print()

def _rotate_scalar_field(array, rotations):
    '''
    :params array:      an input scalar field with shape (a,b,c)
    :params rotations:  tuple of shape (1,3) with rotation 
                        angles along x, y, z axes, in degrees
    :return:            rotated array
    '''
    for axis, angle in enumerate(rotations):
        axes = [0,1,2]
        axes.remove(axis)
        array = ndimage.rotate(array, angle, axes=tuple(axes), reshape=False)
    return array

a = ['Resources/SN2/MeOH.mol', 5]
b = ['Resources/SN2/CH3Br.mol', 1]

inp = [a,b]

objects = [Density_object(m[0], m[1]) for m in inp]

for m in objects:
    m.compute_CoDe()
    m.compute_orbitals()

class Docker:
    def __init__(self, *objects):
        self.objects = list(*objects)
    
    def setup(self, population=5, maxcycles=100):
        self.population = population
        self.maxcycles = maxcycles

    def run(self):
        '''
        '''
        if len(self.objects) != 2:
            raise Exception('Still to be implemented. Sorry.')

        self.objects[1].position = np.array([round(self.objects[1].conf_dens.shape[0]*0.95), 0, 0]) # in voxels!
        self.objects[1].rotation = np.array([0, 0, 180]) # flip along z to face the first molecule

        threads = [self.objects]     # first thread is vanilla, other are randomly modified
        for _ in range(self.population - 1):
            threads.append([deepcopy(self.objects[0]), deepcopy(self.objects[1])])
        
        for first, second in threads[1:]:

            delta_pos = (np.random.rand(3)*2. - 1.) * MAX_DIST / VOXEL_DIM
            delta_rot = (np.random.rand(3)*2. - 1.) * MAX_ANGLE

            np.add(second.position, delta_pos, out=second.position, casting='unsafe')
            np.add(second.rotation, delta_rot, out=second.rotation, casting='unsafe')

        results = {}
        best_score = 0
        outline = 3 * MAX_DIST / VOXEL_DIM
        for thread_number, thread in enumerate(threads):
            iteration = 0
            while True:
                iteration += 1

                shape = (round(np.max(np.array([thread[0].conf_dens.shape[0], thread[1].conf_dens.shape[0], np.abs(thread[0].position[0]) + thread[0].conf_dens.shape[0]/2 + thread[1].conf_dens.shape[0]/2, np.abs(thread[1].position[0]) + thread[0].conf_dens.shape[0]/2 + thread[1].conf_dens.shape[0]/2])) + outline),
                         round(np.max(np.array([thread[0].conf_dens.shape[1], thread[1].conf_dens.shape[1], np.abs(thread[0].position[1]) + thread[0].conf_dens.shape[1]/2 + thread[1].conf_dens.shape[1]/2, np.abs(thread[1].position[1]) + thread[0].conf_dens.shape[1]/2 + thread[1].conf_dens.shape[1]/2])) + outline),
                         round(np.max(np.array([thread[0].conf_dens.shape[2], thread[1].conf_dens.shape[2], np.abs(thread[0].position[2]) + thread[0].conf_dens.shape[2]/2 + thread[1].conf_dens.shape[2]/2, np.abs(thread[1].position[2]) + thread[0].conf_dens.shape[2]/2 + thread[1].conf_dens.shape[2]/2])) + outline))
                
                conf_dens_map = np.zeros(shape)
                orb_dens_map = np.zeros(shape)

                for molecule in thread:

                    x = round(thread[0].conf_dens.shape[0]/2 + molecule.position[0] - molecule.conf_dens.shape[0]/2 + outline/2)
                    y = round(thread[0].conf_dens.shape[1]/2 + molecule.position[1] - molecule.conf_dens.shape[1]/2 + outline/2)
                    z = round(thread[0].conf_dens.shape[2]/2 + molecule.position[2] - molecule.conf_dens.shape[2]/2 + outline/2)
                    # print(f'Molecule position is {molecule.position}, shape is {molecule.conf_dens.shape}, box shape is {conf_dens_map.shape}')
                    # print(f'Broadcasting into {x,y,z}')
                    try:
                        conf_dens_map[x:x+molecule.conf_dens.shape[0], y:y+molecule.conf_dens.shape[1], z:z+molecule.conf_dens.shape[2]] *= _rotate_scalar_field(molecule.conf_dens, molecule.rotation)
                        orb_dens_map[x:x+molecule.orb_dens.shape[0], y:y+molecule.orb_dens.shape[1], z:z+molecule.orb_dens.shape[2]] *= _rotate_scalar_field(molecule.orb_dens, molecule.rotation)
                    except ValueError as e:
                        print(e)
                        quit()

                score = np.sum(orb_dens_map) - np.sum(conf_dens_map)

                print(f'Iteration {iteration} of thread {thread_number + 1}/{self.population}: score {score}, shape {shape}')

                if iteration == 1 or score > best_score:
                    unproductive_iterations = 0
                    results[thread_number] = [thread[0], thread[1]]
                    last_step = deepcopy(thread)
                    last_score = deepcopy(best_score)
                    best_score = deepcopy(score)
                # elif bestscore - score > np.random.rand(): # Monte Carlo criterion
                else:
                    thread = deepcopy(last_step)
                    unproductive_iterations += 1

                if unproductive_iterations == self.maxcycles:
                    break
               
                delta_pos = ((np.random.rand(3)*2. - 1.) * MAX_DIST / VOXEL_DIM) * 1
                delta_rot = ((np.random.rand(3)*2. - 1.) * MAX_ANGLE) * 1

                np.add(thread[1].position, delta_pos, out=thread[1].position, casting='unsafe')
                np.add(thread[1].rotation, delta_rot, out=thread[1].rotation, casting='unsafe')


                # compute CoDe superposition
                # compute Orb superposition
                # score the arrangement
                # if the best of this thread, store it
                # if worse than before, decide if it is worth to keep it (Monte Carlo)
                # if no new structure was found for N cycles, break
                # modify it randomly, based on how far we are from ideality (gradient descent idea)

docker = Docker(objects) # initialize docker with molecule density objects
docker.setup() # set variables
docker.run()