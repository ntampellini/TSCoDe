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

from utils import *
from copy import deepcopy
from optimization_methods import ase_bend

def string_embed(self):
    '''
    return threads: return embedded structures, with position and rotation attributes set, ready to be pumped
    into self.structures. Algorithm used is the "string" algorithm (see docs).
    '''
    assert len(self.objects) == 2

    self.log(f'\n--> Performing string embed ({self.candidates} candidates)')

    for mol in self.objects:
        mol.centers = list(mol.reactive_atoms_classes_dict.values())[0].center
        mol.orb_vers = np.array([norm(v) for v in list(mol.reactive_atoms_classes_dict.values())[0].orb_vecs])

    centers_indexes = cartesian_product(*[np.array(range(len(mol.centers))) for mol in self.objects])
    # for two mols with 3 and 2 centers: [[0 0][0 1][1 0][1 1][2 0][2 1]]
    
    threads = []
    for _ in range(len(centers_indexes)*self.options.rotation_steps):
        threads.append([deepcopy(obj) for obj in self.objects])

    for t, thread in enumerate(threads): # each run is a different "regioisomer", repeated self.options.rotation_steps times

        indexes = centers_indexes[t % len(centers_indexes)] # looping over the indexes that define "regiochemistry"
        repeated = True if t // len(centers_indexes) > 0 else False

        for i, molecule in enumerate(thread[1:]): #first molecule is always frozen in place, other(s) are placed with an orbital criterion

            ref_orb_vec = thread[i].centers[indexes[i]]  # absolute, arbitrarily long
            # reference molecule is the one just before molecule, so the i-th for how the loop is set up
            mol_orb_vec = molecule.centers[indexes[i+1]]

            ref_orb_vers = thread[i].orb_vers[indexes[i]]  # unit length, relative to orbital orientation
            # reference molecule is the one just before
            mol_orb_vers = molecule.orb_vers[indexes[i+1]]

            molecule.rotation = rotation_matrix_from_vectors(mol_orb_vers, -ref_orb_vers)

            molecule.position = thread[i].rotation @ ref_orb_vec + thread[i].position - molecule.rotation @ mol_orb_vec

            if repeated:

                pointer = molecule.rotation @ mol_orb_vers
                # rotation = (np.random.rand()*2. - 1.) * np.pi    # random angle (radians) between -pi and pi
                rotation = t % self.options.rotation_steps * 360/self.options.rotation_steps       # sistematic incremental step angle
                
                delta_rot = rot_mat_from_pointer(pointer, rotation)
                molecule.rotation = delta_rot @ molecule.rotation

                molecule.position = thread[i].rotation @ ref_orb_vec + thread[i].position - molecule.rotation @ mol_orb_vec

    self.dispositions_constrained_indexes = self.get_string_constrained_indexes(len(threads))

    return threads

def cyclical_embed(self):
    '''
    return threads: return embedded structures, with position and rotation attributes set, ready to be pumped
                    into self.structures. Algorithm used is the "cyclical" algorithm (see docs).
    '''
    
    def _get_directions(norms):
        '''
        Returns two or three vectors specifying the direction in which each molecule should be aligned
        in the cyclical TS, pointing towards the center of the polygon.
        '''
        assert len(norms) in (2,3)
        if len(norms) == 2:
            return np.array([[0,1,0],
                                [0,-1,0]])
        else:

            vertexes = np.zeros((3,2))

            vertexes[1] = np.array([norms[0],0])

            a = np.power(norms[0], 2)
            b = np.power(norms[1], 2)
            c = np.power(norms[2], 2)
            x = (a-b+c)/(2*a**0.5)
            y = (c-x**2)**0.5

            vertexes[2] = np.array([x,y])
            # similar to the code from polygonize, to get the active triangle
            # but without the orientation specified in the polygonize function
            
            a = vertexes[1,0] # first point, x
            b = vertexes[2,0] # second point, x
            c = vertexes[2,1] # second point, y

            x = a/2
            y = (b**2 + c**2 - a*b)/(2*c)
            cc = np.array([x,y])
            # 2D coordinates of the triangle circocenter

            v0, v1, v2 = vertexes

            meanpoint1 = np.mean((v0,v1), axis=0)
            meanpoint2 = np.mean((v1,v2), axis=0)
            meanpoint3 = np.mean((v2,v0), axis=0)

            dir1 = cc - meanpoint1
            dir2 = cc - meanpoint2
            dir3 = cc - meanpoint3
            # 2D direction versors connecting center of side with circumcenter.
            # Now we need to understand if we want these or their negative

            if np.any([np.all(d == 0) for d in (dir1, dir2, dir3)]):
            # We have a right triangle. To aviod numerical
            # errors, a small perturbation is made.
            # This should not happen, but just in case...
                norms[0] += 1e-5
                dir1, dir2, dir3 = [t[:-1] for t in _get_directions(norms)]

            angle0_obtuse = (vec_angle(v1-v0, v2-v0) > 90)
            angle1_obtuse = (vec_angle(v0-v1, v2-v1) > 90)
            angle2_obtuse = (vec_angle(v0-v2, v1-v2) > 90)

            dir1 = -dir1 if angle2_obtuse else dir1
            dir2 = -dir2 if angle0_obtuse else dir2
            dir3 = -dir3 if angle1_obtuse else dir3
            # invert the versors sign if circumcenter is
            # one angle is obtuse, because then
            # circumcenter is outside the triangle
            
            dir1 = norm(np.concatenate((dir1, [0])))
            dir2 = norm(np.concatenate((dir2, [0])))
            dir3 = norm(np.concatenate((dir3, [0])))

            return np.vstack((dir1, dir2, dir3))

    def _adjust_directions(self, directions, constrained_indexes, triangle_vectors, v, pivots):
        '''
        For trimolecular TSs, correct molecules pre-alignment. That is, after the initial estimate
        based on pivot triangle circocentrum, systematically rotate each molecule around its pivot
        by fixed increments and look for the arrangement with the smallest deviation from orbital
        parallel interaction. This optimizes the obtainment of poses with the correct inter-reactive
        atoms distances.

        '''
        assert directions.shape[0] == 3

        mols = deepcopy(self.objects)
        p0, p1, p2 = [end - start for start, end in triangle_vectors]
        p0_mean, p1_mean, p2_mean = [np.mean((end, start), axis=0) for start, end in triangle_vectors]

        ############### get triangle vertexes

        vertexes = np.zeros((3,2))
        vertexes[1] = np.array([norms[0],0])

        a = np.power(norms[0], 2)
        b = np.power(norms[1], 2)
        c = np.power(norms[2], 2)
        x = (a-b+c)/(2*a**0.5)
        y = (c-x**2)**0.5

        vertexes[2] = np.array([x,y])
        # similar to the code from polygonize, to get the active triangle
        # but without the orientation specified in the polygonize function
        
        a = vertexes[1,0] # first point, x
        b = vertexes[2,0] # second point, x
        c = vertexes[2,1] # second point, y

        x = a/2
        y = (b**2 + c**2 - a*b)/(2*c)
        cc = np.array([x,y])
        # 2D coordinates of the triangle circocenter

        v0, v1, v2 = vertexes

        v0 = np.concatenate((v0, [0]))
        v1 = np.concatenate((v1, [0]))
        v2 = np.concatenate((v2, [0]))

        ############### set up mols -> pos + rot

        alignment_rotation = np.zeros((3,3,3))
        for i in (0,1,2):

            start, end = triangle_vectors[i]

            mol_direction = pivots[i].meanpoint - np.mean(self.objects[i].atomcoords[0][self.objects[i].reactive_indexes], axis=0)
            if np.all(mol_direction == 0.):
                mol_direction = pivots[i].meanpoint

            mols[i].rotation = R.align_vectors((end-start, directions[i]), (pivots[i].pivot, mol_direction))[0].as_matrix()
            mols[i].position = np.mean(triangle_vectors[i], axis=0) - mols[i].rotation @ pivots[i].meanpoint

        ############### set up pairings between reactive atoms

        pairings = [[None, None] for _ in constrained_indexes]
        for i, c in enumerate(constrained_indexes):
            for m, mol in enumerate(self.objects):
                for index, r_atom in mol.reactive_atoms_classes_dict.items():
                    if r_atom.cumnum == c[0]:
                        pairings[i][0] = (m, index)
                    if r_atom.cumnum == c[1]:
                        pairings[i][1] = (m, index)

        r = np.zeros((3,3), dtype=int)

        for first, second in pairings:
            mol_index = first[0]
            partner_index = second[0]
            reactive_index = first[1]
            r[mol_index, partner_index] = reactive_index

            mol_index = second[0]
            partner_index = first[0]
            reactive_index = second[1]
            r[mol_index, partner_index] = reactive_index

        # r[0,1] is the reactive_index of molecule 0 that faces molecule 1 and so on
        # diagonal of r (r[0,0], r[1,1], r[2,2]) is just unused

        ############### calculate reactive atoms positions

        mol0, mol1, mol2 = mols

        a01 = mol0.rotation @ mol0.atomcoords[0][r[0,1]] + mol0.position
        a02 = mol0.rotation @ mol0.atomcoords[0][r[0,2]] + mol0.position

        a10 = mol1.rotation @ mol1.atomcoords[0][r[1,0]] + mol1.position
        a12 = mol1.rotation @ mol1.atomcoords[0][r[1,2]] + mol1.position

        a20 = mol2.rotation @ mol2.atomcoords[0][r[2,0]] + mol2.position
        a21 = mol2.rotation @ mol2.atomcoords[0][r[2,1]] + mol2.position

        ############### explore all angles combinations

        steps = 6
        angle_range = 30
        step_angle = 2*angle_range/steps
        angles_list = cartesian_product(*[range(steps+1) for _ in range(3)]) * step_angle - angle_range
        # Molecules are rotated around the +angle_range/-angle_range range in the given number of steps.
        # Therefore, the angular resolution between candidates is step_angle (10 degrees)

        candidates = []
        for angles in angles_list:

            rot0 = rot_mat_from_pointer(p0, angles[0])
            new_a01 = rot0 @ a01
            new_a02 = rot0 @ a02
            d0 = p0_mean - np.mean((new_a01, new_a02), axis=0)

            rot1 = rot_mat_from_pointer(p1, angles[1])
            new_a10 = rot1 @ a10
            new_a12 = rot1 @ a12
            d1 = p1_mean - np.mean((new_a10, new_a12), axis=0)

            rot2 = rot_mat_from_pointer(p2, angles[2])
            new_a20 = rot2 @ a20
            new_a21 = rot2 @ a21
            d2 = p2_mean - np.mean((new_a20, new_a21), axis=0)

            cost = 0
            cost += vec_angle(v0 - new_a02, new_a20 - v0)
            cost += vec_angle(v1 - new_a01, new_a10 - v1)
            cost += vec_angle(v2 - new_a21, new_a12 - v2)
                    
            candidates.append((cost, angles, (d0, d1, d2)))

        ############### choose the one with the best alignment, that is minor cost

        cost, angles, directions = sorted(candidates, key=lambda x: x[0])[0]
        
        return np.array(directions)

            
    self.log(f'\n--> Performing {self.embed} embed ({self.candidates} candidates)')

    steps = self.options.rotation_steps
    angle_range = self.options.rotation_range
    systematic_angles = cartesian_product(*[range(steps+1) for _ in self.objects]) * 2*angle_range/steps - angle_range

    pivots_indexes = cartesian_product(*[range(len(mol.pivots)) for mol in self.objects])
    # indexes of pivots in each molecule self.pivots list. For three mols with 2 pivots each: [[0,0,0], [0,0,1], [0,1,0], ...]
    
    threads = []
    constrained_indexes = []
    for p, pi in enumerate(pivots_indexes):

        loadbar(p, len(pivots_indexes), prefix=f'Embedding structures ')
        
        thread_objects = deepcopy(self.objects)
        # Objects to be used to embed structures. Modified later if necessary.

        pivots = [thread_objects[m].pivots[pi[m]] for m in range(len(self.objects))]
        # getting the active pivot for each molecule for this run
        
        norms = np.linalg.norm(np.array([p.pivot for p in pivots]), axis=1)
        # getting the pivots norms to feed into the polygonize function

        try:
            polygon_vectors = polygonize(norms)

        except TriangleError:
            # Raised if we cannot build a triangle with the given norms.
            # Try to bend the structure if it was close or just skip this triangle and go on.

            deltas = [norms[i] - (norms[i-1] + norms[i-2]) for i in range(3)]

            rel_delta = max([deltas[i]/norms[i] for i in range(3)])
            # s = 'Rejected triangle, delta was %s, %s of side length' % (round(delta, 3), str(round(100*rel_delta, 3)) + ' %')
            # self.log(s, p=False)

            if rel_delta < 0.2 and not self.options.rigid:
            # correct the molecule structure with the longest
            # side if the distances are at most 20% off.

                index = deltas.index(max(deltas))
                mol = thread_objects[index]

                if not tuple(sorted(mol.reactive_indexes)) in list(mol.graph.edges):
                    # do not try to bend molecules where the two reactive indices are bonded

                    pivot = pivots[index]

                    # ase_view(mol)
                    maxval = norms[index-1] + norms[index-2]
                    bent_mol = ase_bend(self, mol, pivot, 0.9*maxval)
                    # ase_view(bent_mol)

                    ###################################### DEBUGGING PURPOSES

                    # for p in bent_mol.pivots:
                    #     if p.index == pivot.index:
                    #         new_pivot = p
                    # now = np.linalg.norm(new_pivot.pivot)
                    # self.log(f'Corrected Triangle: Side was {round(norms[index], 3)} A, now it is {round(now, 3)}, {round(now/maxval, 3)} % of maximum value for triangle creation', p=False)
                    
                    #########################################################

                    thread_objects[index] = bent_mol

                    pivots = [thread_objects[m].pivots[pi[m]] for m in range(len(self.objects))]
                    # updating the active pivot for each molecule for this run
                    
                    norms = np.linalg.norm(np.array([p.pivot for p in pivots]), axis=1)
                    # updating the pivots norms to feed into the polygonize function

                    polygon_vectors = polygonize(norms)
                    # repeating the failed polygon creation

                else:
                    continue

            else:
                continue
        
        except TooDifferentLengthsError:

            if not self.options.rigid:

                if self.embed == 'chelotropic':
                    target_length = min(norms)

                else:
                    maxgap = 3 # in Angstrom
                    gap = abs(norms[0]-norms[1])
                    r = 0.5 + 0.5*(gap/maxgap)
                    r = np.clip(5, 0.5, 1)
                    # r is the ratio for the target_length based on the gap
                    # that deformations will need to cover.
                    # It ranges from 0.5 to 1 and if shifted more toward
                    # the shorter norm as the gap rises. For gaps of more
                    # than 3 Angstroms, basically only the molecule
                    # with the longest pivot is bent.

                    target_length = min(norms)*r + max(norms)*(1-r)
            
                for i, mol in enumerate(deepcopy(thread_objects)):

                    if len(mol.reactive_indexes) > 1:
                    # do not try to bend molecules that react with a single atom

                        if not tuple(sorted(mol.reactive_indexes)) in list(mol.graph.edges):
                        # do not try to bend molecules where the two reactive indices are bonded

                            bent_mol = ase_bend(self, mol, pivots[i], target_length, method=f'{self.options.mopac_level}')
                            # ase_view(bent_mol)
                            thread_objects[i] = bent_mol

                # Repeating the previous polygonization steps with the bent molecules

                pivots = [thread_objects[m].pivots[pi[m]] for m in range(len(self.objects))]
                # updating the active pivot for each molecule for this run
                
                norms = np.linalg.norm(np.array([p.pivot for p in pivots]), axis=1)
                # updating the pivots norms to feed into the polygonize function

                
                polygon_vectors = polygonize(norms, override=True)
                # repeating the failed polygon creation



        directions = _get_directions(norms)
        # directions to orient the molecules toward, orthogonal to each vec_pair

        for v, vecs in enumerate(polygon_vectors):
        # getting vertexes to embed molecules with and iterating over start/end points

            ids = self.get_cyclical_reactive_indexes(v)
            # get indexes of atoms that face each other

            if self.pairings == [] or all([pair in ids for pair in self.pairings]):
            # ensure that the active arrangement has all the pairings that the user specified

                if len(self.objects) == 3:

                    directions = _adjust_directions(self, directions, ids, vecs, v, pivots)
                    # For trimolecular TSs, the alignment direction previously get is 
                    # just a general first approximation that needs to be corrected
                    # for the specific case through another algorithm.
                    
                for angles in systematic_angles:

                    threads.append([deepcopy(obj) for obj in thread_objects])
                    thread = threads[-1]
                    # generating the thread we are going to modify and setting the thread pointer

                    constrained_indexes.append(ids)
                    # Save indexes to be constrained later in the optimization step

                    # orb_list = []

                    for i, vec_pair in enumerate(vecs):
                    # setting molecular positions and rotations (embedding)
                    # i is the molecule index, vecs is a tuple of start and end positions
                    # for the pivot vector

                        start, end = vec_pair
                        angle = angles[i]

                        reactive_coords = thread_objects[i].atomcoords[0][thread_objects[i].reactive_indexes]
                        # coordinates for the reactive atoms in this run

                        atomic_pivot_mean = np.mean(reactive_coords, axis=0)
                        # mean position of the atoms active in this run 

                        mol_direction = pivots[i].meanpoint-atomic_pivot_mean
                        if np.all(mol_direction == 0.):
                            mol_direction = pivots[i].meanpoint
                            # log.write(f'mol {i} - improper pivot? Thread {len(threads)-1}\n')

                        # Direction in which the molecule should be oriented, based on the meand of reactive
                        # atom positions and the mean point of the active pivot for the run.
                        # If this vector is too small and gets rounded to zero (as it can happen for
                        # "antrafacial" vectors), we fallback to the vector starting from the molecule
                        # center (mean of atomic positions) and ending in pivot_means[i], so to avoid
                        # numeric errors in the next function.
                            
                        alignment_rotation = R.align_vectors((end-start, directions[i]),
                                                             (pivots[i].pivot, mol_direction))[0].as_matrix()
                        # this rotation superimposes the molecular orbitals active in this run (pivots[i].pivot
                        # goes to end-start) and also aligns the molecules so that they face each other
                        # (mol_direction goes to directions[i])
                        
                        if len(reactive_coords) == 2:
                            axis_of_step_rotation = alignment_rotation @ (reactive_coords[0]-reactive_coords[1])
                        else:
                            axis_of_step_rotation = alignment_rotation @ pivots[i].pivot
                        # molecules with two reactive atoms are step-rotated around the line connecting
                        # the reactive atoms, while single reactive atom mols around their active pivot

                        step_rotation = rot_mat_from_pointer(axis_of_step_rotation, angle)
                        # this rotation cycles through all different rotation angles for each molecule

                        center_of_rotation = alignment_rotation @ atomic_pivot_mean
                        # center_of_rotation is the mean point between the reactive atoms so
                        # as to keep the reactive distances constant

                        thread[i].rotation = step_rotation @ alignment_rotation# @ pre_alignment_rot
                        # overall rotation for the molecule is given by the matrices product

                        pos = np.mean(vec_pair, axis=0) - alignment_rotation @ pivots[i].meanpoint
                        thread[i].position += center_of_rotation - step_rotation @ center_of_rotation + pos
                        # overall position is given by superimposing mean of active pivot (connecting orbitals)
                        # to mean of vec_pair (defining the target position - the side of a triangle for three molecules)

                        # orb_list.append(start)
                        # orb_list.append(end)

                    ################# DEBUGGING OUTPUT

                    # with open('orbs.xyz', 'a') as f:
                    #     an = np.array([3 for _ in range(len(orb_list))])
                    #     write_xyz(np.array(orb_list), an, f)

                    # totalcoords = np.concatenate([[mol.rotation @ v + mol.position for v in mol.atomcoords[0]] for mol in thread])
                    # totalatomnos = np.concatenate([mol.atomnos for mol in thread])
                    # total_reactive_indexes = np.array(ids).ravel()

                    # with open('debug_step.xyz', 'w') as f:
                    #     write_xyz(totalcoords, totalatomnos, f)
                    #     # write_xyz(totalcoords[total_reactive_indexes], totalatomnos[total_reactive_indexes], f)

                    # # os.system('vmd debug_step.xyz')
                    # os.system(r'vmd -e C:\Users\Nik\Desktop\Coding\TSCoDe\Resources\tri\temp_bonds.vmd')
                    # quit()
                    # TODO - clean


    loadbar(1, 1, prefix=f'Embedding structures ')

    self.dispositions_constrained_indexes = np.array(constrained_indexes)

    return threads

