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

import numpy as np
from parameters import orb_dim_dict
from utils import (
                   pt,
                   neighbors,
                   norm,
                   rot_mat_from_pointer,
                   vec_angle
                )


class Single:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'Single Bond'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = neighbors(mol.graph, i)


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.other = mol.atomcoords[atomcoords_index][neighbors_indexes][0]

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + str(self)
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = np.linalg.norm(self.coord - self.other)
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using the bonding distance ({round(orb_dim, 3)} A).')

            self.orb_vecs = np.array([norm(self.coord - self.other)])
            self.center = np.array([orb_dim * self.orb_vecs[0] + self.coord])


class Sp2:
    '''
    '''
    def __init__(self):
        pass

    def __repr__(self):
        return f'sp2'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = neighbors(mol.graph, i)
        


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.others = mol.atomcoords[atomcoords_index][neighbors_indexes]

        self.vectors = self.others - self.coord # vectors connecting reactive atom with neighbors
        self.orb_vec = norm(np.mean(np.array([np.cross(norm(self.vectors[0]), norm(self.vectors[1])),
                                              np.cross(norm(self.vectors[1]), norm(self.vectors[2])),
                                              np.cross(norm(self.vectors[2]), norm(self.vectors[0]))]), axis=0))

        self.orb_vecs = np.vstack((self.orb_vec, -self.orb_vec))

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + str(self)
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = orb_dim_dict['Fallback']
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')
            
            self.center = self.orb_vecs * orb_dim      

            self.center += self.coord


class Sp3:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'sp3'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = neighbors(mol.graph, i)
        


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.others = mol.atomcoords[atomcoords_index][neighbors_indexes]

        if not hasattr(self, 'leaving_group_index'):
            self.leaving_group_index = None

        if len([atom for atom in self.neighbors_symbols if atom in ['O', 'N', 'Cl', 'Br', 'I']]) == 1: # if we can tell where is the leaving group
            self.leaving_group_coords = self.others[self.neighbors_symbols.index([atom for atom in self.neighbors_symbols if atom in ['O', 'Cl', 'Br', 'I']][0])]

        elif len([atom for atom in self.neighbors_symbols if atom not in ['H']]) == 1: # if no clear leaving group but we only have one atom != H
            self.leaving_group_coords = self.others[self.neighbors_symbols.index([atom for atom in self.neighbors_symbols if atom not in ['H']][0])]

        else: # if we cannot infer, ask user if we didn't have already 
            try:
                self.leaving_group_coords = self._set_leaving_group(mol, neighbors_indexes)

            except Exception:
            # if something goes wrong, we fallback to command line input for reactive atom index collection

                if self.leaving_group_index is None:

                    while True:

                        self.leaving_group_index = input(f'Please insert the index of the leaving group atom bonded to the sp3 reactive atom (index {self.index}) of molecule {mol.rootname} : ')
                        
                        if self.leaving_group_index == '' or self.leaving_group_index.lower().islower():
                            pass
                        
                        elif int(self.leaving_group_index) in neighbors_indexes:
                            self.leaving_group_index = int(self.leaving_group_index)
                            break

                        else:
                            print(f'Atom {self.leaving_group_index} is not bonded to the sp3 center with index {self.index}.')
                
                self.leaving_group_coords = self.others[neighbors_indexes.index(self.leaving_group_index)]

        self.orb_vecs = np.array([self.coord - self.leaving_group_coords])
        self.orb_vers = norm(self.orb_vecs[0])

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + str(self)
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = orb_dim_dict['Fallback']
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')

            self.center = np.array([orb_dim * norm(self.orb_vecs[0]) + self.coord])

    def _set_leaving_group(self, mol, neighbors_indexes):
        '''
        Manually set the molecule leaving group from the ASE GUI, imposing
        a constraint on the desired atom.

        '''

        if self.leaving_group_index is None:

            from ase import Atoms
            from ase.gui.gui import GUI
            from ase.gui.images import Images

            atoms = Atoms(mol.atomnos, positions=mol.atomcoords[0])

            while True:
                print(('\nPlease, manually select the leaving group atom for molecule %s'
                    '\nbonded to the sp3 reactive atom with index %s.'
                    '\nRotate with right click and select atoms by clicking.'
                    '\nThen go to Tools -> Constraints -> Constrain, and close the GUI.') % (mol.name, self.index))

                GUI(images=Images([atoms]), show_bonds=True).run()
                
                if atoms.constraints != []:
                    if len(list(atoms.constraints[0].get_indices())) == 1:
                        if list(atoms.constraints[0].get_indices())[0] in neighbors_indexes:
                            self.leaving_group_index = list(atoms.constraints[0].get_indices())[0]
                            break
                        else:
                            print('\nSeems that the atom you selected is not bonded to the reactive center or is the reactive atom itself.\nThis is probably an error, please try again.')
                            atoms.constraints = []
                    else:
                        print('\nPlease only select one leaving group atom.')
                        atoms.constraints = []


        return self.others[neighbors_indexes.index(self.leaving_group_index)]


class Ether:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'Ether'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = neighbors(mol.graph, i)
        


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.others = mol.atomcoords[atomcoords_index][neighbors_indexes]

        self.orb_vecs = self.others - self.coord # vectors connecting center to each of the two substituents

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + str(self)
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = orb_dim_dict['Fallback']
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')

            self.orb_vecs = orb_dim * np.array([norm(v) for v in self.orb_vecs]) # making both vectors a fixed, defined length

            orb_mat = rot_mat_from_pointer(np.mean(self.orb_vecs, axis=0), 90) @ rot_mat_from_pointer(norm(np.cross(self.orb_vecs[0], self.orb_vecs[1])), 180)

            self.orb_vecs = np.array([orb_mat @ v for v in self.orb_vecs])
            
            self.center = self.orb_vecs + self.coord
            # two vectors defining the position of the two orbital lobes centers


class Ketone:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'Ketone'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = neighbors(mol.graph, i)
        


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.other = mol.atomcoords[atomcoords_index][neighbors_indexes][0]

        self.vector = self.other - self.coord # vector connecting center to substituent

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + str(self)
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = orb_dim_dict['Fallback']
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')

            neighbors_of_neighbor_indexes = neighbors(mol.graph, neighbors_indexes[0])
            neighbors_of_neighbor_indexes.remove(i)

            self.vector = norm(self.vector)*orb_dim

            if len(neighbors_of_neighbor_indexes) == 2:
            # if it is a normal ketone (or an enolate), n orbital lobes must be coplanar with
            # atoms connecting to ketone C atom. Then, also p lobes are added.

                a1 = mol.atomcoords[atomcoords_index][neighbors_of_neighbor_indexes[0]]
                a2 = mol.atomcoords[atomcoords_index][neighbors_of_neighbor_indexes[1]]
                pivot = norm(np.cross(a1 - self.coord, a2 - self.coord))

                self.center = np.array([rot_mat_from_pointer(pivot, angle) @ self.vector for angle in (120,240)])
                self.center = np.concatenate((self.center, [pivot*orb_dim], [-pivot*orb_dim]))

            elif len(neighbors_of_neighbor_indexes) in (1, 3):
            # ketene or alkoxide
            
                ketene_sub_indexes = neighbors(mol.graph, neighbors_of_neighbor_indexes[0])
                ketene_sub_indexes.remove(neighbors_indexes[0])

                ketene_sub_coords = mol.atomcoords[atomcoords_index][ketene_sub_indexes[0]]
                n_o_n_coords = mol.atomcoords[atomcoords_index][neighbors_of_neighbor_indexes[0]]

                # vector connecting ketene R with C (O=C=C(R)R)
                v = (ketene_sub_coords - n_o_n_coords)

                # this vector is orthogonal to the ketene O=C=C and coplanar with the ketene
                pointer = v - ((v @ norm(self.vector)) * self.vector)
                pointer = norm(pointer) * orb_dim

                self.center = np.array([rot_mat_from_pointer(self.vector, 90*step) @ pointer for step in range(4)])
            
        
            self.orb_vecs = np.array([norm(center) for center in self.center])
            # unit vectors connecting reactive atom coord with orbital centers

            self.center += self.coord
            # two vectors defining the position of the two orbital lobes centers


class Imine:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return 'Imine'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = neighbors(mol.graph, i)
        


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.others = mol.atomcoords[atomcoords_index][neighbors_indexes]

        self.vectors = self.others - self.coord # vector connecting center to substituent

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + str(self)
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = orb_dim_dict['Fallback']
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')
        
            # adding the first lobe (lone pair)
            self.orb_vecs = np.array([-norm(np.mean([norm(v) for v in self.vectors], axis=0))*orb_dim])

            # adding two p lobes
            p_lobe = norm(np.cross(self.vectors[0], self.vectors[1]))*orb_dim
            self.orb_vecs = np.concatenate((self.orb_vecs, [p_lobe], [-p_lobe]))

            self.center = self.orb_vecs + self.coord
            # two vectors defining the position of the two orbital lobes centers


class Sp_or_carbene:
    '''
    '''
    def __init__(self):
        pass
    
    def __repr__(self):
        return self.type

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = neighbors(mol.graph, i)
        
        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]

        self.coord = mol.atomcoords[atomcoords_index][i]
        self.others = mol.atomcoords[atomcoords_index][neighbors_indexes]

        self.vectors = self.others - self.coord # vector connecting center to substituent


        angle = vec_angle(norm(self.others[0] - self.coord),
                          norm(self.others[1] - self.coord))
        
        if np.abs(angle - 180) < 5:
            self.type = 'sp'

        else:
            self.type = 'bent carbene'

        self.allene = False
        self.ketene = False
        if self.type == 'sp' and all([s == 'C' for s in self.neighbors_symbols]):

            neighbors_of_neighbors_indexes = (neighbors(mol.graph, neighbors_indexes[0]),
                                              neighbors(mol.graph, neighbors_indexes[1]))

            neighbors_of_neighbors_indexes[0].remove(i)
            neighbors_of_neighbors_indexes[1].remove(i)

            if (len(side1) == len(side2) == 2 for side1, side2 in neighbors_of_neighbors_indexes):
                self.allene = True

        elif self.type == 'sp' and sorted(self.neighbors_symbols) in (['C', 'O'], ['C', 'S']):

            self.ketene = True

            neighbors_of_neighbors_indexes = (neighbors(mol.graph, neighbors_indexes[0]),
                                              neighbors(mol.graph, neighbors_indexes[1]))

            neighbors_of_neighbors_indexes[0].remove(i)
            neighbors_of_neighbors_indexes[1].remove(i)
                
            if len(neighbors_of_neighbors_indexes[0]) == 2:
                substituent = mol.atomcoords[atomcoords_index][neighbors_of_neighbors_indexes[0][0]]
                ketene_atom = mol.atomcoords[atomcoords_index][neighbors_indexes[0]]
                self.ketene_ref = substituent - ketene_atom

            elif len(neighbors_of_neighbors_indexes[1]) == 2:
                substituent = mol.atomcoords[atomcoords_index][neighbors_of_neighbors_indexes[1][0]]
                ketene_atom = mol.atomcoords[atomcoords_index][neighbors_indexes[1]]
                self.ketene_ref = substituent - ketene_atom

            else:
                self.ketene = False

        if update:
            if orb_dim is None:
                key = self.symbol + ' ' + self.type
                try:
                    orb_dim = orb_dim_dict[key]
                except KeyError:
                    orb_dim = orb_dim_dict['Fallback']
                    print(f'ATTENTION: COULD NOT SETUP REACTIVE ATOM ORBITAL FROM PARAMETERS. We have no parameters for {key}. Using {orb_dim} A.')
        
            if self.type == 'sp':

                v = np.random.rand(3)
                pivot1 = v - ((v @ norm(self.vectors[0])) * self.vectors[0])

                if self.allene or self.ketene:
                    # if we have an allene or ketene, pivot1 is aligned to
                    # one substituent so that the resulting positions
                    # for the four orbital centers make chemical sense.

                    axis = norm(self.others[0] - self.others[1])
                    # versor connecting reactive atom neighbors
                    
                    if self.allene:
                        ref = (mol.atomcoords[atomcoords_index][neighbors_of_neighbors_indexes[0][0]] -
                               mol.atomcoords[atomcoords_index][neighbors_indexes[0]])
                    else:
                        ref = self.ketene_ref

                    pivot1 = ref - ref @ axis * axis
                    # projection of ref orthogonal to axis (vector rejection)


                pivot2 = norm(np.cross(pivot1, self.vectors[0]))
                        
                self.orb_vecs = np.array([rot_mat_from_pointer(pivot2, 90) @
                                          rot_mat_from_pointer(pivot1, angle) @
                                          norm(self.vectors[0]) for angle in (0, 90, 180, 270)]) * orb_dim

                self.center = self.orb_vecs + self.coord
                # four vectors defining the position of the four orbital lobes centers



            else: # bent carbene case: three centers, sp2+p
                
                self.orb_vecs = np.array([-norm(np.mean([norm(v) for v in self.vectors], axis=0))*orb_dim])
                # one sp2 center first

                p_vec = np.cross(norm(self.vectors[0]), norm(self.vectors[1]))
                p_vecs = np.array([norm(p_vec)*orb_dim, -norm(p_vec)*orb_dim])
                self.orb_vecs = np.concatenate((self.orb_vecs, p_vecs))
                # adding two p centers

                self.center = self.orb_vecs + self.coord
                # three vectors defining the position of the two p lobes and main sp2 lobe centers


class Metal:
    '''
    '''

    def __init__(self):
        pass

    def __repr__(self):
        return 'Metal'

    def init(self, mol, i, update=False, orb_dim=None, atomcoords_index=0) -> None:
        '''
        '''
        self.index = i
        self.symbol = pt[mol.atomnos[i]].symbol
        neighbors_indexes = neighbors(mol.graph, i)
        


        self.neighbors_symbols = [pt[mol.atomnos[i]].symbol for i in neighbors_indexes]
        self.coord = mol.atomcoords[atomcoords_index][i]
        self.others = mol.atomcoords[atomcoords_index][neighbors_indexes]

        self.vectors = self.others - self.coord # vectors connecting reactive atom with neighbors

        v1 = self.vectors[0]

        neighbor_of_neighbor_index = neighbors(mol.graph, neighbors_indexes[0])[0]
        v2 = mol.atomcoords[atomcoords_index][neighbor_of_neighbor_index] - self.coord

        self.orb_vec = norm(rot_mat_from_pointer(np.cross(v1, v2), 120) @ v1)

        steps = 4
        self.orb_vecs = np.array([rot_mat_from_pointer(v1, angle) @ self.orb_vec for angle in range(0,360,int(360/steps))])

        if update:
            if orb_dim is None:
                orb_dim = orb_dim_dict[str(self)]
            
            self.center = (self.orb_vecs * orb_dim) + self.coord

# class Sigma_bond: # when the nucleophile is a sigma bond!

# Keys are made of atom symbol and number of bonds that it makes
atom_type_dict = {
            'H1' : Single(),

            'B3' : Sp2(),
            # 'B4' : This is probably hard and would require a complex reactive atom class

            'C1' : Single(), # deprotonated terminal alkyne. TODO - what if it is a carbylidene? Very rare by the way...
            'C2' : Sp_or_carbene(), # sp if straight, carbene if bent
            'C3' : Sp2(), # double ball
            'C4' : Sp3(), # one ball, on the back of the leaving group. If we can't tell which one it is, we ask user

            'N1' : Single(),
            'N2' : Imine(), # one ball on free side
            'N3' : Sp2(), # double ball
            'N4' : Sp3(), # leaving group

            'O1' : Ketone(), # two balls 120° apart. Also for alkoxides, good enough
            'O2' : Ether(), # or alcohol, two balls about 109,5° apart

            'P2' : Imine(), # one ball on free side
            'P3' : Sp2(), # double ball
            'P4' : Sp3(), # leaving group

            'S1' : Ketone(),
            'S2' : Ether(),
            'S3' : Sp2(), # Not sure if this can be valid, but it's basically treating it as a bent carbonyl, should work
            #  'S3' : Sulphoxide(), # Should we consider this? Or just ok with Sp2()?
            # 'S4' : Sulphone(),

            'F1' : Single(),
            'Cl1': Single(),
            'Br1': Single(),
            'I1' : Single(),
             }

metals = (
    'Li',
    'Na',
    'Mg',
    'K',
    'Ca',
    'Rb',
    'Sr',
    'Cs',
    'Ba',
    'Zn',
)

for metal in metals:
    for bonds in range(1,9):
        bonds = str(bonds)
        atom_type_dict[metal+bonds] = Metal()